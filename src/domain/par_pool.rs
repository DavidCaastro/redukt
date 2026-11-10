//! Persistent worker pool for row-partitioned kernels (R-098).
//!
//! Workers are spawned once (lazily, process-global) and wait for jobs with a
//! short spin followed by `thread::park`. A job is a function over block
//! indices `0..n_blocks`. Blocks are claimed with a CAS on one packed atomic
//! word (`epoch:32 | n_blocks:16 | next:16`), so fast cores simply claim more
//! blocks (self-balancing across P/E cores). The submitting thread works too
//! and returns only after every block has completed.
//!
//! Determinism: the pool decides WHICH thread computes a block, never HOW.
//! Callers partition by output rows and each row is computed by the same
//! single-thread kernel, so results are bit-identical for any thread count.
//!
//! Concurrency: one job in flight at a time. A second concurrent submitter
//! (batch `--jobs`, API server, parallel tests) runs its blocks inline on its
//! own thread — same results, no contention, no deadlock.
//!
//! Configuration: `REDUKT_THREADS=N` (total threads incl. caller, default
//! `available_parallelism`), `REDUKT_SPIN_US=N` (worker spin before park,
//! default 100 µs).

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// Maximum blocks per job (16-bit field in the packed control word).
pub const MAX_BLOCKS: usize = 0xFFFF;
const DEFAULT_SPIN_US: u64 = 100;

/// Job descriptor living on the submitter's stack for the job's duration.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
}

struct Shared {
    /// `epoch:32 | n_blocks:16 | next:16`.
    word: AtomicU64,
    /// `*const Job` of the current epoch. Dereferenced only after a
    /// successful block claim (the job cannot complete before that block).
    job: AtomicPtr<()>,
    done: AtomicUsize,
    panicked: AtomicBool,
    parked: Vec<AtomicBool>,
    spin: Duration,
}

/// Persistent GEMV worker pool. Obtain the process-wide instance via [`global`].
pub struct Pool {
    shared: &'static Shared,
    workers: Vec<Thread>,
    submit: Mutex<()>,
}

#[inline]
fn pack(epoch: u32, n: usize, next: usize) -> u64 {
    ((epoch as u64) << 32) | ((n as u64) << 16) | next as u64
}

#[inline]
fn unpack(w: u64) -> (u32, usize, usize) {
    ((w >> 32) as u32, ((w >> 16) & 0xFFFF) as usize, (w & 0xFFFF) as usize)
}

/// Thread count from `REDUKT_THREADS` or `available_parallelism` (min 1).
pub fn configured_threads() -> usize {
    let env = std::env::var("REDUKT_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
    match env {
        Some(n) if n >= 1 => n.min(256),
        _ => thread::available_parallelism().map(|p| p.get()).unwrap_or(1),
    }
}

fn configured_spin() -> Duration {
    let us = std::env::var("REDUKT_SPIN_US").ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_SPIN_US);
    Duration::from_micros(us)
}

/// Process-wide pool, created on first use.
pub fn global() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(configured_threads(), configured_spin()))
}

impl Pool {
    /// Create a pool with `n_threads` total threads (caller + `n_threads-1` workers).
    /// Worker threads are detached and live for the rest of the process.
    pub fn new(n_threads: usize, spin: Duration) -> Pool {
        let n_workers = n_threads.max(1) - 1;
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            word: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            parked: (0..n_workers).map(|_| AtomicBool::new(false)).collect(),
            spin,
        }));
        let mut workers = Vec::with_capacity(n_workers);
        for idx in 0..n_workers {
            let spawned = thread::Builder::new()
                .name(format!("redukt-pool-{}", idx))
                .spawn(move || worker_loop(shared, idx));
            match spawned {
                Ok(h) => workers.push(h.thread().clone()),
                Err(_) => break, // fewer workers is still correct
            }
        }
        Pool { shared, workers, submit: Mutex::new(()) }
    }

    /// Total threads that participate in a job (workers + caller).
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(b)` for every `b in 0..n_blocks`, distributed over the pool.
    /// Returns after all blocks complete. Falls back to inline execution when
    /// the pool has no workers, another job is in flight, or `n_blocks` is
    /// out of range.
    pub fn run(&self, n_blocks: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() || n_blocks <= 1 || n_blocks > MAX_BLOCKS {
            (0..n_blocks).for_each(f);
            return;
        }
        let _guard = match self.submit.try_lock() {
            Ok(g) => g,
            // A panicking job poisons the lock; the pool state is still valid.
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
            Err(TryLockError::WouldBlock) => {
                (0..n_blocks).for_each(f);
                return;
            }
        };
        let sh = self.shared;
        let job = Job { f };
        sh.done.store(0, Ordering::Relaxed);
        sh.panicked.store(false, Ordering::Relaxed);
        sh.job.store(&job as *const Job as *mut (), Ordering::Release);
        let (old_epoch, _, _) = unpack(sh.word.load(Ordering::Relaxed));
        let epoch = old_epoch.wrapping_add(1);
        sh.word.store(pack(epoch, n_blocks, 0), Ordering::SeqCst);
        for (i, t) in self.workers.iter().enumerate() {
            if sh.parked[i].load(Ordering::SeqCst) {
                t.unpark();
            }
        }
        let own_panic = claim_and_run(sh, epoch, &job as *const Job);
        while sh.done.load(Ordering::Acquire) < n_blocks {
            std::hint::spin_loop();
        }
        // All blocks finished: no worker can touch `job` any more.
        if let Some(payload) = own_panic {
            resume_unwind(payload);
        }
        if sh.panicked.load(Ordering::Acquire) {
            resume_unwind(Box::new("redukt-pool: worker block panicked"));
        }
    }
}

type PanicPayload = Box<dyn std::any::Any + Send + 'static>;

/// Claim blocks of `epoch` until none remain; run each one.
/// Returns the first panic payload raised on this thread, if any.
///
/// `job` is a RAW pointer on purpose: it may already be dangling when a late
/// worker arrives (job finished, submitter returned — possibly on a thread
/// whose stack is gone). A `&Job` would let the compiler load through it
/// speculatively; the raw pointer is read only after a successful claim.
fn claim_and_run(sh: &Shared, epoch: u32, job: *const Job) -> Option<PanicPayload> {
    let mut first_panic = None;
    loop {
        let w = sh.word.load(Ordering::Acquire);
        let (ep, n, next) = unpack(w);
        if ep != epoch || next >= n {
            return first_panic;
        }
        if sh.word.compare_exchange_weak(w, w + 1, Ordering::AcqRel, Ordering::Acquire).is_err() {
            continue;
        }
        // SAFETY: the claim (AcqRel CAS) succeeded for an unfinished block of
        // the live epoch, so the submitter is still blocked in `run` and the
        // Job (and the closure it points to) is alive until this block's
        // `done` increment below.
        let f = unsafe { (*job).f };
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| f(next))) {
            sh.panicked.store(true, Ordering::Release);
            first_panic.get_or_insert(p);
        }
        sh.done.fetch_add(1, Ordering::Release);
    }
}

fn worker_loop(sh: &'static Shared, idx: usize) {
    let mut seen = 0u32;
    loop {
        seen = wait_new_epoch(sh, idx, seen);
        // Published before epoch `seen`; may be stale (see claim_and_run).
        let ptr = sh.job.load(Ordering::Acquire) as *const Job;
        if ptr.is_null() {
            continue;
        }
        // Worker panics are reported through `panicked`; payload is dropped.
        drop(claim_and_run(sh, seen, ptr));
    }
}

/// Spin, then park, until the epoch differs from `seen`. Returns the new epoch.
fn wait_new_epoch(sh: &Shared, idx: usize, seen: u32) -> u32 {
    let start = Instant::now();
    let mut iter: u32 = 0;
    loop {
        let (ep, _, _) = unpack(sh.word.load(Ordering::Acquire));
        if ep != seen {
            return ep;
        }
        iter = iter.wrapping_add(1);
        if iter % 256 != 0 || start.elapsed() < sh.spin {
            std::hint::spin_loop();
            continue;
        }
        sh.parked[idx].store(true, Ordering::SeqCst);
        let (ep2, _, _) = unpack(sh.word.load(Ordering::SeqCst));
        if ep2 == seen {
            thread::park();
        }
        sh.parked[idx].store(false, Ordering::SeqCst);
    }
}

/// Raw pointer wrapper for disjoint writes from pool blocks.
/// Callers guarantee each block writes a disjoint range.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(*mut T);

// SAFETY: used only for disjoint per-block writes coordinated by Pool::run;
// `T: Send` because each block hands its `&mut [T]` range to another thread.
unsafe impl<T: Send> Send for SyncPtr<T> {}
// SAFETY: as above — shared `&SyncPtr` only yields disjoint ranges.
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(p: *mut T) -> Self {
        SyncPtr(p)
    }

    /// Mutable sub-slice `[start, start+len)`.
    ///
    /// # Safety
    /// The range must be in bounds of the original allocation and must not
    /// overlap any range handed out to a concurrently running block.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut<'a>(&self, start: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn pack_unpack_roundtrip() {
        let w = pack(0xDEAD_BEEF, 1234, 77);
        assert_eq!(unpack(w), (0xDEAD_BEEF, 1234, 77));
        assert_eq!(unpack(w + 1), (0xDEAD_BEEF, 1234, 78));
    }

    #[test]
    fn every_block_runs_exactly_once() {
        let pool = Pool::new(4, Duration::from_micros(50));
        for &n in &[2usize, 3, 17, 64, 1000] {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            pool.run(n, &|b| { hits[b].fetch_add(1, Ordering::Relaxed); });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={}", n);
        }
    }

    #[test]
    fn many_jobs_with_parking_between() {
        // spin=0 forces workers to park between jobs (exercises unpark path).
        let pool = Pool::new(3, Duration::from_micros(0));
        let total = AtomicUsize::new(0);
        for _ in 0..500 {
            pool.run(8, &|b| { total.fetch_add(b + 1, Ordering::Relaxed); });
        }
        assert_eq!(total.load(Ordering::Relaxed), 500 * 36);
    }

    #[test]
    fn concurrent_submitters_fall_back_inline() {
        let pool = Pool::new(4, Duration::from_micros(100));
        let total = AtomicUsize::new(0);
        thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..100 {
                        pool.run(16, &|_| { total.fetch_add(1, Ordering::Relaxed); });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 4 * 100 * 16);
    }

    #[test]
    fn single_thread_pool_runs_inline() {
        let pool = Pool::new(1, Duration::from_micros(0));
        assert_eq!(pool.threads(), 1);
        let total = AtomicUsize::new(0);
        pool.run(10, &|b| { total.fetch_add(b, Ordering::Relaxed); });
        assert_eq!(total.load(Ordering::Relaxed), 45);
    }

    #[test]
    fn worker_panic_propagates_without_hanging() {
        let pool = Pool::new(3, Duration::from_micros(100));
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(32, &|b| { if b == 31 { panic!("boom"); } });
        }));
        assert!(r.is_err());
        // Pool remains usable after a panicking job.
        let total = AtomicUsize::new(0);
        pool.run(32, &|_| { total.fetch_add(1, Ordering::Relaxed); });
        assert_eq!(total.load(Ordering::Relaxed), 32);
    }
}
