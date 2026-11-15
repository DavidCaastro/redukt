//! Speed PROXY for Strata tier `cm` (width 0). NOT the specified arithmetic and NOT bit-compatible with it.
//!
//! It performs the per-byte work of specification 2.5 / 3.1 / 3.3 with the v1 default table sizes
//! (8 successor orders, table bits 8/16/16/16/18/18/18/18; 4 bit-cell orders, bits 8/16/22/22; 13 mixer
//! inputs, weight set (bit position, top order); calibration cells; a binary range coder), so that the
//! memory-access pattern and the integer work per byte are those of the tier. Tables `ST` / `SQ`, boot
//! weights and the order lengths are stand-ins, so its code length is only a sanity check.
//!
//! Build: rustc -C opt-level=3 -C target-cpu=x86-64-v3 cm_speed_proxy.rs
//! Run:   cm_speed_proxy <file> [threads] [repeats]
use std::time::Instant;

const NORD: usize = 8;
const LEN: [usize; NORD] = [1, 2, 3, 4, 5, 6, 8, 12];
const TB: [u32; NORD] = [8, 16, 16, 16, 18, 18, 18, 18];
const NBC: usize = 4;
const BTB: [u32; NBC] = [8, 16, 22, 22];
const N_IN: usize = 1 + NORD + NBC;
const MIX_SH: i64 = 14;

fn squash(st: i32) -> i32 {
    // logistic of st / 256 nats, scaled to 1 .. 65535
    let p = 65536.0 / (1.0 + (-(st as f64) / 256.0).exp());
    (p as i32).clamp(1, 65535)
}

fn stretch_table() -> Vec<i32> {
    // index = q >> 10 for q in 0 .. 2^22  ->  4096 entries, ln(p / (1 - p)) * 256
    (0..4096).map(|i| {
        let p = (i as f64 + 0.5) / 4096.0;
        (((p / (1.0 - p)).ln() * 256.0) as i32).clamp(-3071, 3071)
    }).collect()
}

fn qshift(n: u8) -> u32 {
    (32 - (n as u32 + 2).leading_zeros()).clamp(2, 7)
}

struct Coder { low: u32, range: u32, out: u64 }

impl Coder {
    fn encode(&mut self, bit: u32, p1: u32) {
        let bound = ((self.range as u64 * p1 as u64) >> 16) as u32;
        if bit == 1 { self.range = bound; } else { self.low = self.low.wrapping_add(bound); self.range -= bound; }
        while self.range < (1 << 24) {
            self.out += 1;
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

fn ctx_hash(hist: &[u8], len: usize, tb: u32) -> (usize, u8) {
    let n = hist.len();
    match len {
        1 => (hist[n - 1] as usize, 0),
        2 => (((hist[n - 2] as usize) << 8) | hist[n - 1] as usize, 0),
        _ => {
            let mut h: u32 = 0;
            for &c in &hist[n - len..] { h = h.wrapping_add(c as u32 + 1).wrapping_mul(0x9E37_79B1); }
            h ^= h >> 15;
            h = h.wrapping_mul(0x85EB_CA6B);
            h ^= h >> 13;
            ((h >> (32 - tb)) as usize, (h & 0xFF) as u8)
        }
    }
}

/// Codes `data` once; returns (coder bytes, ideal bits).
fn run(data: &[u8], st: &[i32], sq: &[i32]) -> (u64, f64) {
    let mut t: Vec<Vec<u32>> = TB.iter().map(|&b| vec![0u32; 1 << b]).collect();
    let mut qb: Vec<Vec<u32>> = BTB.iter().map(|&b| vec![(1u32 << 21) << 8; 1 << b]).collect();
    let mut q = vec![(3u32 << 20) << 8; NORD * 32 * 8];
    let nset = 8 * (NORD + 1);
    let mut w = vec![0i32; nset * N_IN];
    for s in 0..nset { for m in 0..N_IN { w[s * N_IN + m] = 1 << 14; } }
    let mut trun = [0u8; NORD];
    let mut coder = Coder { low: 0, range: u32::MAX, out: 0 };
    let mut bits = 0.0f64;
    let mut x = [0i32; N_IN];
    for n in 0..data.len() {
        let v = data[n];
        let hist = &data[..n];
        let (c1, c2, c3) = (
            if n >= 1 { data[n - 1] as u32 } else { 0 },
            if n >= 2 { data[n - 2] as u32 } else { 0 },
            if n >= 3 { data[n - 3] as u32 } else { 0 },
        );
        // LOOKUP
        let mut slot = [0usize; NORD];
        let mut check = [0u8; NORD];
        let mut pred = [-1i32; NORD];
        let mut run_i = [0u32; NORD];
        let mut conf = [0usize; NORD];
        for i in 0..NORD {
            if n < LEN[i] { continue; }
            let (s, c) = ctx_hash(hist, LEN[i], TB[i]);
            slot[i] = s;
            check[i] = c;
            let e = t[i][s];
            if e >> 31 == 1 && ((e >> 16) & 0xFF) as u8 == c {
                pred[i] = (e & 0xFF) as i32;
                run_i[i] = (e >> 8) & 3;
            }
            let tl = (8 - trun[i].leading_zeros()).min(7);
            conf[i] = (8 * run_i[i] + tl) as usize;
        }
        let h2 = (c2.wrapping_mul(0x9E37_79B1).wrapping_add(c1)).wrapping_mul(0x85EB_CA6B);
        let h3 = (c3.wrapping_mul(0x9E37_79B1).wrapping_add(c2)).wrapping_mul(0x9E37_79B1).wrapping_add(c1)
            .wrapping_mul(0x85EB_CA6B);
        let base = [0usize, (c1 as usize) << 8, ((h2 >> 18) as usize) << 8, ((h3 >> 18) as usize) << 8];
        // BIT STEPS
        for j in 0..8usize {
            let p = (v as usize) >> (8 - j);
            let node = (1usize << j) | p;
            let b = ((v >> (7 - j)) & 1) as i32;
            x[0] = 0;
            let mut top = 0usize;
            let mut live = [false; NORD];
            for i in 0..NORD {
                x[i + 1] = 0;
                if pred[i] >= 0 && ((pred[i] as usize) >> (8 - j)) == p {
                    let eb = (pred[i] >> (7 - j)) & 1;
                    let cell = q[(i * 32 + conf[i]) * 8 + j];
                    x[i + 1] = (2 * eb - 1) * st[(cell >> 18) as usize];
                    top = i + 1;
                    live[i] = true;
                }
            }
            let mut idx = [0usize; NBC];
            for m in 0..NBC {
                idx[m] = (base[m] | node) & ((1usize << BTB[m]) - 1);
                x[1 + NORD + m] = st[(qb[m][idx[m]] >> 18) as usize] - 0;
            }
            let set = (j * (NORD + 1) + top) * N_IN;
            let ws = &mut w[set..set + N_IN];
            let mut dot: i64 = 0;
            for m in 0..N_IN { dot += ws[m] as i64 * x[m] as i64; }
            let stv = ((dot >> 16) as i32).clamp(-3071, 3071);
            let p1 = sq[(stv + 3071) as usize];
            coder.encode(b as u32, p1 as u32);
            bits -= (if b == 1 { p1 as f64 } else { 65536.0 - p1 as f64 } / 65536.0).log2();
            // UPDATE
            let err = ((b << 16) - p1) as i64;
            for m in 0..N_IN {
                let d = (x[m] as i64 * err + (1 << (MIX_SH - 1))) >> MIX_SH;
                ws[m] = (ws[m] as i64 + d).clamp(-(1 << 20), 1 << 20) as i32;
            }
            for i in 0..NORD {
                if !live[i] { continue; }
                let eb = (pred[i] >> (7 - j)) & 1;
                let c = &mut q[(i * 32 + conf[i]) * 8 + j];
                let (qv, nv) = ((*c >> 8) as i32, (*c & 0xFF) as u8);
                let target = if eb == b { 1 << 22 } else { 0 };
                let qn = qv + ((target - qv) >> qshift(nv));
                *c = ((qn as u32) << 8) | nv.saturating_add(1) as u32;
            }
            for m in 0..NBC {
                let c = &mut qb[m][idx[m]];
                let (qv, nv) = ((*c >> 8) as i32, (*c & 0xFF) as u8);
                let qn = qv + (((b << 22) - qv) >> qshift(nv));
                *c = ((qn as u32) << 8) | nv.saturating_add(1) as u32;
            }
        }
        // COMMIT
        for i in 0..NORD {
            if n < LEN[i] { trun[i] = 0; continue; }
            let hit = pred[i] == v as i32;
            let r = if hit { (run_i[i] + 1).min(3) } else { 0 };
            t[i][slot[i]] = (1 << 31) | ((check[i] as u32) << 16) | (r << 8) | v as u32;
            trun[i] = if hit { trun[i].saturating_add(1) } else { 0 };
        }
    }
    (coder.out, bits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read input");
    let threads: usize = args.get(2).map_or(1, |s| s.parse().unwrap());
    let repeats: usize = args.get(3).map_or(3, |s| s.parse().unwrap());
    let st = stretch_table();
    let sq: Vec<i32> = (-3071..=3071).map(squash).collect();
    let mut best = f64::MAX;
    let mut res = (0u64, 0.0f64);
    for _ in 0..repeats {
        let t0 = Instant::now();
        let out: Vec<(u64, f64)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| run(&data, &st, &sq))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        best = best.min(t0.elapsed().as_secs_f64());
        res = out[0];
    }
    let n = data.len() as f64;
    println!(
        "{} bytes x {} threads: {:.3} us/byte per stream, {:.0} KB/s per thread, {:.0} KB/s total; \
         coder {} B ({:.4} bpb), ideal {:.4} bpb",
        data.len(), threads, best * 1e6 / n, n / best / 1e3, n * threads as f64 / best / 1e3,
        res.0, res.0 as f64 * 8.0 / n, res.1 / n
    );
}
