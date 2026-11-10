//! API vs CLI byte parity (needs real Mamba weights in weights/mamba-130m).
//!
//! Run with: cargo test --release --test api_parity -- --ignored
//! Build the binary first (`cargo build --release`): the test drives `target/<profile>/redukt`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn redukt_bin() -> PathBuf {
    let mut path = std::env::current_exe().expect("current_exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push(format!("redukt{}", std::env::consts::EXE_SUFFIX));
    path
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind probe");
    l.local_addr().expect("local_addr").port()
}

/// Kills the server even when an assertion fails.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Minimal HTTP/1.1 client: returns (status, body).
fn http(port: u16, method: &str, path: &str, body: &[u8]) -> (u16, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(600))).expect("timeout");
    let head = format!(
        "{} {} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        method, path, body.len()
    );
    s.write_all(head.as_bytes()).expect("write head");
    s.write_all(body).expect("write body");
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read");
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("header end");
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).expect("status");
    (status, raw[split + 4..].to_vec())
}

fn wait_ready(server: &mut Server, port: u16) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(300) {
        if let Ok(Some(st)) = server.0.try_wait() {
            panic!("api exited early: {}", st);
        }
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!("api did not start in time");
}

fn sample_4kb() -> Vec<u8> {
    let line = "192.168.1.10 - - [10/Nov/2026:10:00:01 +0000] \"GET /index.html HTTP/1.1\" 200 512\n";
    line.repeat(4096 / line.len() + 1)[..4096].as_bytes().to_vec()
}

#[test]
#[ignore] // requires weights/mamba-130m
fn api_compress_matches_cli_bytes_and_roundtrips() {
    if !Path::new("weights/mamba-130m").exists() {
        eprintln!("SKIP: no mamba weights");
        return;
    }
    let bin = redukt_bin();
    assert!(bin.exists(), "binary not found at {:?} — run `cargo build --release`", bin);
    let dir = std::env::temp_dir().join(format!("redukt-api-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let input = dir.join("in.log");
    let cli_out = dir.join("cli.azth");
    let data = sample_4kb();
    std::fs::write(&input, &data).expect("write input");

    let st = Command::new(&bin)
        .args(["compress", "--input"]).arg(&input).arg("--output").arg(&cli_out)
        .args(["--backbone", "mamba", "--quant", "q8"])
        .stderr(Stdio::null()).status().expect("run cli");
    assert!(st.success(), "cli compress failed");
    let cli_bytes = std::fs::read(&cli_out).expect("cli output");

    let port = free_port();
    let child = Command::new(&bin)
        .args(["api", "--bind", "127.0.0.1", "--port", &port.to_string()])
        .args(["--backbone", "mamba", "--quant", "q8"])
        .stderr(Stdio::null()).spawn().expect("spawn api");
    let mut server = Server(child);
    wait_ready(&mut server, port);

    let (code, status) = http(port, "GET", "/v1/status", b"");
    assert_eq!(code, 200);
    let status = String::from_utf8(status).expect("utf8");
    assert!(status.contains("\"backbone\":\"mamba\"") && status.contains("\"profile\":\"Q8\""), "{}", status);

    let (code, api_bytes) = http(port, "POST", "/v1/compress", &data);
    assert_eq!(code, 200);
    assert_eq!(api_bytes, cli_bytes, "API archive differs from CLI archive");

    let (code, restored) = http(port, "POST", "/v1/decompress", &cli_bytes);
    assert_eq!(code, 200);
    assert_eq!(restored, data, "API decompress of CLI archive differs from input");

    // An archive from a profile that is not loaded is a clear 4xx, not a panic.
    let mut f32_hdr = cli_bytes.clone();
    f32_hdr[4] = 9; // unsupported version -> 400
    let (code, msg) = http(port, "POST", "/v1/decompress", &f32_hdr);
    assert_eq!(code, 400, "{}", String::from_utf8_lossy(&msg));
    let _ = std::fs::remove_dir_all(&dir);
}
