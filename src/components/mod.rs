//! # Component subsystem

pub mod gpu;
pub mod eeprom;
pub mod keyboard;
pub mod screen;
pub mod filesystem;

use std::collections::HashMap;

pub type Address = String;

/// Generate a random UUID v4 address.
pub fn new_address() -> Address {
    use std::fmt::Write;
    let mut buf = String::with_capacity(36);
    let bytes: [u8; 16] = std::array::from_fn(|_| fastrand_byte());
    for (i, b) in bytes.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) { buf.push('-'); }
        let _ = write!(buf, "{:02x}", b);
    }
    buf
}

fn fastrand_byte() -> u8 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64 | 1
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s.set(x);
        x as u8
    })
}