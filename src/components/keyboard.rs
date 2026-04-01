//! # Keyboard Component
//!
//! Converts host input events into OC signals (`key_down`, `key_up`, `clipboard`).
//! Mirrors `Keyboard.scala`.

use crate::machine::Machine;
use crate::machine::signal::Signal;
use crate::components::Address;

/// Keyboard component.
pub struct Keyboard {
    pub address: Address,
}

impl Keyboard {
    pub fn new() -> Self {
        Self { address: crate::components::new_address() }
    }

    pub const fn component_name() -> &'static str { "keyboard" }

    /// Called when the host detects a key press.
    ///
    /// `char_code` is the Unicode character (0 for non-printable keys),
    /// `key_code` is the platform scan code.
    pub fn key_down(&self, machine: &mut Machine, char_code: u16, key_code: u32) {
        machine.push_signal(
            Signal::new("key_down")
                .with_string(self.address.clone())
                .with_int(char_code as i64)
                .with_int(key_code as i64)
        );
    }

    /// Called when the host detects a key release.
    pub fn key_up(&self, machine: &mut Machine, char_code: u16, key_code: u32) {
        machine.push_signal(
            Signal::new("key_up")
                .with_string(self.address.clone())
                .with_int(char_code as i64)
                .with_int(key_code as i64)
        );
    }

    /// Called when the host pastes text from clipboard.
    pub fn clipboard(&self, machine: &mut Machine, text: &str) {
        // OC sends clipboard line-by-line.
        for line in text.lines() {
            machine.push_signal(
                Signal::new("clipboard")
                    .with_string(self.address.clone())
                    .with_string(line.to_owned())
            );
        }
    }
}