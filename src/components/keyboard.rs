//! # Keyboard Component
//!
//! Converts host-side input events into OpenComputers signals that the
//! Lua guest can process. The three signal types produced are:
//!
//! * `key_down` - A key was pressed.
//! * `key_up` - A key was released.
//! * `clipboard` - Text was pasted from the host clipboard.
//!
//! ## Relationship to OpenComputers
//!
//! Mirrors `li.cil.oc.common.tileentity.traits.Keyboard` and
//! `li.cil.oc.server.component.Keyboard` from the Scala source.
//!
//! In-game, the keyboard is part of the screen block. When a player
//! focuses the screen, key events are forwarded to the computer as
//! signals. This emulator uses the host window's keyboard events
//! instead.
//!
//! ## Signal format
//!
//! ### `key_down` / `key_up`
//!
//! ```text
//! signal_name, keyboard_address, char_code, key_code
//! ```
//!
//! * `keyboard_address` - UUID of this keyboard component.
//! * `char_code` - Unicode code point of the character (0 for
//!   non-printable keys like F1, arrows, etc.).
//! * `key_code` - LWJGL2 scan code (not the host platform's native
//!   scan code). The emulator translates from winit key codes to
//!   LWJGL2 codes in `main.rs`.
//!
//! ### `clipboard`
//!
//! ```text
//! "clipboard", keyboard_address, line_text
//! ```
//!
//! Clipboard paste is split into individual lines. Each line generates
//! a separate `clipboard` signal, matching OC's behaviour.
//!
//! ## Key code mapping
//!
//! OpenComputers uses LWJGL2 key codes (inherited from Minecraft).
//! These are NOT the same as:
//! * Windows virtual key codes (VK_*)
//! * Linux/X11 keysyms
//! * USB HID usage codes
//!
//! The mapping from host key codes to LWJGL2 codes is done in
//! [`winit_to_lwjgl`](crate::winit_to_lwjgl) in `main.rs`.
//!
//! ## Thread safety
//!
//! Not `Sync`. Events are pushed from the main thread only.

use crate::machine::Machine;
use crate::machine::signal::Signal;
use crate::components::Address;

/// Keyboard component.
///
/// Stateless apart from its address. All keyboard state (which keys
/// are held, etc.) is managed by the host window system. This struct
/// simply converts events into signals.
pub struct Keyboard {
    /// Unique UUID address of this keyboard component.
    ///
    /// Included as the second argument in every keyboard signal,
    /// allowing Lua programs to distinguish multiple keyboards
    /// (relevant in multi-screen setups in OC, less so here).
    pub address: Address,
}

impl Keyboard {
    /// Create a new keyboard component with a fresh UUID address.
    pub fn new() -> Self {
        Self { address: crate::components::new_address() }
    }

    /// Returns the OC component type name: `"keyboard"`.
    pub const fn component_name() -> &'static str { "keyboard" }

    /// Generate a `key_down` signal from a host key press event.
    ///
    /// Called by the main event loop when a key is pressed while the
    /// emulator window is focused (and the settings GUI is not open).
    ///
    /// # Arguments
    ///
    /// * `machine` - The machine to push the signal into.
    /// * `char_code` - The Unicode character produced by the key press.
    ///   0 for non-printable keys (F-keys, arrows, modifiers).
    ///   For Enter, this is 13 (CR). For Tab, 9. For Backspace, 8.
    /// * `key_code` - The LWJGL2 scan code for the physical key.
    ///   Obtained by mapping the host key code through `winit_to_lwjgl`.
    ///
    /// # Signal format
    ///
    /// ```text
    /// ("key_down", keyboard_address: string, char_code: int, key_code: int)
    /// ```
    ///
    /// # Example Lua handler
    ///
    /// ```lua
    /// local _, _, char, code = event.pull("key_down")
    /// if char == string.byte("q") then os.exit() end
    /// ```
    pub fn key_down(&self, machine: &mut Machine, char_code: u16, key_code: u32) {
        machine.push_signal(
            Signal::new("key_down")
                .with_string(self.address.clone())
                .with_int(char_code as i64)
                .with_int(key_code as i64)
        );
    }

    /// Generate a `key_up` signal from a host key release event.
    ///
    /// Called by the main event loop when a key is released.
    ///
    /// # Arguments
    ///
    /// Same as [`key_down`](Keyboard::key_down).
    ///
    /// # Signal format
    ///
    /// ```text
    /// ("key_up", keyboard_address: string, char_code: int, key_code: int)
    /// ```
    pub fn key_up(&self, machine: &mut Machine, char_code: u16, key_code: u32) {
        machine.push_signal(
            Signal::new("key_up")
                .with_string(self.address.clone())
                .with_int(char_code as i64)
                .with_int(key_code as i64)
        );
    }

    /// Generate `clipboard` signals from pasted text.
    ///
    /// Called when the host detects a clipboard paste event (e.g.
    /// Ctrl+V on the emulator window).
    ///
    /// # Arguments
    ///
    /// * `machine` - The machine to push signals into.
    /// * `text` - The full pasted text. May contain multiple lines.
    ///
    /// # Behaviour
    ///
    /// The text is split by line boundaries (using Rust's
    /// `str::lines()`). Each line generates a separate `clipboard`
    /// signal, matching OpenComputers' behaviour where the keyboard
    /// component sends one signal per line.
    ///
    /// # Signal format (per line)
    ///
    /// ```text
    /// ("clipboard", keyboard_address: string, line_text: string)
    /// ```
    ///
    /// # Example
    ///
    /// Pasting `"hello\nworld"` generates two signals:
    /// 1. `("clipboard", addr, "hello")`
    /// 2. `("clipboard", addr, "world")`
    pub fn clipboard(&self, machine: &mut Machine, text: &str) {
        for line in text.lines() {
            machine.push_signal(
                Signal::new("clipboard")
                    .with_string(self.address.clone())
                    .with_string(line.to_owned())
            );
        }
    }
}