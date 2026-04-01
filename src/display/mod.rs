//! # Display subsystem
//!
//! Text buffer, color encoding, palette management, and font atlas.

mod text_buffer;
mod palette;
pub mod font;

pub use text_buffer::TextBuffer;
pub use palette::{ColorDepth, PackedColor, Palette, DEFAULT_PALETTE_16};

/// A single cell in the text buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Cell {
    pub codepoint: u32,
    pub foreground: PackedColor,
    pub background: PackedColor,
}

impl Default for Cell {
    #[inline]
    fn default() -> Self {
        Self {
            codepoint: b' ' as u32,
            foreground: PackedColor::rgb(0xFF_FF_FF),
            background: PackedColor::rgb(0x00_00_00),
        }
    }
}