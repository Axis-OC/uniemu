//! # Display subsystem
//!
//! This module provides the core data structures for the emulated
//! text display:
//!
//! * [`TextBuffer`] - A 2D grid of character cells with per-cell colors.
//! * [`Cell`] - A single cell: code point + foreground + background.
//! * [`Palette`] - Configurable color palette (up to 16 entries).
//! * [`PackedColor`] - A color that is either direct RGB or a palette index.
//! * [`ColorDepth`] - The current depth (1-bit, 4-bit, or 8-bit).
//! * [`font`] - Glyph atlas parser and builder.
//!
//! ## Architecture
//!
//! The display subsystem is intentionally decoupled from rendering.
//! The `TextBuffer` is the canonical data source; renderers (software,
//! Vulkan INDIRECT, Vulkan DIRECT) read from it and produce pixels.
//!
//! ```text
//! GPU component
//!   |
//!   | writes cells, colors, resolution
//!   v
//! TextBuffer (CPU-side, this module)
//!   |
//!   | read by renderer
//!   v
//! Software rasteriser / Vulkan SSBO upload / DIRECT persistent map
//! ```
//!
//! ## Cell format
//!
//! Each cell is 12 bytes:
//! * `codepoint: u32` - Unicode code point (0-65535 for BMP, but
//!   stored as u32 for `char::from_u32` compatibility).
//! * `foreground: PackedColor` - 5 bytes (u32 value + bool flag),
//!   but packed into a `#[repr(C)]` struct.
//! * `background: PackedColor` - Same.
//!
//! The `#[repr(C)]` layout ensures the struct can be safely
//! reinterpreted as raw bytes for GPU upload.

mod text_buffer;
mod palette;
pub mod font;

pub use text_buffer::TextBuffer;
pub use palette::{ColorDepth, PackedColor, Palette, DEFAULT_PALETTE_16};

/// A single cell in the text buffer.
///
/// Represents one character position on the screen. Contains the
/// Unicode code point to display and the foreground/background colors
/// for that specific cell.
///
/// # Memory layout
///
/// `#[repr(C)]` ensures a predictable layout for GPU upload:
///
/// ```text
/// Offset  Size  Field
/// 0       4     codepoint (u32)
/// 4       5     foreground (PackedColor: u32 + bool)
/// 9       5     background (PackedColor: u32 + bool)
/// ```
///
/// (Actual size may include padding; use `std::mem::size_of::<Cell>()`
/// for the true size.)
///
/// # Default
///
/// A default cell is a space character (`' '`, U+0020) with white
/// foreground (`0xFFFFFF`) and black background (`0x000000`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Cell {
    /// Unicode code point of the character displayed in this cell.
    ///
    /// For ASCII characters, this is the same as the byte value
    /// (e.g. `'A'` = 65). For non-BMP characters (> U+FFFF), the
    /// glyph atlas only covers the BMP, so they render as '?'.
    pub codepoint: u32,

    /// Foreground (text) color of this cell.
    ///
    /// Can be either a direct 24-bit RGB value or a palette index.
    /// Resolved to RGB via `PackedColor::resolve(palette)` at render time.
    pub foreground: PackedColor,

    /// Background color of this cell.
    ///
    /// Same encoding as `foreground`.
    pub background: PackedColor,
}

impl Default for Cell {
    /// Default cell: space character, white on black.
    ///
    /// This matches the initial state of an OpenComputers screen
    /// after `gpu.fill(1, 1, w, h, " ")` with default colors.
    #[inline]
    fn default() -> Self {
        Self {
            codepoint: b' ' as u32,
            foreground: PackedColor::rgb(0xFF_FF_FF),
            background: PackedColor::rgb(0x00_00_00),
        }
    }
}