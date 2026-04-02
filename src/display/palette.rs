//! # Color depth and palette management
//!
//! This module implements the three color depths from OpenComputers'
//! `api.internal.TextBuffer.ColorDepth` and the corresponding palette
//! system from `PackedColor.scala`.
//!
//! ## Color depths
//!
//! * **1-bit (OneBit)**: Monochrome. Only black (`0x000000`) and white
//!   (`0xFFFFFF`) are available. No palette.
//!
//! * **4-bit (FourBit)**: 16 colors from a user-configurable palette.
//!   Default palette matches Minecraft's dye colors (white, orange,
//!   magenta, light blue, yellow, lime, pink, gray, silver, cyan,
//!   purple, blue, brown, green, red, black).
//!
//! * **8-bit (EightBit)**: 240 fixed colors + 16 configurable palette
//!   entries. The fixed colors are derived from a 6x8x5 color cube
//!   (indices 16-255), with a grayscale ramp for the remaining slots.
//!   The 16 configurable entries default to a grayscale ramp.
//!
//! ## Color packing
//!
//! Colors in OpenComputers can be specified either as direct 24-bit
//! RGB values (`0x00RRGGBB`) or as palette indices (0-15). The
//! [`PackedColor`] type encodes both representations in a compact
//! struct, with a boolean flag distinguishing them.
//!
//! This is important because:
//! * Palette colors change when the palette is modified (the cell
//!   remembers the index, not the RGB value).
//! * Direct RGB colors are fixed regardless of palette changes.
//! * `gpu.get(x, y)` must be able to report whether a cell's color
//!   came from the palette or was set directly.
//!
//! ## Default 16-color palette
//!
//! The default palette for 4-bit depth matches Minecraft's dye colors
//! as used by OpenComputers:
//!
//! ```text
//! Index  Name         Hex
//!  0     White        F0F0F0
//!  1     Orange       F2B233
//!  2     Magenta      E57FD8
//!  3     Light Blue   99B2F2
//!  4     Yellow       DEDE6C
//!  5     Lime         7FCC19
//!  6     Pink         F2B2CC
//!  7     Gray         4C4C4C
//!  8     Silver       999999
//!  9     Cyan         4C99B2
//! 10     Purple       B266E5
//! 11     Blue         3366CC
//! 12     Brown        7F664C
//! 13     Green        57A64E
//! 14     Red          CC4C4C
//! 15     Black        1E1B1B
//! ```

/// Color depth enum matching `api.internal.TextBuffer.ColorDepth`.
///
/// The variants are ordered by bit count, and [`PartialOrd`] /
/// [`Ord`] are derived so that `OneBit < FourBit < EightBit`.
/// This allows comparisons like `if self.max_depth >= FourBit`.
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ColorDepth {
    /// Monochrome: 1-bit. Only black and white.
    ///
    /// No palette is available. All colors are snapped to either
    /// `0x000000` (black) or `0xFFFFFF` (white).
    OneBit = 0,

    /// 16-color palette: 4-bit.
    ///
    /// All 16 palette entries are user-configurable via
    /// `gpu.setPaletteColor()`. Defaults to Minecraft dye colors.
    FourBit = 1,

    /// 256-color: 8-bit.
    ///
    /// 16 configurable palette entries + 240 fixed colors from a
    /// 6x8x5 color cube and grayscale ramp. Defaults to a grayscale
    /// palette for the configurable slots.
    EightBit = 2,
}

impl ColorDepth {
    /// Number of color bits for display in Lua (1, 4, or 8).
    ///
    /// Used by `gpu.getDepth()` and `gpu.setDepth()`.
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::OneBit   => 1,
            Self::FourBit  => 4,
            Self::EightBit => 8,
        }
    }

    /// Number of user-configurable palette entries at this depth.
    ///
    /// * 1-bit: 0 (no palette)
    /// * 4-bit: 16 (all configurable)
    /// * 8-bit: 16 (configurable) + 240 fixed = 16 configurable
    #[inline]
    pub const fn palette_size(self) -> usize {
        match self {
            Self::OneBit   => 0,
            Self::FourBit  => 16,
            Self::EightBit => 16,
        }
    }
}

/// Packed color value.
///
/// Stores either a raw 24-bit RGB value or a palette index, with a
/// boolean flag distinguishing the two. This mirrors the encoding
/// used in `PackedColor.scala`.
///
/// # Why not just use `u32`?
///
/// Because `gpu.get(x, y)` must report whether a cell's color was set
/// via a palette index or directly. If we discarded that information,
/// programs that rely on palette semantics (e.g. changing a palette
/// entry to update all cells using that color) would break.
///
/// # Memory layout
///
/// `#[repr(C)]` for predictable layout in GPU uploads:
/// * `value: u32` at offset 0 (4 bytes)
/// * `is_palette: bool` at offset 4 (1 byte, may have padding)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PackedColor {
    /// The raw value.
    ///
    /// If `is_palette` is `false`: a 24-bit RGB color (`0x00RRGGBB`).
    /// If `is_palette` is `true`: a palette index (0-15).
    value: u32,

    /// Whether `value` is a palette index rather than an RGB triplet.
    is_palette: bool,
}

impl PackedColor {
    /// Create a direct RGB color.
    ///
    /// The upper 8 bits are masked off (`& 0x00FFFFFF`).
    ///
    /// # Examples (conceptual)
    ///
    /// ```text
    /// let white = PackedColor::rgb(0xFFFFFF);
    /// let red   = PackedColor::rgb(0xFF0000);
    /// ```
    #[inline]
    pub const fn rgb(rgb: u32) -> Self {
        Self { value: rgb & 0x00FF_FFFF, is_palette: false }
    }

    /// Create a palette-indexed color.
    ///
    /// # Arguments
    ///
    /// * `index` - Palette index (0-15). Values > 15 are technically
    ///   valid but will reference fixed colors at 8-bit depth or
    ///   wrap at 4-bit depth.
    #[inline]
    pub const fn palette(index: u8) -> Self {
        Self { value: index as u32, is_palette: true }
    }

    /// The raw stored value (RGB or palette index).
    #[inline]
    pub const fn value(self) -> u32 { self.value }

    /// Whether this color references a palette entry.
    #[inline]
    pub const fn is_from_palette(self) -> bool { self.is_palette }

    /// Resolve to a concrete 24-bit RGB value using the given palette.
    ///
    /// * If this is a direct RGB color, returns `self.value` unchanged.
    /// * If this is a palette index, looks up the palette and returns
    ///   the RGB value stored there.
    ///
    /// # Arguments
    ///
    /// * `palette` - The palette to look up indices in.
    ///
    /// # Returns
    ///
    /// A 24-bit RGB value (`0x00RRGGBB`).
    #[inline]
    pub fn resolve(self, palette: &Palette) -> u32 {
        if self.is_palette {
            palette.get(self.value as usize)
        } else {
            self.value
        }
    }
}

/// Default 16-color palette matching Minecraft dye colors.
///
/// Source: `PackedColor.scala` in the OpenComputers source code.
///
/// These are the colors used when the screen is at 4-bit depth and
/// the palette has not been modified by the user.
pub const DEFAULT_PALETTE_16: [u32; 16] = [
    0xF0F0F0, // 0  white
    0xF2B233, // 1  orange
    0xE57FD8, // 2  magenta
    0x99B2F2, // 3  light blue
    0xDEDE6C, // 4  yellow
    0x7FCC19, // 5  lime
    0xF2B2CC, // 6  pink
    0x4C4C4C, // 7  gray
    0x999999, // 8  silver
    0x4C99B2, // 9  cyan
    0xB266E5, // 10 purple
    0x3366CC, // 11 blue
    0x7F664C, // 12 brown
    0x57A64E, // 13 green
    0xCC4C4C, // 14 red
    0x1E1B1B, // 15 black
];

/// Configurable color palette with up to 16 entries.
///
/// At 4-bit depth, all 16 entries are user-configurable via
/// `gpu.setPaletteColor(index, color)`.
///
/// At 8-bit depth, the first 16 entries are configurable; indices
/// 16-255 reference fixed colors computed from a 6x8x5 color cube
/// and a grayscale ramp (see [`fixed_color_8bit`](Palette::fixed_color_8bit)).
///
/// At 1-bit depth, the palette is unused (all colors snap to black
/// or white).
#[derive(Debug, Clone)]
pub struct Palette {
    /// The 16 configurable palette entries, stored as 24-bit RGB.
    entries: [u32; 16],

    /// The current color depth, which determines how the palette
    /// is initialised and how indices > 15 are handled.
    depth: ColorDepth,
}

impl Palette {
    /// Create a new palette initialised to the defaults for the given depth.
    ///
    /// * 1-bit: all entries zeroed (palette not used).
    /// * 4-bit: entries set to [`DEFAULT_PALETTE_16`] (Minecraft dye colors).
    /// * 8-bit: entries set to a 16-step grayscale ramp (0x000000 to 0xFFFFFF).
    pub fn new(depth: ColorDepth) -> Self {
        let entries = match depth {
            ColorDepth::OneBit => [0; 16],
            ColorDepth::FourBit => DEFAULT_PALETTE_16,
            ColorDepth::EightBit => {
                let mut e = [0u32; 16];
                for i in 0..16 {
                    let v = (i * 255 / 15) as u32;
                    e[i] = (v << 16) | (v << 8) | v;
                }
                e
            }
        };
        Self { entries, depth }
    }

    /// Get the RGB value at the given palette index.
    ///
    /// * For indices 0-15: returns the configurable entry.
    /// * For indices 16-255 at 8-bit depth: computes and returns the
    ///   fixed color from the 6x8x5 color cube + grayscale ramp.
    /// * For indices 16+ at other depths: returns 0 (black).
    ///
    /// # Arguments
    ///
    /// * `index` - Palette index (0-255).
    #[inline]
    pub fn get(&self, index: usize) -> u32 {
        if index < 16 {
            self.entries[index]
        } else if self.depth == ColorDepth::EightBit {
            Self::fixed_color_8bit(index)
        } else {
            0
        }
    }

    /// Set a configurable palette entry.
    ///
    /// Only indices 0-15 are mutable. Indices 16+ are fixed and
    /// cannot be changed.
    ///
    /// # Arguments
    ///
    /// * `index` - Palette index (0-15).
    /// * `color` - 24-bit RGB value (`0x00RRGGBB`).
    ///
    /// # Returns
    ///
    /// `true` if the index was valid and the color was set.
    /// `false` if the index was out of range (>= 16).
    pub fn set(&mut self, index: usize, color: u32) -> bool {
        if index < 16 {
            self.entries[index] = color & 0x00FF_FFFF;
            true
        } else {
            false
        }
    }

    /// Current color depth.
    #[inline]
    pub fn depth(&self) -> ColorDepth { self.depth }

    /// Change the color depth, reinitialising the palette to defaults.
    ///
    /// This is called when `gpu.setDepth()` changes the depth. All
    /// palette entries are reset to their default values for the new
    /// depth.
    pub fn set_depth(&mut self, depth: ColorDepth) {
        *self = Self::new(depth);
    }

    /// Compute a fixed 8-bit color for indices 16-255.
    ///
    /// OpenComputers uses a 6x8x5 color cube for indices 16..255:
    ///
    /// ```text
    /// index_in_cube = index - 16
    /// blue  = index_in_cube % 5
    /// green = (index_in_cube / 5) % 8
    /// red   = index_in_cube / 40
    ///
    /// R = red   * 255 / 5
    /// G = green * 255 / 7
    /// B = blue  * 255 / 4
    /// ```
    ///
    /// Indices beyond the 6x8x5 cube (240 entries) are mapped to a
    /// grayscale ramp.
    ///
    /// # Arguments
    ///
    /// * `index` - Palette index (16-255).
    ///
    /// # Returns
    ///
    /// A 24-bit RGB value.
    fn fixed_color_8bit(index: usize) -> u32 {
        let idx = index - 16;
        if idx < 6 * 8 * 5 {
            let b = idx % 5;
            let g = (idx / 5) % 8;
            let r = idx / 40;
            let rb = (r * 255 / 5) as u32;
            let gb = (g * 255 / 7) as u32;
            let bb = (b * 255 / 4) as u32;
            (rb << 16) | (gb << 8) | bb
        } else {
            let gray = ((idx - 240) * 255 / 15).min(255) as u32;
            (gray << 16) | (gray << 8) | gray
        }
    }
}