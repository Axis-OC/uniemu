//! # Color depth and palette management
//!
//! Implements the three color depths from `TextBuffer.ColorDepth`:
//! - [`OneBit`]:  monochrome / only black (`0x000000`) and white (`0xFFFFFF`)
//! - [`FourBit`]: 16 colors from a configurable palette (defaults to MC dye colors)
//! - [`EightBit`]: 240 fixed colors + 16 palette entries (defaults to grayscale)
//!
//! Color packing follows `PackedColor.scala`.

/// Color depth enum matching `api.internal.TextBuffer.ColorDepth`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ColorDepth {
    /// Monochrome: 1-bit, black and white only.
    OneBit = 0,
    /// 16-color palette: 4-bit, defaults to Minecraft dye colors.
    FourBit = 1,
    /// 240 + 16 palette: 8-bit, defaults to a grayscale palette.
    EightBit = 2,
}

impl ColorDepth {
    /// Number of distinct displayable bits (for display in Lua: 1, 4, 8).
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::OneBit   => 1,
            Self::FourBit  => 4,
            Self::EightBit => 8,
        }
    }

    /// Number of palette entries available at this depth.
    #[inline]
    pub const fn palette_size(self) -> usize {
        match self {
            Self::OneBit   => 0,  // no palette
            Self::FourBit  => 16,
            Self::EightBit => 16, // 16 configurable + 240 fixed
        }
    }
}

/// Packed color value.
///
/// Stores either a raw 24-bit RGB value or a palette index, with a flag
/// distinguishing the two.  This mirrors `PackedColor.scala`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PackedColor {
    /// The raw value / either `0x00RRGGBB` (RGB) or a palette index (0–15).
    value: u32,
    /// Whether [`value`] is a palette index rather than an RGB triplet.
    is_palette: bool,
}

impl PackedColor {
    /// Create a direct RGB color.
    #[inline]
    pub const fn rgb(rgb: u32) -> Self {
        Self { value: rgb & 0x00FF_FFFF, is_palette: false }
    }

    /// Create a palette-indexed color.
    #[inline]
    pub const fn palette(index: u8) -> Self {
        Self { value: index as u32, is_palette: true }
    }

    /// The raw stored value.
    #[inline]
    pub const fn value(self) -> u32 { self.value }

    /// Whether this color references a palette entry.
    #[inline]
    pub const fn is_from_palette(self) -> bool { self.is_palette }

    /// Resolve to a concrete 24-bit RGB value using the given palette.
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
/// Source: `PackedColor.scala` / vanilla OC defaults.
/// Index 0 = white (0xF0F0F0), index 15 = black (0x1E1B1B).
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

/// Configurable palette with up to 16 entries.
///
/// At 4-bit depth all 16 entries are user-configurable.
/// At 8-bit depth the first 16 entries are configurable; the remaining
/// 240 colors are fixed (derived from a cube + grayscale ramp).
#[derive(Debug, Clone)]
pub struct Palette {
    entries: [u32; 16],
    depth: ColorDepth,
}

impl Palette {
    /// Create a new palette initialised to the defaults for the given depth.
    pub fn new(depth: ColorDepth) -> Self {
        let entries = match depth {
            ColorDepth::OneBit => [0; 16],
            ColorDepth::FourBit => DEFAULT_PALETTE_16,
            ColorDepth::EightBit => {
                // 8-bit default palette is grayscale for the configurable slots.
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
    /// For 8-bit depth and indices ≥ 16, computes the fixed color from
    /// the 6×8×5 color cube + grayscale ramp used by OC.
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

    /// Set a palette entry.  Only indices 0–15 are mutable.
    ///
    /// Returns `true` if the index was valid and the color was set.
    pub fn set(&mut self, index: usize, color: u32) -> bool {
        if index < 16 {
            self.entries[index] = color & 0x00FF_FFFF;
            true
        } else {
            false
        }
    }

    /// Current depth.
    #[inline]
    pub fn depth(&self) -> ColorDepth { self.depth }

    /// Change depth, reinitializing palette to defaults.
    pub fn set_depth(&mut self, depth: ColorDepth) {
        *self = Self::new(depth);
    }

    /// Compute a fixed 8-bit color for indices 16–255.
    ///
    /// OC uses a 6×8×5 color cube for indices 16..255.
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
            // Grayscale ramp for the remaining indices.
            let gray = ((idx - 240) * 255 / 15).min(255) as u32;
            (gray << 16) | (gray << 8) | gray
        }
    }
}