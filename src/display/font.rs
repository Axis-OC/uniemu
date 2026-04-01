//! # Unifont `.hex` parser and glyph atlas builder
//!
//! Parses the GNU Unifont HEX format used by OpenComputers and produces
//! a 4096×4096 R8 texture atlas covering the Basic Multilingual Plane
//! (U+0000–U+FFFF).
//!
//! ## Atlas layout
//!
//! - 256 columns × 256 rows of 16×16 pixel cells.
//! - Code point `cp` maps to column `cp & 0xFF`, row `(cp >> 8) & 0xFF`.
//! - Narrow (8-wide) glyphs are left-aligned with 8 blank columns on the right.

/// Atlas width and height in pixels.
pub const ATLAS_SIZE: u32 = 4096;

/// Number of glyph cells per atlas axis.
pub const ATLAS_CELLS: u32 = 256;

/// Pixel dimensions of each glyph cell.
pub const CELL_W: u32 = 16;
pub const CELL_H: u32 = 16;

/// Parsed glyph atlas / a flat `ATLAS_SIZE × ATLAS_SIZE` array of `u8`.
///
/// Each byte is 0x00 (background) or 0xFF (foreground).
pub struct GlyphAtlas {
    /// Row-major pixel data, R8 format.
    pub pixels: Vec<u8>,
    /// Width and height metrics per code point (8 or 16).
    pub widths: Vec<u8>,
}

impl GlyphAtlas {
    /// Create an empty atlas (all black).
    pub fn new() -> Self {
        Self {
            pixels: vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize],
            widths: vec![8u8; 65536], // default: all narrow
        }
    }

    /// Parse a `.hex` file and populate the atlas.
    ///
    /// Each line has the format: `CODEPOINT:HEXBITMAP`
    /// - 32 hex chars → 8×16 narrow glyph (16 bytes)
    /// - 64 hex chars → 16×16 wide glyph (32 bytes)
    pub fn load_hex(&mut self, hex_data: &str) {
        for line in hex_data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((cp_str, bitmap_str)) = line.split_once(':') {
                let cp = match u32::from_str_radix(cp_str, 16) {
                    Ok(v) if v < 0x10000 => v,
                    _ => continue,
                };

                let bytes = match hex_decode(bitmap_str) {
                    Some(b) => b,
                    None => continue,
                };

                let glyph_w: u32 = if bytes.len() <= 16 { 8 } else { 16 };
                let glyph_h: u32 = 16;
                let bytes_per_row = (glyph_w / 8) as usize;

                if cp < 65536 {
                    self.widths[cp as usize] = glyph_w as u8;
                }

                // Compute atlas cell origin.
                let cell_col = cp & 0xFF;
                let cell_row = (cp >> 8) & 0xFF;
                let origin_x = cell_col * CELL_W;
                let origin_y = cell_row * CELL_H;

                for row in 0..glyph_h.min(bytes.len() as u32 / bytes_per_row as u32) {
                    for col in 0..glyph_w {
                        let byte_idx = (row as usize) * bytes_per_row + (col as usize / 8);
                        let bit = 7 - (col % 8);
                        let set = if byte_idx < bytes.len() {
                            (bytes[byte_idx] >> bit) & 1 != 0
                        } else {
                            false
                        };

                        let px = origin_x + col;
                        let py = origin_y + row;
                        if px < ATLAS_SIZE && py < ATLAS_SIZE {
                            self.pixels[(py * ATLAS_SIZE + px) as usize] =
                                if set { 0xFF } else { 0x00 };
                        }
                    }
                }
            }
        }
    }

    /// Get the display width (in cells) of a code point.
    ///
    /// Returns 1 for narrow (8px) glyphs, 2 for wide (16px) glyphs.
    #[inline]
    pub fn char_width(&self, cp: u32) -> u32 {
        if (cp as usize) < self.widths.len() && self.widths[cp as usize] > 8 {
            2
        } else {
            1
        }
    }
}

/// Decode a hex string into bytes. Returns `None` on invalid input.
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 { return None; }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}