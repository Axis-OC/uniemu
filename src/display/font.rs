//! # Unifont `.hex` parser and glyph atlas builder
//!
//! This module parses the GNU Unifont HEX format (the font format used
//! by OpenComputers) and produces a 4096x4096 R8 texture atlas covering
//! the entire Basic Multilingual Plane (U+0000 through U+FFFF).
//!
//! ## Background
//!
//! GNU Unifont is a bitmap font that provides a glyph for every Unicode
//! code point in the BMP. OpenComputers bundles a copy of Unifont and
//! uses it for all text rendering. The `.hex` format is a simple
//! text-based encoding:
//!
//! ```text
//! 0041:00000000081422417F4141414141000000
//! ```
//!
//! Where `0041` is the code point (U+0041 = 'A') and the hex string
//! encodes the bitmap row by row, MSB first.
//!
//! ## Atlas layout
//!
//! The atlas is a 4096x4096 pixel image divided into a 256x256 grid
//! of 16x16 pixel cells:
//!
//! ```text
//! +---+---+---+---+- ... -+---+
//! |0,0|0,1|0,2|0,3|       |0,FF|  <- row 0 (U+00xx)
//! +---+---+---+---+- ... -+---+
//! |1,0|1,1|1,2|1,3|       |1,FF|  <- row 1 (U+01xx)
//! +---+---+---+---+- ... -+---+
//! |   :   :   :   :       :   |
//! +---+---+---+---+- ... -+---+
//! |FF,0| ...                     <- row 255 (U+FFxx)
//! +---+---+---+---+- ... -+---+
//! ```
//!
//! Code point `cp` maps to:
//! * Column: `cp & 0xFF`
//! * Row: `(cp >> 8) & 0xFF`
//! * Pixel origin: `(column * 16, row * 16)`
//!
//! ## Narrow vs. wide glyphs
//!
//! * Narrow glyphs (most Latin, Cyrillic, etc.): 8 pixels wide, 16 tall.
//!   Left-aligned in the 16x16 cell with 8 blank columns on the right.
//! * Wide glyphs (CJK ideographs, etc.): 16 pixels wide, 16 tall.
//!   Fill the entire cell.
//!
//! The distinction matters for text layout: a wide glyph occupies two
//! text columns on screen (like in a terminal).
//!
//! ## Memory usage
//!
//! The atlas is `4096 * 4096 = 16 MiB` of R8 data. This is uploaded
//! to the GPU as a `VK_FORMAT_R8_UNORM` texture (or used directly in
//! software rendering). Each byte is either `0x00` (background) or
//! `0xFF` (foreground); the fragment shader uses it as an alpha mask.

/// Atlas width and height in pixels.
///
/// 4096 = 256 cells * 16 pixels per cell.
pub const ATLAS_SIZE: u32 = 4096;

/// Number of glyph cells per atlas axis (256 columns, 256 rows).
pub const ATLAS_CELLS: u32 = 256;

/// Width of each glyph cell in pixels.
///
/// Even narrow (8px) glyphs are placed in 16px-wide cells, with
/// the right half empty.
pub const CELL_W: u32 = 16;

/// Height of each glyph cell in pixels.
pub const CELL_H: u32 = 16;

/// Parsed glyph atlas: a flat `ATLAS_SIZE * ATLAS_SIZE` array of bytes.
///
/// Each byte is either `0x00` (background pixel) or `0xFF` (foreground
/// pixel). This format is directly uploadable as an R8 texture.
///
/// Also stores per-codepoint width metadata (8 or 16 pixels).
pub struct GlyphAtlas {
    /// Row-major pixel data in R8 format.
    ///
    /// Index `[y * ATLAS_SIZE + x]` gives the pixel at `(x, y)`.
    /// Total size: 16,777,216 bytes (16 MiB).
    pub pixels: Vec<u8>,

    /// Per-codepoint width in pixels (8 for narrow, 16 for wide).
    ///
    /// Indexed by code point: `widths[cp]` gives the pixel width
    /// of the glyph for code point `cp`.
    ///
    /// Length: 65536 entries (one per BMP code point).
    /// Default: all 8 (narrow) until overridden by `.hex` data.
    pub widths: Vec<u8>,
}

impl GlyphAtlas {
    /// Create an empty atlas (all black pixels, all narrow widths).
    ///
    /// Call [`load_hex`](GlyphAtlas::load_hex) afterwards to populate
    /// it with actual glyph data.
    pub fn new() -> Self {
        Self {
            pixels: vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize],
            widths: vec![8u8; 65536],
        }
    }

    /// Parse a GNU Unifont `.hex` file and populate the atlas.
    ///
    /// # Format
    ///
    /// Each line has the format:
    ///
    /// ```text
    /// CODEPOINT:HEXBITMAP
    /// ```
    ///
    /// Where:
    /// * `CODEPOINT` is a 2-6 digit hex number (e.g. `0041` for 'A').
    /// * `HEXBITMAP` is a hex string encoding the bitmap:
    ///   * 32 hex chars (16 bytes) -> 8x16 narrow glyph
    ///   * 64 hex chars (32 bytes) -> 16x16 wide glyph
    ///
    /// Lines starting with `#` are comments. Empty lines are skipped.
    /// Code points >= U+10000 (outside BMP) are skipped.
    ///
    /// # Bitmap encoding
    ///
    /// Each byte encodes 8 horizontal pixels, MSB = leftmost.
    /// For a narrow glyph (8px wide), each row is 1 byte (8 bits).
    /// For a wide glyph (16px wide), each row is 2 bytes (16 bits).
    /// Rows are stored top-to-bottom.
    ///
    /// # Example
    ///
    /// ```text
    /// 0041:00000000081422417F4141414141000000
    /// ```
    ///
    /// This is U+0041 ('A'), 32 hex chars = 16 bytes = 8x16 narrow:
    ///
    /// ```text
    /// Row  0: 0x00 = ........
    /// Row  1: 0x00 = ........
    /// Row  2: 0x00 = ........
    /// Row  3: 0x00 = ........
    /// Row  4: 0x08 = ....X...
    /// Row  5: 0x14 = ...X.X..
    /// Row  6: 0x22 = ..X...X.
    /// Row  7: 0x41 = .X.....X
    /// Row  8: 0x7F = .XXXXXXX
    /// Row  9: 0x41 = .X.....X
    /// Row 10: 0x41 = .X.....X
    /// Row 11: 0x41 = .X.....X
    /// Row 12: 0x41 = .X.....X
    /// Row 13: 0x41 = .X.....X
    /// Row 14: 0x00 = ........
    /// Row 15: 0x00 = ........
    /// ```
    ///
    /// # Arguments
    ///
    /// * `hex_data` - The entire contents of a `.hex` file as a string.
    ///
    /// # Side effects
    ///
    /// Modifies `self.pixels` and `self.widths` in place. Can be called
    /// multiple times to overlay multiple font files.
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

    /// Get the display width of a code point in text columns.
    ///
    /// * Returns 1 for narrow (8px) glyphs (most characters).
    /// * Returns 2 for wide (16px) glyphs (CJK, some symbols).
    ///
    /// This is used by text layout code to determine how many cells
    /// a character occupies in the text grid.
    ///
    /// # Arguments
    ///
    /// * `cp` - Unicode code point.
    #[inline]
    pub fn char_width(&self, cp: u32) -> u32 {
        if (cp as usize) < self.widths.len() && self.widths[cp as usize] > 8 {
            2
        } else {
            1
        }
    }
}

/// Decode a hexadecimal string into a byte vector.
///
/// # Arguments
///
/// * `s` - A string of hex characters (0-9, a-f, A-F). Must have even
///   length. Leading/trailing whitespace is trimmed.
///
/// # Returns
///
/// * `Some(bytes)` - The decoded bytes.
/// * `None` - Invalid input (odd length, non-hex characters).
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

/// Convert a single ASCII hex character to its numeric value (0-15).
///
/// # Returns
///
/// * `Some(0..=15)` for valid hex chars.
/// * `None` for anything else.
#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}