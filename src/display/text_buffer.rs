//! # Text buffer
//!
//! A CPU-side character grid with per-cell foreground/background colors.
//! This is the canonical data structure that both INDIRECT and DIRECT
//! rendering backends read from.
//!
//! Mirrors the behaviour of `common.component.TextBuffer` and the
//! `api.internal.TextBuffer` interface.

use super::{Cell, PackedColor, Palette, ColorDepth};

/// A two-dimensional text buffer with color information.
///
/// The buffer owns its palette and tracks a "dirty" flag for efficient
/// synchronisation with the GPU in DIRECT mode.
#[derive(Debug, Clone)]
pub struct TextBuffer {
    width: u32,
    height: u32,
    cells: Vec<Cell>,
    palette: Palette,
    /// Current foreground for new writes.
    fg: PackedColor,
    /// Current background for new writes.
    bg: PackedColor,
    /// Viewport dimensions (may be ≤ buffer dimensions).
    viewport: (u32, u32),
    /// Set to `true` whenever any cell is modified.
    dirty: bool,
}

impl TextBuffer {
    /// Create a new buffer filled with spaces on a black background.
    pub fn new(width: u32, height: u32, depth: ColorDepth) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            cells: vec![Cell::default(); size],
            palette: Palette::new(depth),
            fg: PackedColor::rgb(0xFF_FF_FF),
            bg: PackedColor::rgb(0x00_00_00),
            viewport: (width, height),
            dirty: true,
        }
    }

    // ── Dimensions ──────────────────────────────────────────────────────

    #[inline] pub fn width(&self)  -> u32 { self.width }
    #[inline] pub fn height(&self) -> u32 { self.height }
    #[inline] pub fn viewport(&self) -> (u32, u32) { self.viewport }

    /// Resize the buffer, clearing all content.  Returns `true` if changed.
    pub fn set_resolution(&mut self, w: u32, h: u32) -> bool {
        if w == self.width && h == self.height { return false; }
        self.width = w;
        self.height = h;
        self.cells.resize((w * h) as usize, Cell::default());
        self.cells.fill(Cell::default());
        self.viewport = (w, h);
        self.dirty = true;
        true
    }

    /// Set viewport (clamped to current resolution).  Returns `true` if changed.
    pub fn set_viewport(&mut self, w: u32, h: u32) -> bool {
        let w = w.min(self.width);
        let h = h.min(self.height);
        if w == self.viewport.0 && h == self.viewport.1 { return false; }
        self.viewport = (w, h);
        true
    }

    // ── Color state ─────────────────────────────────────────────────────

    #[inline] pub fn foreground(&self) -> PackedColor { self.fg }
    #[inline] pub fn background(&self) -> PackedColor { self.bg }

    pub fn set_foreground(&mut self, color: PackedColor) { self.fg = color; }
    pub fn set_background(&mut self, color: PackedColor) { self.bg = color; }

    #[inline] pub fn palette(&self) -> &Palette { &self.palette }
    #[inline] pub fn palette_mut(&mut self) -> &mut Palette { &mut self.palette }

    pub fn set_color_depth(&mut self, depth: ColorDepth) -> bool {
        if self.palette.depth() == depth { return false; }
        self.palette.set_depth(depth);
        self.dirty = true;
        true
    }

    // ── Cell access ─────────────────────────────────────────────────────

    #[inline]
    fn index(&self, col: u32, row: u32) -> Option<usize> {
        if col < self.width && row < self.height {
            Some((row * self.width + col) as usize)
        } else {
            None
        }
    }

    /// Read a single cell.
    #[inline]
    pub fn get(&self, col: u32, row: u32) -> Option<&Cell> {
        self.index(col, row).map(|i| &self.cells[i])
    }

    /// Write a string starting at `(col, row)`.
    ///
    /// Applies the current foreground and background colors.
    /// If `vertical` is true, characters advance downward instead of rightward.
    pub fn set(&mut self, col: u32, row: u32, text: &str, vertical: bool) {
        let (mut c, mut r) = (col, row);
        for ch in text.chars() {
            if let Some(idx) = self.index(c, r) {
                self.cells[idx] = Cell {
                    codepoint: ch as u32,
                    foreground: self.fg,
                    background: self.bg,
                };
            }
            if vertical { r += 1; } else { c += 1; }
        }
        self.dirty = true;
    }

    /// Fill a rectangle with the given code point and current colors.
    pub fn fill(&mut self, col: u32, row: u32, w: u32, h: u32, codepoint: u32) {
        let cell = Cell {
            codepoint,
            foreground: self.fg,
            background: self.bg,
        };
        for r in row..row.saturating_add(h).min(self.height) {
            for c in col..col.saturating_add(w).min(self.width) {
                self.cells[(r * self.width + c) as usize] = cell;
            }
        }
        self.dirty = true;
    }

    /// Copy a rectangular region by the given translation.
    ///
    /// Overlapping regions are handled correctly (like `memmove`).
    pub fn copy(&mut self, col: u32, row: u32, w: u32, h: u32, tx: i32, ty: i32) {
        // Determine iteration order to handle overlap.
        let (row_range, col_range): (Box<dyn Iterator<Item=u32>>, Box<dyn Iterator<Item=u32>>) = (
            if ty > 0 { Box::new((row..row+h).rev()) } else { Box::new(row..row+h) },
            if tx > 0 { Box::new((col..col+w).rev()) } else { Box::new(col..col+w) },
        );

        for r in row_range {
            let col_iter: Box<dyn Iterator<Item=u32>> =
                if tx > 0 { Box::new((col..col+w).rev()) } else { Box::new(col..col+w) };
            for c in col_iter {
                let dr = r as i32 + ty;
                let dc = c as i32 + tx;
                if dr >= 0 && dr < self.height as i32
                && dc >= 0 && dc < self.width as i32
                {
                    let src = (r * self.width + c) as usize;
                    let dst = (dr as u32 * self.width + dc as u32) as usize;
                    self.cells[dst] = self.cells[src];
                }
            }
        }
        self.dirty = true;
    }

    // ── Dirty tracking ──────────────────────────────────────────────────

    /// Whether any cell has been modified since the last [`clear_dirty`] call.
    #[inline] pub fn is_dirty(&self) -> bool { self.dirty }

    /// Clear the dirty flag (called by the renderer after uploading).
    #[inline] pub fn clear_dirty(&mut self) { self.dirty = false; }

    /// Raw cell slice for bulk upload to GPU.
    #[inline] pub fn cells(&self) -> &[Cell] { &self.cells }
}