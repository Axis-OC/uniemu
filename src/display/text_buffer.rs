//! # Text buffer
//!
//! A CPU-side character grid with per-cell foreground/background colors.
//! This is the canonical data structure that all rendering backends
//! (software, Vulkan INDIRECT, Vulkan DIRECT) read from.
//!
//! ## Relationship to OpenComputers
//!
//! Mirrors `li.cil.oc.common.component.TextBuffer` and the
//! `li.cil.oc.api.internal.TextBuffer` interface from the Scala source.
//!
//! In OC, the text buffer is the authoritative copy of screen contents.
//! The GPU writes to it; the renderer reads from it. This emulator
//! follows the same architecture.
//!
//! ## Dirty tracking
//!
//! The buffer has a simple boolean `dirty` flag that is set to `true`
//! whenever any cell is modified. Renderers check this flag and skip
//! uploading/rendering when nothing has changed. After uploading, the
//! renderer calls [`clear_dirty()`](TextBuffer::clear_dirty).
//!
//! The DIRECT renderer also maintains its own per-row shadow buffer
//! for finer-grained dirty tracking, but that logic lives in
//! `render/direct.rs`, not here.
//!
//! ## Coordinate system
//!
//! Internally, all coordinates are 0-based:
//! * Column 0 is the leftmost column.
//! * Row 0 is the topmost row.
//!
//! The GPU dispatch layer in `host.rs` translates from Lua's 1-based
//! coordinates to 0-based before calling these methods.
//!
//! ## Viewport
//!
//! The viewport is a sub-region of the buffer that is actually displayed.
//! It can be smaller than the buffer resolution, allowing scrolling or
//! double-buffering effects. In practice, the viewport usually equals
//! the resolution.

use super::{Cell, PackedColor, Palette, ColorDepth};

/// A two-dimensional text buffer with color information.
///
/// The buffer owns its cells, palette, current foreground/background
/// colors, viewport dimensions, and a dirty flag.
///
/// # Ownership
///
/// This struct is owned by [`EmulatorState`](crate::lua::host::EmulatorState).
/// The GPU component writes to it; renderers read from it.
///
/// # Resizing
///
/// Calling [`set_resolution`](TextBuffer::set_resolution) clears the
/// entire buffer and resets all cells to default. The viewport is also
/// reset to match the new resolution.
#[derive(Debug, Clone)]
pub struct TextBuffer {
    /// Width of the buffer in columns (characters).
    width: u32,

    /// Height of the buffer in rows (characters).
    height: u32,

    /// Flat array of cells, row-major order.
    ///
    /// Index `[row * width + col]` gives the cell at `(col, row)`.
    /// Length: `width * height`.
    cells: Vec<Cell>,

    /// The color palette for this buffer.
    ///
    /// Determines how palette-indexed colors are resolved to RGB.
    /// Changed when `gpu.setDepth()` or `gpu.setPaletteColor()` is called.
    palette: Palette,

    /// Current foreground color for new writes.
    ///
    /// Set by `gpu.setForeground()`. Applied to all cells written by
    /// subsequent `set()` and `fill()` calls.
    fg: PackedColor,

    /// Current background color for new writes.
    ///
    /// Set by `gpu.setBackground()`.
    bg: PackedColor,

    /// Viewport dimensions (may be <= buffer dimensions).
    ///
    /// Only the viewport area is displayed on screen. Cells outside
    /// the viewport exist in memory but are not rendered.
    viewport: (u32, u32),

    /// Set to `true` whenever any cell is modified.
    ///
    /// Cleared by the renderer after uploading. Used to avoid redundant
    /// GPU uploads when nothing has changed.
    dirty: bool,
}

impl TextBuffer {
    /// Create a new buffer filled with default cells (spaces, white on black).
    ///
    /// # Arguments
    ///
    /// * `width` - Number of columns.
    /// * `height` - Number of rows.
    /// * `depth` - Initial color depth (determines palette defaults).
    ///
    /// # Returns
    ///
    /// A buffer with all cells set to default, viewport equal to
    /// resolution, and the dirty flag set to `true` (so the renderer
    /// picks it up on the first frame).
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

    // -------------------------------------------------------------------
    // Dimensions
    // -------------------------------------------------------------------

    /// Width of the buffer in columns.
    #[inline] pub fn width(&self)  -> u32 { self.width }

    /// Height of the buffer in rows.
    #[inline] pub fn height(&self) -> u32 { self.height }

    /// Current viewport dimensions `(width, height)`.
    #[inline] pub fn viewport(&self) -> (u32, u32) { self.viewport }

    /// Resize the buffer, clearing all content.
    ///
    /// All cells are reset to default (space, white on black).
    /// The viewport is reset to match the new resolution.
    ///
    /// # Arguments
    ///
    /// * `w` - New width in columns.
    /// * `h` - New height in rows.
    ///
    /// # Returns
    ///
    /// `true` if the resolution actually changed (and thus content
    /// was cleared). `false` if the resolution was already `(w, h)`.
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

    /// Set the viewport size (clamped to current resolution).
    ///
    /// # Returns
    ///
    /// `true` if the viewport changed, `false` if it was already the
    /// requested size.
    pub fn set_viewport(&mut self, w: u32, h: u32) -> bool {
        let w = w.min(self.width);
        let h = h.min(self.height);
        if w == self.viewport.0 && h == self.viewport.1 { return false; }
        self.viewport = (w, h);
        true
    }

    // -------------------------------------------------------------------
    // Color state
    // -------------------------------------------------------------------

    /// Current foreground color (for new writes).
    #[inline] pub fn foreground(&self) -> PackedColor { self.fg }

    /// Current background color (for new writes).
    #[inline] pub fn background(&self) -> PackedColor { self.bg }

    /// Set the foreground color for subsequent writes.
    pub fn set_foreground(&mut self, color: PackedColor) { self.fg = color; }

    /// Set the background color for subsequent writes.
    pub fn set_background(&mut self, color: PackedColor) { self.bg = color; }

    /// Immutable reference to the palette.
    #[inline] pub fn palette(&self) -> &Palette { &self.palette }

    /// Mutable reference to the palette (for `gpu.setPaletteColor()`).
    #[inline] pub fn palette_mut(&mut self) -> &mut Palette { &mut self.palette }

    /// Change the color depth, reinitialising the palette.
    ///
    /// # Returns
    ///
    /// `true` if the depth changed, `false` if it was already set.
    pub fn set_color_depth(&mut self, depth: ColorDepth) -> bool {
        if self.palette.depth() == depth { return false; }
        self.palette.set_depth(depth);
        self.dirty = true;
        true
    }

    // -------------------------------------------------------------------
    // Cell access
    // -------------------------------------------------------------------

    /// Compute the flat array index for a cell at `(col, row)`.
    ///
    /// Returns `None` if out of bounds.
    #[inline]
    fn index(&self, col: u32, row: u32) -> Option<usize> {
        if col < self.width && row < self.height {
            Some((row * self.width + col) as usize)
        } else {
            None
        }
    }

    /// Read a single cell at `(col, row)`.
    ///
    /// Returns `None` if out of bounds.
    ///
    /// # Note
    ///
    /// Returns a reference to the cell, not a copy. The cell's colors
    /// may be palette references; resolve them with `cell.foreground.resolve(palette)`.
    #[inline]
    pub fn get(&self, col: u32, row: u32) -> Option<&Cell> {
        self.index(col, row).map(|i| &self.cells[i])
    }

    /// Write a string starting at `(col, row)`.
    ///
    /// Each character of `text` is written to successive cells,
    /// advancing rightward (or downward if `vertical` is true).
    /// The current foreground and background colors are applied.
    ///
    /// Characters that fall outside the buffer bounds are silently
    /// discarded (no wrapping to the next line).
    ///
    /// # Arguments
    ///
    /// * `col` - Starting column (0-based).
    /// * `row` - Starting row (0-based).
    /// * `text` - The string to write.
    /// * `vertical` - If `true`, advance downward instead of rightward.
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
    ///
    /// All cells in the rectangle `[col..col+w) x [row..row+h)` are
    /// set to the specified code point with the current foreground and
    /// background colors.
    ///
    /// Out-of-bounds regions are clipped silently.
    ///
    /// # Arguments
    ///
    /// * `col` - Left column (0-based).
    /// * `row` - Top row (0-based).
    /// * `w` - Width of the rectangle in cells.
    /// * `h` - Height of the rectangle in cells.
    /// * `codepoint` - Unicode code point to fill with (e.g. `' ' as u32`).
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

    /// Copy a rectangular region by the given translation offset.
    ///
    /// Cells in the source rectangle are copied to
    /// `(col + tx, row + ty)`. Overlapping source and destination
    /// regions are handled correctly (like `memmove` vs `memcpy`):
    /// the iteration order is chosen to prevent overwriting source
    /// data before it is read.
    ///
    /// # Arguments
    ///
    /// * `col` - Left column of source rectangle (0-based).
    /// * `row` - Top row of source rectangle (0-based).
    /// * `w` - Width of the rectangle.
    /// * `h` - Height of the rectangle.
    /// * `tx` - Horizontal translation (positive = right).
    /// * `ty` - Vertical translation (positive = down).
    ///
    /// # Clipping
    ///
    /// Source cells that map to out-of-bounds destinations are silently
    /// skipped.
    pub fn copy(&mut self, col: u32, row: u32, w: u32, h: u32, tx: i32, ty: i32) {
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

    // -------------------------------------------------------------------
    // Dirty tracking
    // -------------------------------------------------------------------

    /// Whether any cell has been modified since the last `clear_dirty()` call.
    #[inline] pub fn is_dirty(&self) -> bool { self.dirty }

    /// Clear the dirty flag.
    ///
    /// Called by the renderer after uploading the buffer contents to
    /// the GPU (or after rasterising in software). Until the next
    /// modification, `is_dirty()` will return `false`.
    #[inline] pub fn clear_dirty(&mut self) { self.dirty = false; }

    /// Raw cell slice for bulk upload to GPU.
    ///
    /// Returns the flat array of cells in row-major order. Each cell
    /// is 12+ bytes (codepoint + fg + bg). The SSBO upload code reads
    /// this slice and extracts `(codepoint, resolved_fg, resolved_bg)`
    /// triples.
    #[inline] pub fn cells(&self) -> &[Cell] { &self.cells }
}