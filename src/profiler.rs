//! # Built-in profiler with timeline visualisation
//!
//! Provides a zero-overhead (when disabled) instrumentation system that
//! records timestamped events into a fixed-size ring buffer and renders
//! a multi-lane timeline overlay directly onto the software framebuffer.
//!
//! ## Design goals
//!
//! 1. **Zero cost when off**: A single `atomic::load(Relaxed)` guards
//!    every instrumentation point. When the profiler is not visible,
//!    no allocations, no syscalls, no timestamp reads occur.
//!
//! 2. **Lock-free recording**: The profiler is stored in a `thread_local!`
//!    `RefCell`, so recording never contends with the audio thread or
//!    any other thread. The main thread is the only writer.
//!
//! 3. **Fixed memory**: The ring buffer holds 65536 events (~2.5 MiB).
//!    Old events are silently overwritten. No dynamic allocation occurs
//!    after initialisation.
//!
//! 4. **Game-engine style timeline**: Inspired by Chrome's `chrome://tracing`,
//!    Unreal's Insights, and RenderDoc's CPU timeline. Each "lane"
//!    corresponds to a subsystem (Lua, GPU, filesystem, Vulkan, etc.)
//!    and shows coloured bars proportional to wall-clock duration.
//!
//! ## Visual layout (rendered via software rasteriser)
//!
//! ```text
//! ╔═══════════════════════════════════════════════════════════════════╗
//! ║ PROFILER │ Space:pause  <->:scroll  +/-:zoom  R:reset │ 100ms   ║
//! ║──────────┼── 10ms ──── 20ms ──── 30ms ──── 40ms ─────── 50ms ──║
//! ║ Tick     │████              ████                        20/s    ║
//! ║ Lua Step │██ ██ █           ██ ██ █                     156     ║
//! ║ GPU Calls│ █ █ ██ █████      █ █ ██                     1240    ║
//! ║ Budget   │▓▓▓▓▓▓▓░░░░      ▓▓▓▓▓▓▓░░                  67%     ║
//! ║ FS I/O   │      █    █           █                      45      ║
//! ║ Drive    │          █                                   12      ║
//! ║ Real Disk│                  ██                           3      ║
//! ║ Signals  │↑ ↑↑  ↑   ↑     ↑  ↑↑  ↑                    89      ║
//! ║ VK Fence │██                ██                          0.2ms   ║
//! ║ VK Acq   │ █                 █                          0.0ms   ║
//! ║ VK Submit│  █                 █                         0.1ms   ║
//! ║ VK Pres  │   █                 █                        0.3ms   ║
//! ║ Render   │████████          ████████                    2.1ms   ║
//! ╠═══════════════════════════════════════════════════════════════════╣
//! ║ Events: 1580 | Lua: 156 | GPU: 1240 | VK: 0.6ms | Window: 100ms║
//! ║ F7: close | Space: pause | +/-: zoom | <->: scroll | R: reset  ║
//! ╚═══════════════════════════════════════════════════════════════════╝
//! ```
//!
//! ## Activation
//!
//! Press **F7** to toggle the profiler overlay. While visible, the
//! profiler captures events and renders the timeline. When hidden,
//! recording stops and the hot-path check (`is_enabled()`) is a single
//! relaxed atomic load (~0.5 ns on x86-64).
//!
//! ## Controls
//!
//! | Key           | Action                                    |
//! |---------------|-------------------------------------------|
//! | F7            | Toggle profiler visibility                |
//! | Space         | Pause / resume recording                  |
//! | Left / Right  | Scroll timeline backward / forward        |
//! | Plus / Minus  | Zoom in / out (5 ms to 10 s range)        |
//! | R             | Clear all recorded events and reset view  |
//!
//! ## Event categories (lanes)
//!
//! Each lane represents a subsystem. Events within a lane are drawn
//! as coloured bars whose width is proportional to wall-clock duration.
//!
//! | Lane       | What it measures                              | Colour    |
//! |------------|-----------------------------------------------|-----------|
//! | Tick       | OC server tick (50 ms nominal)                | Mauve     |
//! | Lua Step   | `step_kernel` resume/yield cycles             | Green     |
//! | GPU Calls  | `component.invoke` on `"gpu"` components      | Blue      |
//! | Budget     | GPU call budget fill level (0–100%)           | Gradient  |
//! | FS I/O     | `component.invoke` on `"filesystem"` (VFS)    | Yellow    |
//! | Drive      | `component.invoke` on `"drive"` (raw sectors) | Peach     |
//! | Real Disk  | Host filesystem I/O (save/load to disk)       | Flamingo  |
//! | Signals    | `computer.pushSignal` instant markers         | Teal      |
//! | VK Fence   | `vkWaitForFences` blocking time               | Red       |
//! | VK Acquire | `vkAcquireNextImageKHR` blocking time         | Maroon    |
//! | VK Submit  | `vkQueueSubmit` call duration                 | Pink      |
//! | VK Present | `vkQueuePresentKHR` call duration              | Lavender  |
//! | Render     | Entire `App::render()` frame time             | Sapphire  |
//!
//! ## Instrumentation API
//!
//! ### Scoped measurement (RAII guard)
//!
//! ```rust
//! use crate::profiler::{self, Cat};
//!
//! fn expensive_operation() {
//!     let _guard = profiler::scope(Cat::Render, "render_frame");
//!     // ... work happens here ...
//!     // Event is automatically recorded with correct duration on drop.
//! }
//! ```
//!
//! ### Manual begin/end
//!
//! ```rust
//! let id = profiler::begin(Cat::VkFence, "wait_fence");
//! device.wait_for_fences(&[fence], true, u64::MAX);
//! profiler::end(id);
//! ```
//!
//! ### Instant events (zero-duration markers)
//!
//! ```rust
//! profiler::instant(Cat::Signal);
//! ```
//!
//! ### Budget sampling (GPU call budget fill level)
//!
//! ```rust
//! let pct = remaining_budget / max_budget;
//! profiler::budget(pct as f32);
//! ```
//!
//! ## Ring buffer design
//!
//! The ring buffer is a fixed-size array of `RING_CAP` (65536) events.
//! The write pointer advances monotonically and wraps via bitmask
//! (`write & RING_MASK`). There is no read pointer; queries scan the
//! entire buffer and filter by timestamp range. This is O(N) per frame
//! but N is bounded and the scan is cache-friendly (sequential access
//! over a contiguous array).
//!
//! ```text
//! ring[0]  ring[1]  ...  ring[write-1]  ring[write]  ...  ring[RING_CAP-1]
//!   ^                        ^               ^
//!   |                        |               |
//!   oldest (overwritten)     newest          next write
//! ```
//!
//! ## Timestamp source
//!
//! All timestamps are microseconds since `Profiler::epoch` (set to
//! `Instant::now()` at construction). `Instant::elapsed()` is used
//! rather than `SystemTime` because it is monotonic and not affected
//! by NTP adjustments or wall-clock changes.
//!
//! ## Colour theme
//!
//! Uses the Catppuccin Mocha palette, matching the settings GUI overlay.
//! Lane colours are chosen for maximum distinguishability:
//! warm colours for I/O operations, cool colours for GPU/Vulkan,
//! green for Lua, gradient for the budget meter.
//!
//! ## Thread safety
//!
//! The `Profiler` struct is stored in `thread_local!` and is never
//! shared between threads. The `ENABLED` flag is an `AtomicBool`
//! accessed with `Relaxed` ordering (sufficient because the profiler
//! state is only read/written from the main thread; the atomic is
//! merely a global on/off switch that avoids passing a reference
//! through every call site).
//!
//! ## Memory usage
//!
//! ```text
//! Ring buffer:  65536 events * 20 bytes = 1.25 MiB
//! (plus Vec overhead and alignment padding: ~1.3 MiB total)
//! ```
//!
//! ## Performance impact
//!
//! * **Disabled**: ~0.5 ns per instrumentation point (one `atomic::load`).
//! * **Enabled, not rendering**: ~50 ns per `scope()` (two `Instant::now()`
//!   calls plus one ring buffer write).
//! * **Enabled, rendering overlay**: ~0.5–2 ms per frame (ring buffer scan
//!   plus software rasterisation of the timeline). This is acceptable
//!   because the profiler overlay forces software rendering anyway.

use std::cell::RefCell;
use std::cmp::Ordering as CmpOrd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::display::font::{GlyphAtlas, ATLAS_SIZE, CELL_H, CELL_W};

// ═══════════════════════════════════════════════════════════════════════
// Global enable flag
// ═══════════════════════════════════════════════════════════════════════

/// Global on/off flag checked at every instrumentation point.
///
/// Uses `Relaxed` ordering because the profiler state is only ever
/// accessed from the main thread. The atomic merely allows checking
/// the flag without passing a reference through every function signature.
///
/// When `false`, all recording functions (`scope`, `begin`, `end`,
/// `instant`, `budget`) return immediately without reading the clock
/// or touching the ring buffer.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Thread-local profiler instance.
///
/// Stored in `thread_local!` to avoid any synchronisation overhead.
/// Only the main thread ever writes events; the audio thread and any
/// background threads do not interact with the profiler.
thread_local! {
    static PROFILER: RefCell<Profiler> = RefCell::new(Profiler::new());
}

// ═══════════════════════════════════════════════════════════════════════
// Event categories
// ═══════════════════════════════════════════════════════════════════════

/// Identifies which subsystem produced an event.
///
/// Each category corresponds to one horizontal lane in the timeline
/// visualisation. The discriminant values are used as array indices
/// into per-lane statistics arrays, so they must be contiguous starting
/// from 0.
///
/// # Adding a new category
///
/// 1. Add a variant here (with the next discriminant value).
/// 2. Increment `N_LANES`.
/// 3. Add the variant to the `LANES` array.
/// 4. Implement `label()` and `color()` for the new variant.
/// 5. Add `profiler::scope(Cat::NewCat, "description")` at the
///    appropriate call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cat {
    /// OC server tick (nominally 50 ms). One event per `App::tick()` call.
    Tick       = 0,
    /// Lua VM resume/yield cycle. One event per `step_kernel()` call.
    LuaStep    = 1,
    /// GPU component method invocation (`gpu.set`, `gpu.fill`, etc.).
    GpuCall    = 2,
    /// GPU call budget fill level (0.0 = empty, 1.0 = full).
    /// Rendered as a filled bar chart rather than discrete events.
    GpuBudget  = 3,
    /// Virtual filesystem operation (`filesystem.open`, `.read`, `.write`).
    FsOp       = 4,
    /// Unmanaged drive operation (`drive.readSector`, `.writeSector`).
    DriveOp    = 5,
    /// Real host filesystem I/O (drive persistence, state save/load).
    RealDisk   = 6,
    /// Signal pushed via `computer.pushSignal`. Rendered as point markers.
    Signal     = 7,
    /// Time spent blocked in `vkWaitForFences` (frame-in-flight sync).
    VkFence    = 8,
    /// Time spent in `vkAcquireNextImageKHR` (swapchain image acquisition).
    VkAcquire  = 9,
    /// Time spent in `vkQueueSubmit` (command buffer submission).
    VkSubmit   = 10,
    /// Time spent in `vkQueuePresentKHR` (swapchain presentation).
    VkPresent  = 11,
    /// Total time for the entire `App::render()` call (including Vulkan ops).
    Render     = 12,
}

/// Total number of event categories (lanes in the timeline).
pub const N_LANES: usize = 13;

/// All categories in display order (top to bottom in the timeline).
///
/// This array determines the vertical ordering of lanes. It must
/// contain exactly `N_LANES` elements, one per `Cat` variant.
pub const LANES: [Cat; N_LANES] = [
    Cat::Tick,      Cat::LuaStep,   Cat::GpuCall,   Cat::GpuBudget,
    Cat::FsOp,      Cat::DriveOp,   Cat::RealDisk,  Cat::Signal,
    Cat::VkFence,   Cat::VkAcquire, Cat::VkSubmit,  Cat::VkPresent,
    Cat::Render,
];

impl Cat {
    /// Human-readable label for this category, displayed in the left
    /// column of the timeline. Maximum 10 characters to fit the layout.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tick       => "Tick",
            Self::LuaStep    => "Lua Step",
            Self::GpuCall    => "GPU Calls",
            Self::GpuBudget  => "Budget",
            Self::FsOp       => "FS I/O",
            Self::DriveOp    => "Drive",
            Self::RealDisk   => "Real Disk",
            Self::Signal     => "Signals",
            Self::VkFence    => "VK Fence",
            Self::VkAcquire  => "VK Acquire",
            Self::VkSubmit   => "VK Submit",
            Self::VkPresent  => "VK Present",
            Self::Render     => "Render",
        }
    }

    /// RGB colour for this category's bars in the timeline.
    ///
    /// Colours are from the Catppuccin Mocha palette, chosen for
    /// visual distinctiveness across adjacent lanes.
    pub const fn color(self) -> u32 {
        match self {
            Self::Tick       => 0xCBA6F7, // Mauve
            Self::LuaStep    => 0xA6E3A1, // Green
            Self::GpuCall    => 0x89B4FA, // Blue
            Self::GpuBudget  => 0xA6E3A1, // Green (overridden by gradient)
            Self::FsOp       => 0xF9E2AF, // Yellow
            Self::DriveOp    => 0xFAB387, // Peach
            Self::RealDisk   => 0xF2CDCD, // Flamingo
            Self::Signal     => 0x94E2D5, // Teal
            Self::VkFence    => 0xF38BA8, // Red
            Self::VkAcquire  => 0xEBA0AC, // Maroon
            Self::VkSubmit   => 0xF5C2E7, // Pink
            Self::VkPresent  => 0xB4BEFE, // Lavender
            Self::Render     => 0x74C7EC, // Sapphire
        }
    }

    /// Array index for per-lane statistics. Equal to the discriminant.
    #[inline]
    pub const fn idx(self) -> usize { self as usize }
}

// ═══════════════════════════════════════════════════════════════════════
// Event structure
// ═══════════════════════════════════════════════════════════════════════

/// A single profiler event stored in the ring buffer.
///
/// Events are 20 bytes each (with padding). The ring buffer holds
/// 65536 events = ~1.3 MiB.
///
/// # Fields
///
/// * `start_us` — Microseconds since the profiler epoch (`Instant::now()`
///   at `Profiler` construction). Zero indicates an uninitialised slot.
///
/// * `duration_us` — Duration in microseconds. For instant events
///   (e.g. signals), this is 1 (minimum visible width).
///
/// * `cat` — Which lane this event belongs to.
///
/// * `value` — Auxiliary floating-point value. Currently only used by
///   `Cat::GpuBudget` to store the fill percentage (0.0–1.0).
///   Other categories ignore this field.
#[derive(Clone, Copy)]
pub struct Event {
    pub start_us:    u64,
    pub duration_us: u32,
    pub cat:         Cat,
    pub value:       f32,
}

impl Event {
    /// An empty/uninitialised event. `start_us == 0` marks it as invalid.
    const fn empty() -> Self {
        Self { start_us: 0, duration_us: 0, cat: Cat::Tick, value: 0.0 }
    }

    /// End timestamp (exclusive) in microseconds.
    #[inline]
    fn end_us(self) -> u64 {
        self.start_us + self.duration_us as u64
    }

    /// Whether this slot contains a valid event (non-zero start time).
    #[inline]
    fn is_valid(self) -> bool {
        self.start_us > 0
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Ring buffer parameters
// ═══════════════════════════════════════════════════════════════════════

/// Ring buffer capacity. Must be a power of two for bitmask wrapping.
const RING_CAP: usize = 1 << 16; // 65536 events

/// Bitmask for wrapping the write pointer: `write & RING_MASK`.
const RING_MASK: usize = RING_CAP - 1;

// ═══════════════════════════════════════════════════════════════════════
// Profiler state
// ═══════════════════════════════════════════════════════════════════════

/// The profiler state, including the event ring buffer and view parameters.
///
/// Stored in `thread_local!` and accessed via [`with`] / [`with_mut`].
/// Never shared between threads.
///
/// # View model
///
/// The timeline display is parameterised by two values:
///
/// * `window_us` — The width of the visible time window in microseconds
///   (default: 100,000 = 100 ms). Controlled by +/− keys.
///
/// * `scroll_us` — Signed offset from "now" in microseconds (always ≤ 0).
///   The visible window ends at `now + scroll_us`. When `scroll_us == 0`,
///   the right edge of the timeline is "now" (real-time following mode).
///   Left/Right arrow keys adjust this value.
pub struct Profiler {
    /// Fixed-size ring buffer of events.
    ///
    /// Slots with `start_us == 0` are uninitialised (not yet written).
    /// The write pointer wraps via `write & RING_MASK`.
    ring: Vec<Event>,

    /// Next write position in the ring buffer. Incremented on each
    /// `push()` call and wrapped via bitmask.
    write: usize,

    /// Time origin. All `start_us` values are relative to this instant.
    /// Set once at `Profiler::new()` and never changed.
    epoch: Instant,

    /// When `true`, new events are not recorded (but the view is still
    /// rendered from existing data). Toggled by the Space key.
    paused: bool,

    /// Whether the profiler overlay is currently visible.
    /// Controlled by the F7 key.
    pub visible: bool,

    /// Width of the visible time window in microseconds.
    ///
    /// Minimum: 5,000 (5 ms). Maximum: 10,000,000 (10 s).
    /// Default: 100,000 (100 ms).
    ///
    /// Adjusted by +/− keys (halved/doubled).
    window_us: u64,

    /// Signed scroll offset from "now" in microseconds.
    ///
    /// Always ≤ 0. When 0, the timeline tracks real-time.
    /// Left arrow decreases (scrolls into the past).
    /// Right arrow increases toward 0 (scrolls toward present).
    scroll_us: i64,
}

impl Profiler {
    /// Create a new profiler with an empty ring buffer.
    fn new() -> Self {
        Self {
            ring:      vec![Event::empty(); RING_CAP],
            write:     0,
            epoch:     Instant::now(),
            paused:    false,
            visible:   false,
            window_us: 100_000,
            scroll_us: 0,
        }
    }

    /// Current time in microseconds since the profiler epoch.
    #[inline]
    fn now_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// Record an event into the ring buffer, advancing the write pointer.
    ///
    /// Old events are silently overwritten when the buffer wraps.
    fn push(&mut self, ev: Event) {
        self.ring[self.write] = ev;
        self.write = (self.write + 1) & RING_MASK;
    }

    /// Iterate over all valid events whose time range overlaps `[lo, hi)`.
    ///
    /// This scans the entire ring buffer (O(N) where N = RING_CAP).
    /// With 65536 events at 20 bytes each, the scan is cache-friendly
    /// and completes in well under 1 ms on modern hardware.
    fn events_in(&self, lo: u64, hi: u64) -> impl Iterator<Item = &Event> {
        self.ring.iter().filter(move |e| {
            e.is_valid() && e.start_us < hi && e.end_us() > lo
        })
    }

    /// Compute the visible time window as `(start_us, end_us)`.
    ///
    /// The window ends at `now + scroll_us` and extends `window_us`
    /// microseconds into the past.
    fn visible_window(&self) -> (u64, u64) {
        let now = self.now_us();
        let end = (now as i64 + self.scroll_us).max(0) as u64;
        let start = end.saturating_sub(self.window_us);
        (start, end)
    }

    /// Compute per-lane statistics for events within `[lo, hi)`.
    ///
    /// Returns an array indexed by `Cat::idx()`, where each element is:
    /// `(event_count, total_duration_us, last_budget_value)`.
    ///
    /// The `last_budget_value` field is only meaningful for `Cat::GpuBudget`.
    fn lane_stats(&self, lo: u64, hi: u64) -> [(u32, u64, f32); N_LANES] {
        let mut stats = [(0u32, 0u64, 0.0f32); N_LANES];
        for ev in self.events_in(lo, hi) {
            let i = ev.cat.idx();
            stats[i].0 += 1;
            stats[i].1 += ev.duration_us as u64;
            if ev.cat == Cat::GpuBudget {
                stats[i].2 = ev.value;
            }
        }
        stats
    }

    // ───────────────────────────────────────────────────────────────
    // Keyboard input handling
    // ───────────────────────────────────────────────────────────────

    /// Handle a key press while the profiler is visible.
    ///
    /// Returns `true` if the key was consumed (caller should not
    /// propagate it further).
    ///
    /// If the profiler is not visible, only F7 is handled (to open it).
    pub fn handle_key(&mut self, kc: winit::keyboard::KeyCode) -> bool {
        use winit::keyboard::KeyCode as K;

        if !self.visible {
            if kc == K::F7 {
                self.visible = true;
                ENABLED.store(true, Ordering::Relaxed);
                return true;
            }
            return false;
        }

        match kc {
            K::F7 => {
                self.visible = false;
                ENABLED.store(false, Ordering::Relaxed);
            }
            K::Space => {
                self.paused = !self.paused;
            }
            K::ArrowLeft => {
                self.scroll_us -= self.window_us as i64 / 4;
            }
            K::ArrowRight => {
                self.scroll_us = (self.scroll_us + self.window_us as i64 / 4).min(0);
            }
            K::Equal | K::NumpadAdd => {
                self.window_us = (self.window_us / 2).max(5_000);
            }
            K::Minus | K::NumpadSubtract => {
                self.window_us = (self.window_us * 2).min(10_000_000);
            }
            K::KeyR => {
                self.ring.fill(Event::empty());
                self.write = 0;
                self.scroll_us = 0;
            }
            _ => return false,
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scope guard (RAII)
// ═══════════════════════════════════════════════════════════════════════

/// Opaque handle returned by [`begin`], passed to [`end`] to finalise
/// the event's duration. Contains the ring buffer index of the event.
pub struct ScopeId(usize);

/// RAII guard that calls [`end`] on drop, recording the event duration.
///
/// Created by [`scope`]. If profiling is disabled, the inner `Option`
/// is `None` and the drop is a no-op.
///
/// # Example
///
/// ```rust
/// {
///     let _guard = profiler::scope(Cat::Render, "frame");
///     // ... rendering code ...
/// }   // <-- duration is recorded here when _guard is dropped
/// ```
pub struct Guard(Option<ScopeId>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(id) = self.0.take() {
            end(id);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Public instrumentation API
// ═══════════════════════════════════════════════════════════════════════

/// Check whether the profiler is currently recording.
///
/// This is the hot-path guard: a single `atomic::load(Relaxed)`.
/// All instrumentation functions check this first and return
/// immediately if profiling is disabled.
///
/// # Performance
///
/// ~0.5 ns on x86-64 (same cost as reading a `bool` from L1 cache).
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Begin a scoped measurement and return an RAII guard.
///
/// When the returned [`Guard`] is dropped, the event's duration is
/// automatically recorded. This is the preferred instrumentation method
/// for functions with a single entry and exit point.
///
/// # Arguments
///
/// * `cat` — The event category (determines which lane it appears in).
/// * `_label` — A static label string (reserved for future use in
///   tooltip display; currently unused but kept for API consistency
///   with Chrome Tracing / Tracy conventions).
///
/// # Returns
///
/// A `Guard` that records the event duration on drop. If profiling is
/// disabled, returns a no-op guard (zero cost).
///
/// # Example
///
/// ```rust
/// fn expensive_function() {
///     let _prof = profiler::scope(Cat::LuaStep, "step_kernel");
///     // ... work ...
/// } // duration recorded here
/// ```
pub fn scope(cat: Cat, _label: &'static str) -> Guard {
    if !is_enabled() {
        return Guard(None);
    }
    Guard(Some(begin(cat, _label)))
}

/// Begin a measurement manually. Returns a [`ScopeId`] that must be
/// passed to [`end`] to finalise the event.
///
/// Prefer [`scope`] when possible. Use `begin`/`end` only when the
/// measurement spans a non-RAII code path (e.g. across an FFI boundary).
///
/// # Arguments
///
/// * `cat` — The event category.
/// * `_label` — Static label (reserved for future use).
///
/// # Returns
///
/// A `ScopeId` containing the ring buffer index of the newly created
/// event. Pass this to [`end`] to set the duration.
pub fn begin(cat: Cat, _label: &'static str) -> ScopeId {
    with_mut(|p| {
        if p.paused {
            return ScopeId(0);
        }
        let idx = p.write;
        p.push(Event {
            start_us:    p.now_us(),
            duration_us: 0,
            cat,
            value:       0.0,
        });
        ScopeId(idx)
    })
}

/// Finalise a measurement started by [`begin`].
///
/// Computes the elapsed time since `begin` was called and stores it
/// in the event's `duration_us` field.
///
/// # Arguments
///
/// * `id` — The `ScopeId` returned by [`begin`].
pub fn end(id: ScopeId) {
    with_mut(|p| {
        let now = p.now_us();
        let ev = &mut p.ring[id.0 & RING_MASK];
        ev.duration_us = now.saturating_sub(ev.start_us) as u32;
    });
}

/// Record an instant event (zero-duration marker).
///
/// Useful for discrete occurrences like signal pushes, where there is
/// no meaningful duration. Rendered as a single-pixel-wide bar in the
/// timeline (or the minimum visible width at the current zoom level).
///
/// # Arguments
///
/// * `cat` — The event category.
pub fn instant(cat: Cat) {
    if !is_enabled() {
        return;
    }
    with_mut(|p| {
        if p.paused {
            return;
        }
        p.push(Event {
            start_us:    p.now_us(),
            duration_us: 1, // 1 µs minimum so it renders as a visible dot
            cat,
            value:       0.0,
        });
    });
}

/// Record a GPU call budget sample.
///
/// The budget lane is rendered differently from other lanes: instead of
/// discrete bars, it draws a continuous fill-level chart (like a
/// VU meter or capacity gauge).
///
/// # Arguments
///
/// * `pct` — Fill percentage, clamped to `0.0..=1.0`.
///   - 0.0 = budget exhausted (all calls consumed).
///   - 1.0 = budget full (just reset at tick start).
pub fn budget(pct: f32) {
    if !is_enabled() {
        return;
    }
    with_mut(|p| {
        if p.paused {
            return;
        }
        p.push(Event {
            start_us:    p.now_us(),
            duration_us: 0,
            cat:         Cat::GpuBudget,
            value:       pct.clamp(0.0, 1.0),
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Thread-local access helpers
// ═══════════════════════════════════════════════════════════════════════

/// Immutable access to the thread-local profiler.
pub fn with<R>(f: impl FnOnce(&Profiler) -> R) -> R {
    PROFILER.with(|p| f(&p.borrow()))
}

/// Mutable access to the thread-local profiler.
pub fn with_mut<R>(f: impl FnOnce(&mut Profiler) -> R) -> R {
    PROFILER.with(|p| f(&mut p.borrow_mut()))
}

/// Check whether the profiler overlay is currently visible.
///
/// Called by the render path to decide whether to draw the overlay.
pub fn is_visible() -> bool {
    with(|p| p.visible)
}

/// Forward a key event to the profiler. Returns `true` if consumed.
///
/// Should be called from the main event loop before other key handlers.
pub fn handle_key(kc: winit::keyboard::KeyCode) -> bool {
    with_mut(|p| p.handle_key(kc))
}

// ═══════════════════════════════════════════════════════════════════════
// Overlay rendering
// ═══════════════════════════════════════════════════════════════════════

/// Catppuccin Mocha colour constants for the profiler overlay.
///
/// Kept in a private sub-module to avoid polluting the parent namespace
/// and to clearly separate data from logic.
mod colours {
    /// Near-black background (Crust).
    pub const BG: u32 = 0x11111B;
    /// Panel background (Mantle).
    pub const PANEL: u32 = 0x181825;
    /// Border colour (Surface0).
    pub const BORDER: u32 = 0x313244;
    /// Primary text colour (Text).
    pub const TEXT: u32 = 0xCDD6F4;
    /// Dimmed text for hints and secondary info (Overlay0).
    pub const DIM: u32 = 0x6C7086;
    /// Accent colour for the title (Blue).
    pub const ACCENT: u32 = 0x89B4FA;
    /// Time ruler tick marks and separators (Surface1).
    pub const RULER: u32 = 0x45475A;
    /// Even-row lane background (Base).
    pub const LANE_BG: u32 = 0x1E1E2E;
    /// Odd-row lane background (slightly darker than Base).
    pub const LANE_ALT: u32 = 0x1A1A28;
    /// Warning colour for low budget (Red).
    pub const WARN: u32 = 0xF38BA8;
    /// Healthy colour for high budget (Green).
    pub const GOOD: u32 = 0xA6E3A1;
    /// Mid-range colour for medium budget (Yellow).
    pub const MID: u32 = 0xF9E2AF;
}

/// Compute the UI scale factor based on window height.
///
/// Returns 2 for displays >= 1400px tall (HiDPI), 1 otherwise.
/// This keeps the profiler readable on both 1080p and 4K displays.
fn compute_scale(window_height: u32) -> u32 {
    if window_height >= 1400 { 2 } else { 1 }
}

/// Render the profiler overlay onto a software pixel buffer.
///
/// Called from `App::render()` when `profiler::is_visible()` is true.
/// The overlay is drawn on top of whatever is already in the pixel buffer
/// (the OC screen, debug bar, settings GUI, etc.).
///
/// # Arguments
///
/// * `atlas` — The glyph atlas for text rendering.
/// * `pixels` — The software framebuffer (format: `0x00RRGGBB`).
/// * `win_w` — Window width in pixels.
/// * `win_h` — Window height in pixels.
pub fn render_overlay(
    atlas: &GlyphAtlas,
    pixels: &mut [u32],
    win_w: u32,
    win_h: u32,
) {
    with(|p| draw_overlay(p, atlas, pixels, win_w, win_h));
}

/// Internal overlay drawing function.
///
/// Computes the panel layout, draws the background, time ruler, event
/// lanes, statistics column, summary bar, and help text.
///
/// # Layout computation
///
/// The panel is centred vertically in the window. Its width is
/// `window_width - 2 * margin`. The layout is divided into three
/// columns:
///
/// ```text
/// | label (11 chars) | timeline (flexible) | stats (14 chars) |
/// ```
///
/// The timeline column fills all remaining horizontal space.
fn draw_overlay(
    p: &Profiler,
    atlas: &GlyphAtlas,
    px: &mut [u32],
    ww: u32,
    wh: u32,
) {
    let scale = compute_scale(wh);
    let cw = 8 * scale;  // character width in pixels
    let ch = 16 * scale; // character height in pixels
    let row_h = ch + 2 * scale; // row height including padding

    // Column widths in character units
    let label_chars: u32 = 11;
    let stats_chars: u32 = 14;
    let total_cols = ww / cw;
    if total_cols < label_chars + stats_chars + 10 {
        return; // window too narrow to render anything useful
    }
    let timeline_chars = total_cols - label_chars - stats_chars - 2;

    // Panel dimensions in pixels
    let margin = cw;
    let panel_w = ww - margin * 2;
    let header_rows: u32 = 3;
    let lane_rows = N_LANES as u32;
    let footer_rows: u32 = 3;
    let total_rows = header_rows + lane_rows + footer_rows;
    let panel_h = total_rows * row_h + 4 * scale;

    // Panel origin (centred vertically)
    let ox = margin;
    let oy = wh.saturating_sub(panel_h) / 2;

    // Column X origins in pixels
    let label_x = ox + cw;
    let timeline_x = ox + (label_chars + 1) * cw;
    let timeline_w_px = timeline_chars * cw;
    let stats_x = timeline_x + timeline_w_px + cw;

    // Visible time window
    let (win_lo, win_hi) = p.visible_window();
    let win_dur = (win_hi - win_lo).max(1);
    let lane_stats = p.lane_stats(win_lo, win_hi);

    // Border + panel background
    let border_size = 2 * scale;
    rect(px, ww, wh,
         ox.wrapping_sub(border_size), oy.wrapping_sub(border_size),
         panel_w + border_size * 2, panel_h + border_size * 2,
         colours::BORDER);
    rect(px, ww, wh, ox, oy, panel_w, panel_h, colours::PANEL);

    let mut y = oy + 2 * scale;

    // ── Title row ──────────────────────────────────────────────
    let title = if p.paused { "PROFILER [PAUSED]" } else { "PROFILER" };
    stext(atlas, px, ww, wh, label_x, y, title, colours::ACCENT, colours::PANEL, scale);

    let controls = format!(
        "F7:close Space:{} +/-:zoom <->:scroll  {:.0}ms",
        if p.paused { "resume" } else { "pause " },
        win_dur as f64 / 1000.0
    );
    stext(atlas, px, ww, wh, stats_x, y, &controls, colours::DIM, colours::PANEL, scale);
    y += row_h;

    // ── Top separator ──────────────────────────────────────────
    let sep_w = panel_w - 2 * cw;
    rect(px, ww, wh, label_x, y + ch / 2, sep_w, scale.max(1), colours::RULER);
    y += row_h / 2;

    // ── Time ruler ─────────────────────────────────────────────
    draw_time_ruler(atlas, px, ww, wh, timeline_x, y, timeline_w_px,
                    win_lo, win_hi, win_dur, scale);
    y += row_h;

    // ── Event lanes ────────────────────────────────────────────
    for (lane_idx, &cat) in LANES.iter().enumerate() {
        let lane_bg = if lane_idx % 2 == 0 { colours::LANE_BG } else { colours::LANE_ALT };
        rect(px, ww, wh, timeline_x, y, timeline_w_px, row_h, lane_bg);

        // Lane label
        stext(atlas, px, ww, wh, label_x, y + scale,
              cat.label(), colours::TEXT, colours::PANEL, scale);

        // Draw events for this lane
        if cat == Cat::GpuBudget {
            draw_budget_lane(p, px, ww, wh, timeline_x, y, timeline_w_px,
                             row_h, win_lo, win_dur, scale);
        } else {
            draw_event_bars(p, px, ww, wh, timeline_x, y, timeline_w_px,
                            row_h, win_lo, win_dur, cat, scale);
        }

        // Per-lane statistics
        let stat_str = format_lane_stat(cat, &lane_stats);
        stext(atlas, px, ww, wh, stats_x, y + scale,
              &stat_str, colours::DIM, colours::PANEL, scale);

        y += row_h;
    }

    // ── Bottom separator ───────────────────────────────────────
    rect(px, ww, wh, label_x, y + ch / 2, sep_w, scale.max(1), colours::RULER);
    y += row_h / 2;

    // ── Summary line ───────────────────────────────────────────
    let total_events: u32 = lane_stats.iter().map(|s| s.0).sum();
    let vk_total_us = lane_stats[Cat::VkFence.idx()].1
        + lane_stats[Cat::VkSubmit.idx()].1
        + lane_stats[Cat::VkPresent.idx()].1;
    let summary = format!(
        "Events: {} | Lua: {} | GPU: {} | FS: {} | Drive: {} | VK: {:.2}ms | Win: {:.1}ms",
        total_events,
        lane_stats[Cat::LuaStep.idx()].0,
        lane_stats[Cat::GpuCall.idx()].0,
        lane_stats[Cat::FsOp.idx()].0,
        lane_stats[Cat::DriveOp.idx()].0,
        vk_total_us as f64 / 1000.0,
        win_dur as f64 / 1000.0,
    );
    stext(atlas, px, ww, wh, label_x, y, &summary, colours::TEXT, colours::PANEL, scale);
    y += row_h;

    // ── Help line ──────────────────────────────────────────────
    stext(atlas, px, ww, wh, label_x, y,
          "F7: close | Space: pause | +/-: zoom | <->: scroll | R: reset",
          colours::DIM, colours::PANEL, scale);
}

/// Draw the time ruler with tick marks and labels.
///
/// Tick marks are evenly spaced at intervals determined by the visible
/// window width (e.g. every 10 ms for a 100 ms window). Each mark has
/// a 1-pixel-wide vertical line and a time label (e.g. "30ms").
fn draw_time_ruler(
    atlas: &GlyphAtlas,
    px: &mut [u32],
    ww: u32, wh: u32,
    tl_x: u32, y: u32, tl_w: u32,
    win_lo: u64, win_hi: u64, win_dur: u64,
    scale: u32,
) {
    let ch = 16 * scale;
    let cw = 8 * scale;
    let tick_us = ruler_tick_interval(win_dur);
    let first = (win_lo / tick_us + 1) * tick_us;
    let mut t = first;
    while t < win_hi {
        let frac = (t - win_lo) as f64 / win_dur as f64;
        let xp = tl_x + (frac * tl_w as f64) as u32;
        rect(px, ww, wh, xp, y, scale.max(1), ch, colours::RULER);
        let label = format!("{:.0}ms", t as f64 / 1000.0);
        if xp + label.len() as u32 * cw < tl_x + tl_w {
            stext(atlas, px, ww, wh, xp + 2 * scale, y, &label,
                  colours::DIM, colours::PANEL, scale);
        }
        t += tick_us;
    }
}

/// Draw coloured bars for duration-based events in a single lane.
///
/// Each event whose time range intersects the visible window is drawn
/// as a filled rectangle. The bar width is proportional to the event's
/// duration, with a minimum width of `scale` pixels (so even sub-µs
/// events are visible as single-pixel dots).
fn draw_event_bars(
    p: &Profiler,
    px: &mut [u32],
    ww: u32, wh: u32,
    tl_x: u32, y: u32, tl_w: u32, row_h: u32,
    win_lo: u64, win_dur: u64,
    cat: Cat,
    scale: u32,
) {
    let win_hi = win_lo + win_dur;
    let bar_y = y + 2 * scale;
    let bar_h = row_h.saturating_sub(4 * scale);
    let colour = cat.color();

    for ev in p.events_in(win_lo, win_hi) {
        if ev.cat != cat {
            continue;
        }
        let x0_frac = ev.start_us.saturating_sub(win_lo) as f64 / win_dur as f64;
        let x1_frac = ev.end_us().saturating_sub(win_lo) as f64 / win_dur as f64;
        let x0 = tl_x + (x0_frac * tl_w as f64) as u32;
        let x1 = tl_x + (x1_frac * tl_w as f64).ceil() as u32;
        let bar_w = x1.saturating_sub(x0).max(scale); // min visible width
        let clipped_w = bar_w.min(tl_x + tl_w - x0);
        if x0 < tl_x + tl_w {
            rect(px, ww, wh, x0, bar_y, clipped_w, bar_h, colour);
        }
    }
}

/// Draw the GPU budget lane as a continuous fill-level chart.
///
/// Unlike other lanes, the budget lane does not draw discrete event bars.
/// Instead, it draws a vertical column for each pixel of the timeline,
/// with the column height proportional to the budget fill level at that
/// point in time. The colour grades from red (low) through yellow (mid)
/// to green (full).
///
/// Budget samples are collected from `Cat::GpuBudget` events via binary
/// search (samples are sorted by time).
fn draw_budget_lane(
    p: &Profiler,
    px: &mut [u32],
    ww: u32, wh: u32,
    tl_x: u32, y: u32, tl_w: u32, row_h: u32,
    win_lo: u64, win_dur: u64,
    scale: u32,
) {
    let bar_y = y + 2 * scale;
    let bar_h = row_h.saturating_sub(4 * scale);
    if bar_h == 0 {
        return;
    }

    let win_hi = win_lo + win_dur;

    // Collect and sort budget samples by fractional position within the window
    let mut samples: Vec<(f64, f32)> = Vec::new();
    for ev in p.events_in(win_lo, win_hi) {
        if ev.cat != Cat::GpuBudget {
            continue;
        }
        let frac = ev.start_us.saturating_sub(win_lo) as f64 / win_dur as f64;
        samples.push((frac, ev.value));
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(CmpOrd::Equal));

    // For each pixel column, find the nearest sample (to the left) via binary search
    for px_x in 0..tl_w {
        let frac = px_x as f64 / tl_w as f64;
        let val = match samples.binary_search_by(|s|
            s.0.partial_cmp(&frac).unwrap_or(CmpOrd::Equal)
        ) {
            Ok(i) => samples[i].1,
            Err(0) => samples.first().map(|s| s.1).unwrap_or(1.0),
            Err(i) => samples[i - 1].1,
        };

        let fill_h = (val * bar_h as f32) as u32;
        let colour = budget_fill_colour(val);
        let col_x = tl_x + px_x;
        let col_y = bar_y + bar_h - fill_h;
        if fill_h > 0 {
            rect(px, ww, wh, col_x, col_y, 1, fill_h, colour);
        }
    }
}

/// Format a per-lane statistic string for the right-hand column.
///
/// Different categories format their stats differently:
/// - Budget: percentage ("67%")
/// - Signal: count only ("89")
/// - Duration-based: count + average ("156 0.3ms" or "12 45µs")
fn format_lane_stat(cat: Cat, stats: &[(u32, u64, f32); N_LANES]) -> String {
    let (count, total_us, value) = stats[cat.idx()];
    if cat == Cat::GpuBudget {
        return format!("{:.0}%", value * 100.0);
    }
    if cat == Cat::Signal {
        return format!("{count}");
    }
    if total_us > 0 && count > 0 {
        let avg_us = total_us as f64 / count as f64;
        if avg_us >= 1000.0 {
            format!("{count} {:.1}ms", avg_us / 1000.0)
        } else {
            format!("{count} {:.0}us", avg_us)
        }
    } else {
        format!("{count}")
    }
}

/// Compute the colour for a budget fill level using a three-stop gradient.
///
/// The gradient transitions:
/// - 0.00–0.33: dark red to warn (indicates budget is nearly exhausted)
/// - 0.33–0.66: warn to mid-yellow
/// - 0.66–1.00: mid-yellow to green (healthy budget)
fn budget_fill_colour(pct: f32) -> u32 {
    if pct > 0.66 {
        lerp_colour(colours::MID, colours::GOOD, (pct - 0.66) / 0.34)
    } else if pct > 0.33 {
        lerp_colour(colours::WARN, colours::MID, (pct - 0.33) / 0.33)
    } else {
        lerp_colour(0xBA1A1A, colours::WARN, pct / 0.33)
    }
}

/// Linearly interpolate between two RGB colours.
///
/// `t` is clamped to `[0.0, 1.0]`. At `t=0.0` returns `a`, at
/// `t=1.0` returns `b`.
fn lerp_colour(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |shift: u32| {
        let va = ((a >> shift) & 0xFF) as f32;
        let vb = ((b >> shift) & 0xFF) as f32;
        (va + (vb - va) * t) as u32
    };
    (mix(16) << 16) | (mix(8) << 8) | mix(0)
}

/// Choose the time ruler tick interval (in µs) based on the visible
/// window width.
///
/// The goal is to have roughly 5–10 tick marks visible at any zoom level.
///
/// | Window width | Tick interval |
/// |-------------|---------------|
/// | ≤ 20 ms     | 2 ms          |
/// | 21–50 ms    | 5 ms          |
/// | 51–200 ms   | 10 ms         |
/// | 201–500 ms  | 50 ms         |
/// | 501 ms–2 s  | 100 ms        |
/// | > 2 s       | 500 ms        |
fn ruler_tick_interval(window_us: u64) -> u64 {
    let ms = window_us / 1000;
    match ms {
        0..=20     => 2_000,
        21..=50    => 5_000,
        51..=200   => 10_000,
        201..=500  => 50_000,
        501..=2000 => 100_000,
        _          => 500_000,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Low-level software drawing primitives
// ═══════════════════════════════════════════════════════════════════════

/// Fill a rectangle with a solid colour.
///
/// Coordinates are clipped to the framebuffer bounds. Out-of-bounds
/// pixels are silently skipped.
///
/// # Arguments
///
/// * `px` — Pixel buffer (row-major, format `0x00RRGGBB`).
/// * `ww`, `wh` — Framebuffer dimensions.
/// * `x`, `y` — Top-left corner of the rectangle.
/// * `w`, `h` — Width and height in pixels.
/// * `colour` — Fill colour.
fn rect(px: &mut [u32], ww: u32, wh: u32, x: u32, y: u32, w: u32, h: u32, colour: u32) {
    for ry in y..y.saturating_add(h).min(wh) {
        let off = (ry * ww) as usize;
        for rx in x..x.saturating_add(w).min(ww) {
            let i = off + rx as usize;
            if i < px.len() {
                px[i] = colour;
            }
        }
    }
}

/// Render a text string at a pixel position with a given scale factor.
///
/// Each character is drawn using the glyph atlas. Characters that would
/// extend past the right edge of the framebuffer are skipped.
///
/// # Arguments
///
/// * `atlas` — Glyph atlas for character bitmaps.
/// * `px` — Pixel buffer.
/// * `ww`, `wh` — Framebuffer dimensions.
/// * `x`, `y` — Top-left pixel position of the first character.
/// * `text` — The string to render.
/// * `fg`, `bg` — Foreground and background RGB colours.
/// * `scale` — Scale factor (1 = native, 2 = double size, etc.).
fn stext(
    atlas: &GlyphAtlas,
    px: &mut [u32],
    ww: u32, wh: u32,
    mut x: u32, y: u32,
    text: &str,
    fg: u32, bg: u32,
    scale: u32,
) {
    let cw = 8 * scale;
    for ch in text.chars() {
        if x + cw > ww {
            break;
        }
        sglyph(atlas, px, ww, wh, x, y, ch as u32, fg, bg, scale);
        x += cw;
    }
}

/// Render a single glyph from the atlas at a pixel position with scaling.
///
/// Each pixel of the 8×16 glyph bitmap is expanded to a `scale × scale`
/// block of screen pixels. The atlas is consulted to determine whether
/// each pixel is foreground (> 128) or background.
///
/// # Arguments
///
/// * `atlas` — Glyph atlas.
/// * `px` — Pixel buffer.
/// * `ww`, `wh` — Framebuffer dimensions.
/// * `x`, `y` — Top-left screen position.
/// * `cp` — Unicode code point to render.
/// * `fg`, `bg` — Foreground and background RGB colours.
/// * `scale` — Scale factor.
fn sglyph(
    atlas: &GlyphAtlas,
    px: &mut [u32],
    ww: u32, wh: u32,
    x: u32, y: u32,
    cp: u32,
    fg: u32, bg: u32,
    scale: u32,
) {
    let atlas_col = cp & 0xFF;
    let atlas_row = (cp >> 8) & 0xFF;
    for gy in 0..CELL_H {
        for gx in 0..8u32 {
            let ax = (atlas_col * CELL_W + gx) as usize;
            let ay = (atlas_row * CELL_H + gy) as usize;
            let hit = ax < ATLAS_SIZE as usize
                && ay < ATLAS_SIZE as usize
                && atlas.pixels[ay * ATLAS_SIZE as usize + ax] > 128;
            let colour = if hit { fg } else { bg };
            for sy in 0..scale {
                for sx in 0..scale {
                    let px_x = x + gx * scale + sx;
                    let px_y = y + gy * scale + sy;
                    if px_x < ww && px_y < wh {
                        let idx = (px_y * ww + px_x) as usize;
                        if idx < px.len() {
                            px[idx] = colour;
                        }
                    }
                }
            }
        }
    }
}