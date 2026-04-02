#![allow(warnings)]
//! # Main entry point
//!
//! This file contains:
//!
//! * The `App` struct (the central application state).
//! * The `winit` `ApplicationHandler` implementation (event loop).
//! * Window creation, Vulkan/softbuffer initialisation.
//! * The per-tick Lua stepping loop.
//! * The per-frame rendering dispatch (software / Vulkan INDIRECT / DIRECT).
//! * LWJGL2 key code mapping for OpenComputers compatibility.
//! * Filesystem loading from `assets/openos/`.
//! * Bluescreen-of-death rendering for unrecoverable errors.
//!
//! ## Event loop architecture
//!
//! ```text
//! EventLoop::run_app(&mut app)
//!   |
//!   | winit dispatches events
//!   v
//! ApplicationHandler::window_event()
//!   |
//!   +-- CloseRequested -> exit
//!   +-- KeyboardInput -> forward to OC / settings GUI
//!   +-- Resized -> recreate swapchain
//!   +-- RedrawRequested:
//!   |     |
//!   |     +-- tick() if 50ms elapsed
//!   |     +-- render()
//!   |     +-- request_redraw() (continuous loop)
//!   |
//!   +-- about_to_wait -> request_redraw()
//! ```
//!
//! ## Tick vs. frame
//!
//! * **Tick**: 50 ms interval (20 tps). Advances the Lua VM.
//! * **Frame**: As fast as possible (vsync or FPS limit). Renders
//!   the current TextBuffer to the screen.
//!
//! Multiple frames may be rendered between ticks (at high FPS).
//! Multiple Lua resume/yield cycles may happen within a single tick
//! (up to `MAX_STEPS_PER_TICK` or `TICK_BUDGET` wall time).
//!
//! ## Key code mapping
//!
//! OpenComputers expects LWJGL2 key codes (inherited from Minecraft).
//! The `winit_to_lwjgl` function maps from `winit::keyboard::KeyCode`
//! to the LWJGL2 integer codes. This is a large match statement
//! covering all common keys.
//!
//! ## Render mode cycling
//!
//! F5 cycles through: Software -> Vulkan INDIRECT -> Vulkan DIRECT.
//! If Vulkan is not available, INDIRECT/DIRECT are skipped.
//! If the DIRECT renderer failed to initialise, it is also skipped.

pub mod config;
pub mod display;
pub mod machine;
pub mod components;
pub mod render;
pub mod lua;
pub mod fs;
pub mod overlay;
pub mod sound;
pub mod settings_file;
pub mod logging;
pub mod profiler;

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread::Thread;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode as WinitKey;
use winit::window::{Window, WindowId, WindowAttributes};

use config::OcConfig;
use display::font::GlyphAtlas;
use lua::host::{EmulatorState, ExecResult};
use machine::signal::Signal;

use render::software;
use render::RenderMode;
use render::direct::DirectRenderer;
use overlay::build_debug_cells;
use crate::display::TextBuffer;

/// Server tick interval: 50 ms = 20 ticks per second.
///
/// Matches Minecraft's server tick rate, which is what OC's timing
/// is based on.
const TICK: Duration = Duration::from_millis(50);

/// Maximum wall-clock time per tick for Lua processing.
///
/// If the Lua VM takes longer than this within a single tick, the
/// stepping loop breaks early to avoid blocking rendering.
const TICK_BUDGET: Duration = Duration::from_millis(15);

/// Maximum number of resume/yield cycles per tick.
///
/// Prevents infinite loops in the case where Lua yields with zero
/// sleep and there are always signals pending.
const MAX_STEPS_PER_TICK: u32 = 512;

// ═══════════════════════════════════════════════════════════════════════
// LWJGL2 key-code mapping
// ═══════════════════════════════════════════════════════════════════════

fn winit_to_lwjgl(key: WinitKey) -> u32 {
    use WinitKey::*;
    match key {
        Escape=>1,Digit1=>2,Digit2=>3,Digit3=>4,Digit4=>5,Digit5=>6,
        Digit6=>7,Digit7=>8,Digit8=>9,Digit9=>10,Digit0=>11,
        Minus=>12,Equal=>13,Backspace=>14,Tab=>15,
        KeyQ=>16,KeyW=>17,KeyE=>18,KeyR=>19,KeyT=>20,KeyY=>21,KeyU=>22,
        KeyI=>23,KeyO=>24,KeyP=>25,BracketLeft=>26,BracketRight=>27,
        Enter=>28,ControlLeft=>29,
        KeyA=>30,KeyS=>31,KeyD=>32,KeyF=>33,KeyG=>34,KeyH=>35,KeyJ=>36,
        KeyK=>37,KeyL=>38,Semicolon=>39,Quote=>40,Backquote=>41,
        ShiftLeft=>42,Backslash=>43,
        KeyZ=>44,KeyX=>45,KeyC=>46,KeyV=>47,KeyB=>48,KeyN=>49,KeyM=>50,
        Comma=>51,Period=>52,Slash=>53,ShiftRight=>54,
        NumpadMultiply=>55,AltLeft=>56,Space=>57,CapsLock=>58,
        F1=>59,F2=>60,F3=>61,F4=>62,F5=>63,F6=>64,F7=>65,F8=>66,
        F9=>67,F10=>68,NumLock=>69,ScrollLock=>70,
        Numpad7=>71,Numpad8=>72,Numpad9=>73,NumpadSubtract=>74,
        Numpad4=>75,Numpad5=>76,Numpad6=>77,NumpadAdd=>78,
        Numpad1=>79,Numpad2=>80,Numpad3=>81,Numpad0=>82,
        NumpadDecimal=>83,F11=>87,F12=>88,
        NumpadEnter=>156,ControlRight=>157,NumpadDivide=>181,
        AltRight=>184,Home=>199,ArrowUp=>200,PageUp=>201,
        ArrowLeft=>203,ArrowRight=>205,End=>207,ArrowDown=>208,
        PageDown=>209,Insert=>210,Delete=>211,
        _=>0,
    }
}

fn get_char_code(event: &winit::event::KeyEvent) -> u16 {
    if let Some(text) = event.text.as_ref() {
        if let Some(ch) = text.chars().next() {
            return match ch {
                '\r'|'\n' => 13, '\t' => 9, '\x08' => 8,
                c => c as u16,
            };
        }
    }
    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
        use WinitKey::*;
        return match code {
            Space=>32, Enter|NumpadEnter=>13, Backspace=>8,
            Tab=>9, Delete=>127, Escape=>27, _=>0,
        };
    }
    0
}

// ═══════════════════════════════════════════════════════════════════════
// Filesystem loader
// ═══════════════════════════════════════════════════════════════════════

fn load_directory_recursive(
    base: &std::path::Path,
    prefix: &str,
    files: &mut HashMap<String, Vec<u8>>,
) {
    for entry in std::fs::read_dir(base).into_iter().flatten().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let key = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
        if path.is_dir() {
            load_directory_recursive(&path, &key, files);
        } else if let Ok(data) = std::fs::read(&path) {
            files.insert(key, data);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Vulkan wrapper (context + renderer, correct drop order)
// ═══════════════════════════════════════════════════════════════════════

struct VulkanState {
    ctx: render::vulkan_ctx::VulkanContext,
    //renderer: render::indirect::IndirectRenderer,
    indirect: render::indirect::IndirectRenderer,
    direct: Option<render::direct::DirectRenderer>,
}

impl Drop for VulkanState {
    fn drop(&mut self) {
        //self.renderer.destroy(&self.ctx.device);
        if let Some(d) = self.direct.as_mut() { d.destroy(&self.ctx.device); }
        self.indirect.destroy(&self.ctx.device);
    }
}

impl VulkanState {
    /// Try to create Vulkan state.  Returns `None` on any failure.
    fn try_new(window: &Window, atlas: &GlyphAtlas) -> Option<Self> {
        unsafe {
            let sz = window.inner_size();
            let mut ctx = match render::vulkan_ctx::VulkanContext::new(
                window, window, sz.width.max(1), sz.height.max(1),
            ) {
                Ok(c) => c,
                Err(e) => { eprintln!("[vk] Context failed: {e}"); return None; }
            };

            let indirect = match render::indirect::IndirectRenderer::new(&mut ctx, &atlas.pixels) {
                Ok(r) => r,
                Err(e) => { eprintln!("[vk] Renderer failed: {e}"); return None; }
            };

            let direct = match render::direct::DirectRenderer::new(&mut ctx, &atlas.pixels) {
                Ok(d) => { eprintln!("[vk] DIRECT renderer ready"); Some(d) }
                Err(e) => { eprintln!("[vk] DIRECT renderer failed: {e}, will use INDIRECT"); None }
            };

            eprintln!("[vk] Initialised on: {}", ctx.gpu_name);
            //Some(Self { ctx, renderer })
            Some(Self { ctx, indirect, direct })
        }
    }

    fn render(&mut self, mode: RenderMode, buffer: &TextBuffer,
              debug_cells: Option<&[[u32; 3]]>) -> Result<(), render::RenderError> {
        match mode {
            RenderMode::VulkanDirect => {
                if let Some(d) = self.direct.as_mut() {
                    d.render_frame(&mut self.ctx, buffer, debug_cells)
                } else {
                    self.indirect.render_frame(&mut self.ctx, buffer, debug_cells)
                }
            }
            _ => self.indirect.render_frame(&mut self.ctx, buffer, debug_cells),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Application
// ═══════════════════════════════════════════════════════════════════════

struct App {
    config: OcConfig,
    emu: EmulatorState,
    lua: Option<lua::state::LuaState>,
    atlas: GlyphAtlas,
    // Window
    window: Option<Arc<Window>>,
    // Softbuffer (always available)
    sb_ctx: Option<softbuffer::Context<Arc<Window>>>,
    sb_srf: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    // Vulkan (optional)
    vulkan: Option<VulkanState>,
    // Render mode
    render_mode: RenderMode,
    // State
    last_tick: Instant,
    running: bool,
    sleep_until: Option<Instant>,
    init_done: bool,
    // FPS counter
    frame_count: u32,
    fps_timer: Instant,
    fps: u32,
    overlay: overlay::OverlayState,
    debug_metrics: overlay::DebugMetrics,
    frame_time_us: u64,
    saved_render_mode: Option<RenderMode>,
    lua_steps_last_tick: u32,
    fps_limit: Option<u32>,
    vsync_enabled: bool,
    ctrl_held: bool,
    alt_held: bool,
    sound: Option<Arc<sound::SoundSystem>>,
    profiler_needs_sw: bool,
    was_vulkan_last_frame: bool,
}

impl App {
    fn new() -> Self {
        let mut cfg = OcConfig::default();
        let saved = settings_file::load_settings();
        saved.apply_to_config(&mut cfg);
        let mut emu = EmulatorState::new(&cfg);

        emu.eeprom.flash(include_str!("../assets/bios.lua"));

        // -- OpenOS filesystem --
        let openos_path = std::path::Path::new("assets/openos");
        if openos_path.is_dir() {
            let mut files = HashMap::new();
            load_directory_recursive(openos_path, "", &mut files);
            eprintln!("[oc-emu] Loaded OpenOS: {} files", files.len());
            let vfs = crate::fs::VirtualFs::from_files(files);
            let mut fs_comp = crate::components::filesystem::FilesystemComponent::new(vfs);
            fs_comp.label = Some("openos".into());
            emu.boot_address = fs_comp.address.clone();
            emu.filesystems.push(fs_comp);
        }

        // -- Unmanaged HDD (tier 2, 2 MiB) --
        let mut hdd = crate::components::drive::Drive::new_hdd(1);
        hdd.set_save_path(format!("save/drive_{}.bin.gz", hdd.address));
        let _ = hdd.load_from_disk();
        emu.drives.push(hdd);

        // -- Tmpfs --
        if cfg.tmp_size_kib > 0 {
            let tmp_vfs = crate::fs::VirtualFs::new(
                (cfg.tmp_size_kib * 1024) as u64, false,
            );
            let mut tmp_comp = crate::components::filesystem::FilesystemComponent::new(tmp_vfs);
            tmp_comp.label = Some("tmpfs".into());
            emu.filesystems.push(tmp_comp);
        }

        // -- Tmpfs --
        if cfg.tmp_size_kib > 0 {
            let tmp_vfs = crate::fs::VirtualFs::new(
                (cfg.tmp_size_kib * 1024) as u64, false,
            );
            let mut tmp_comp = crate::components::filesystem::FilesystemComponent::new(tmp_vfs);
            tmp_comp.label = Some("tmpfs".into());
            emu.filesystems.push(tmp_comp);
        }

        // -- Writable HDD filesystem (2 MiB managed disk for OpenOS installation) --
        {
        let hdd_vfs = crate::fs::VirtualFs::new(2 * 1024 * 1024, false);
            let mut hdd_fs = crate::components::filesystem::FilesystemComponent::new(hdd_vfs);
            hdd_fs.label = Some("hdd".into());
            hdd_fs.save_path = Some("save/hdd".into());
            match hdd_fs.load_from_disk() {
                Ok(()) if std::path::Path::new("save/hdd").is_dir() =>
                    eprintln!("[oc-emu] Loaded HDD from save/hdd/ (addr: {})", hdd_fs.address),
                Err(e) => eprintln!("[oc-emu] HDD load failed: {e}"),
                _ => eprintln!("[oc-emu] Created empty HDD (addr: {})", hdd_fs.address),
            }
            emu.filesystems.push(hdd_fs);
        }

        emu.register_defaults();

        log::info!("Emulator state created: address={}", emu.address);
        log::info!("Components registered: {}", emu.component_types.len());
        for (addr, ty) in &emu.component_types {
            log::debug!("  component {ty:12} -> {addr}");
        }
        log::info!("Boot address: {}", emu.boot_address);
        if !emu.drives.is_empty() {
            log::info!("Drives: {} (first: {} bytes)", emu.drives.len(), emu.drives[0].capacity());
        }
        for fs in &emu.filesystems {
            log::info!("Filesystem: addr={} label={:?} ro={}",
                fs.address, fs.label, fs.is_read_only());
        }

        // -- Sound system --
        let sound_sys = Arc::new(sound::SoundSystem::new(
            cfg.master_volume, cfg.effect_volume, cfg.beep_volume,
        ));
        *lua::host::SOUND_SYSTEM.lock().unwrap() = Some(sound_sys.clone());

        let atlas = software::load_or_create_atlas();
        let now = Instant::now();

        Self {
            config: cfg, emu, lua: None, atlas,
            window: None, sb_ctx: None, sb_srf: None,
            vulkan: None,
            render_mode: saved.render_mode(),
            last_tick: now, running: false, sleep_until: None, init_done: false,
            frame_count: 0, fps_timer: now, fps: 0,
            overlay: overlay::OverlayState::new(),
            debug_metrics: overlay::DebugMetrics::default(),
            frame_time_us: 0,
            saved_render_mode: None,
            lua_steps_last_tick: 0,
            fps_limit: saved.fps_limit,
            vsync_enabled: saved.vsync,
            ctrl_held: false,
            alt_held: false,
            sound: Some(sound_sys), profiler_needs_sw: false, was_vulkan_last_frame: false,
        }
    }

    fn boot(&mut self) {
        log::info!("Booting machine...");
        let lua = match lua::state::LuaState::new() {
            Some(l) => {
                log::info!("Lua 5.4 state created");
                l
            }
            None => {
                log::error!("FATAL: Cannot create Lua state");
                return;
            }
        };
        lua::host::register_apis(&lua, &mut self.emu);
        log::debug!("Host APIs registered in Lua state");

        let src = include_str!("../assets/machine.lua");
        if let Err(e) = lua::host::load_kernel(&lua, src) {
            log::error!("FATAL: Kernel load failed: {e}");
            return;
        }
        log::info!("Kernel loaded ({} bytes), performing initial yield...", src.len());
        match lua::host::step_kernel(&lua, &mut self.emu) {
            ExecResult::Sleep(s) => log::info!("Initial yield OK (sleep {s:.3}s)"),
            ref other => log::warn!("Initial yield unexpected: {other:?}"),
        }

        if let Some(snd) = &self.sound {
            snd.start_running_loop();
            log::debug!("Computer running sound loop started");
        }

        self.lua = Some(lua);
        self.running = true;
        self.init_done = true;
        log::info!("Machine booted successfully");
    }
    /// Stop the current machine and reboot from the BIOS.
    ///
    /// Drops the Lua state, saves writable filesystems, closes all
    /// file handles, resets uptime, and calls `boot()` to start fresh.
    fn reboot(&mut self) {
        eprintln!("[oc-emu] Rebooting...");

        // Save state before rebooting
        self.save_all();

        // Stop sound
        if let Some(snd) = &self.sound {
            snd.stop_running_loop();
        }

        // Drop old Lua state
        self.lua = None;
        self.running = false;
        self.sleep_until = None;

        // Close all file handles across all filesystems
        for fs in &mut self.emu.filesystems {
            fs.close_all();
        }

        // Reset uptime
        self.emu.uptime_ticks = 0;
        self.emu.start_time = std::time::Instant::now();

        // Clear signal queue
        self.emu.signals.clear();

        // Re-register components (addresses haven't changed)
        self.emu.component_types.clear();
        self.emu.register_defaults();

        // Boot fresh
        self.boot();
        self.update_title();
    }

    fn tick(&mut self) {
        if !self.running { return; }
        let _prof = profiler::scope(profiler::Cat::Tick, "tick");
        self.emu.uptime_ticks += 1;
        log::trace!("tick #{} (uptime {:.2}s)", self.emu.uptime_ticks, self.emu.uptime());

        let lua = match self.lua.as_ref() {
            Some(l) => l,
            None => return,
        };

        let budget_end = Instant::now() + TICK_BUDGET;
        let mut steps: u32 = 0;

        loop {
            if let Some(wake) = self.sleep_until {
                if Instant::now() < wake && self.emu.signals.is_empty() { break; }
                self.sleep_until = None;
            }

            match lua::host::step_kernel(lua, &mut self.emu) {
                ExecResult::Sleep(s) if s.is_infinite() => {
                    self.sleep_until = Some(Instant::now() + Duration::from_millis(100));
                    break;
                }
                ExecResult::Sleep(s) => {
                    if s > 0.001 {
                        self.sleep_until = Some(Instant::now() + Duration::from_secs_f64(s.min(60.0)));
                    }
                    if self.emu.signals.is_empty() { break; }
                }
                ExecResult::Shutdown { reboot } => {
                    eprintln!("[oc-emu] Shutdown (reboot={reboot})");
                    self.save_all();
                    if let Some(snd) = &self.sound { snd.stop_running_loop(); }
                    self.running = false;
                    break;
                }
                ExecResult::Halted => {
                    eprintln!("[oc-emu] Halted");
                    self.save_all();
                    if let Some(snd) = &self.sound { snd.stop_running_loop(); }
                    self.running = false;
                    break;
                }
                ExecResult::Error(msg) => {
                    eprintln!("[oc-emu] ERROR: {msg}");
                    self.save_all();
                    if let Some(snd) = &self.sound {
                        snd.beep(1000.0, 0.75);
                        
                        snd.stop_running_loop();
                    }
                    show_bluescreen(&mut self.emu.buffer, &msg);
                    self.running = false;
                    break;
                }
                ExecResult::SynchronizedCall => {}
            }
            
            steps += 1;
            if steps >= MAX_STEPS_PER_TICK || Instant::now() >= budget_end { break; }
            self.lua_steps_last_tick = steps;
            log::trace!("tick #{}: {steps} Lua steps", self.emu.uptime_ticks);
        }
    }

    /// Persist all writable filesystems and drives to host disk.
    fn save_all(&mut self) {
        for fs in &self.emu.filesystems {
            if let Err(e) = fs.save_to_disk() {
                eprintln!("[oc-emu] FS save error: {e}");
            }
        }
        for drive in &mut self.emu.drives {
            if let Err(e) = drive.save_to_disk() {
                eprintln!("[oc-emu] Drive save error: {e}");
            }
        }
    }

    /// Render one frame to the screen.
    ///
    /// This is the top-level render dispatch. It decides which rendering
    /// path to use based on the current render mode, whether overlays
    /// are open (settings GUI or profiler force software compositing),
    /// and whether Vulkan is available.
    ///
    /// ## Path selection
    ///
    /// ```text
    /// profiler visible OR settings visible
    ///   -> software composited (OC screen + overlays blitted together)
    ///
    /// Vulkan INDIRECT or DIRECT
    ///   -> try Vulkan; on SwapchainOutOfDate, recreate and retry;
    ///      on failure, fall back to Software
    ///
    /// Software
    ///   -> softbuffer blit
    /// ```
    ///
    /// After rendering, clears the text buffer dirty flag, records
    /// frame timing, updates the FPS counter, and optionally sleeps
    /// to enforce the FPS limit (when vsync is off).
/// Render one frame to the screen.
    ///
    /// Dispatches to the appropriate backend (software, Vulkan INDIRECT,
    /// or Vulkan DIRECT) based on current mode and overlay state.
    ///
    /// When any overlay is open (settings, profiler), rendering is forced
    /// through the software path so CPU-side overlays can be composited.
    /// A `device_wait_idle` call ensures the Vulkan swapchain releases
    /// its hold on the window surface before softbuffer presents, which
    /// prevents the freeze that occurs with FIFO present mode (VSync).
    fn render(&mut self) {
        let _prof = crate::profiler::scope(crate::profiler::Cat::Render, "render");
        let t0 = Instant::now();
        self.collect_debug_metrics();

        // Live-apply settings while overlay is open
        if self.overlay.settings.visible {
            if self.overlay.settings.take_changed() {
                self.apply_settings_live();
            }
            // Play UI feedback beep if queued
            if let Some((freq, dur)) = self.overlay.settings.take_beep() {
                if let Some(snd) = &self.sound {
                    snd.beep(freq, dur);
                }
            }
        }
        
        // Play GUI feedback beep if one was queued by a widget interaction.
        if let Some((freq, dur)) = self.overlay.settings.take_beep() {
            if let Some(snd) = &self.sound {
                snd.beep(freq, dur);
            }
        }

        let profiler_visible = profiler::is_visible();
        let need_software = profiler_visible || self.overlay.settings.visible;

        if need_software {
            // Flush all in-flight Vulkan work before touching the window
            // surface through softbuffer. Without this, FIFO present mode
            // holds the surface and softbuffer's present silently fails,
            // causing a frozen screen while the emulator continues running.
            if self.was_vulkan_last_frame {
                if let Some(vk) = &self.vulkan {
                    unsafe { vk.ctx.device.device_wait_idle().ok(); }
                }
                self.was_vulkan_last_frame = false;
            }
            self.render_sw_with_overlays(profiler_visible);
        } else {
            let debug_cells = if self.overlay.debug_bar_visible {
                let cols = match &self.vulkan {
                    Some(vk) => (vk.ctx.swapchain_extent.width / 8).min(512),
                    None => {
                        let sz = self.window.as_ref()
                            .map(|w| w.inner_size()).unwrap_or_default();
                        (sz.width / 8).min(512)
                    }
                };
                Some(build_debug_cells(&self.debug_metrics, cols))
            } else {
                None
            };

            let is_vk = matches!(
                self.render_mode,
                RenderMode::VulkanIndirect | RenderMode::VulkanDirect
            );
            if is_vk {
                let cells_ref = debug_cells.as_deref();
                if self.render_vulkan(cells_ref) {
                    self.was_vulkan_last_frame = true;
                } else {
                    self.render_mode = RenderMode::Software;
                    self.was_vulkan_last_frame = false;
                    self.render_sw_plain(&debug_cells);
                }
            } else {
                self.was_vulkan_last_frame = false;
                self.render_sw_plain(&debug_cells);
            }
        }

        self.emu.buffer.clear_dirty();
        self.frame_time_us = t0.elapsed().as_micros() as u64;
        self.count_frame();

        // FPS limiter: sleep to cap frame rate when VSync is off.
        // Only effective on the software path and Vulkan without FIFO.
        if !self.vsync_enabled {
            if let Some(cap) = self.fps_limit {
                let target = Duration::from_nanos(1_000_000_000 / cap as u64);
                let elapsed = t0.elapsed();
                if elapsed < target {
                    std::thread::sleep(target - elapsed);
                }
            }
        }
    }

    fn render_vulkan(&mut self, debug_cells: Option<&[[u32; 3]]>) -> bool {
        let vk = match self.vulkan.as_mut() { Some(v) => v, None => return false };
        match vk.render(self.render_mode, &self.emu.buffer, debug_cells) {
            Ok(()) => true,
            Err(render::RenderError::SwapchainOutOfDate) => {
                if let Some(win) = &self.window {
                    let sz = win.inner_size();
                    if sz.width > 0 && sz.height > 0 {
                        let ok = unsafe { vk.ctx.recreate_swapchain(sz.width, sz.height) };
                        if ok.is_ok() {
                            return vk.render(self.render_mode, &self.emu.buffer, debug_cells).is_ok();
                        }
                    }
                }
                false
            }
            Err(e) => { eprintln!("[vk] {e}"); false }
        }
    }

    fn render_sw_plain(&mut self, debug_cells: &Option<Vec<[u32; 3]>>) {
        let srf = match self.sb_srf.as_mut() { Some(s) => s, None => return };
        let win = self.window.as_ref().unwrap();
        let sz = win.inner_size();
        let (w, h) = (sz.width, sz.height);
        if w == 0 || h == 0 { return; }
        srf.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).ok();
        let mut buf = match srf.buffer_mut() { Ok(b) => b, Err(_) => return };
        software::render_text_buffer(&self.emu.buffer, &self.atlas, &mut *buf, w, h);
        if debug_cells.is_some() {
            overlay::render_debug_bar_sw(&self.debug_metrics, &self.atlas, &mut *buf, w, h);
        }
        buf.present().ok();
    }

    fn render_sw_composed(&mut self, debug_cells: &Option<Vec<[u32; 3]>>) {
        let srf = match self.sb_srf.as_mut() { Some(s) => s, None => return };
        let win = self.window.as_ref().unwrap();
        let sz = win.inner_size();
        let (w, h) = (sz.width, sz.height);
        if w == 0 || h == 0 { return; }
        srf.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).ok();
        let mut buf = match srf.buffer_mut() { Ok(b) => b, Err(_) => return };
        software::render_text_buffer(&self.emu.buffer, &self.atlas, &mut *buf, w, h);
        if debug_cells.is_some() {
            overlay::render_debug_bar_sw(&self.debug_metrics, &self.atlas, &mut *buf, w, h);
        }
        self.overlay.settings.render(&self.atlas, &mut *buf, w, h);
        buf.present().ok();
    }

    fn render_software(&mut self) {
        let srf = match self.sb_srf.as_mut() {
            Some(s) => s,
            None => return,
        };
        let win = self.window.as_ref().unwrap();
        let sz = win.inner_size();
        let (w, h) = (sz.width, sz.height);
        if w == 0 || h == 0 { return; }

        srf.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).ok();
        let mut buf = match srf.buffer_mut() {
            Ok(b) => b,
            Err(_) => return,
        };
        software::render_text_buffer(&self.emu.buffer, &self.atlas, &mut *buf, w, h);
        buf.present().ok();
    }

    
    /// Software-render the OC screen plus all active overlays.
    ///
    /// Used when the profiler or settings GUI is open, because those
    /// overlays are drawn directly into the pixel buffer and cannot
    /// be composited by the Vulkan fragment shader.
    ///
    /// Compositing order (back to front):
    /// 1. OC text buffer (base layer)
    /// 2. Debug bar (F9), if active
    /// 3. Settings GUI (F8), if visible
    /// 4. Profiler timeline (F7), if visible
    fn render_sw_with_overlays(&mut self, profiler_visible: bool) {
        let srf = match self.sb_srf.as_mut() { Some(s) => s, None => return };
        let win = self.window.as_ref().unwrap();
        let sz = win.inner_size();
        let (w, h) = (sz.width, sz.height);
        if w == 0 || h == 0 { return; }
        srf.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).ok();
        let mut buf = match srf.buffer_mut() { Ok(b) => b, Err(_) => return };

        // Layer 1: OC text screen.
        software::render_text_buffer(&self.emu.buffer, &self.atlas, &mut *buf, w, h);

        // Layer 2: debug bar (F9).
        if self.overlay.debug_bar_visible {
            overlay::render_debug_bar_sw(
                &self.debug_metrics, &self.atlas, &mut *buf, w, h);
        }

        // Layer 3: settings GUI (F8).
        if self.overlay.settings.visible {
            self.overlay.settings.render(&self.atlas, &mut *buf, w, h);
        }

        // Layer 4: profiler timeline (F7).
        if profiler_visible {
            profiler::render_overlay(&self.atlas, &mut *buf, w, h);
        }

        buf.present().ok();
    }

    fn render_sw_with_profiler(&mut self) {
        let srf = match self.sb_srf.as_mut() { Some(s) => s, None => return };
        let win = self.window.as_ref().unwrap();
        let sz = win.inner_size();
        let (w, h) = (sz.width, sz.height);
        if w == 0 || h == 0 { return; }
        srf.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).ok();
        let mut buf = match srf.buffer_mut() { Ok(b) => b, Err(_) => return };
        software::render_text_buffer(&self.emu.buffer, &self.atlas, &mut *buf, w, h);

        // Debug bar
        if self.overlay.debug_bar_visible {
            overlay::render_debug_bar_sw(&self.debug_metrics, &self.atlas, &mut *buf, w, h);
        }

        // Settings GUI
        if self.overlay.settings.visible {
            self.overlay.settings.render(&self.atlas, &mut *buf, w, h);
        }

        // Profiler overlay
        if profiler::is_visible() {
            profiler::render_overlay(&self.atlas, &mut *buf, w, h);
        }

        buf.present().ok();
    }

    
    fn collect_debug_metrics(&mut self) {
        self.debug_metrics.fps = self.fps;
        self.debug_metrics.frame_time_us = self.frame_time_us;
        self.debug_metrics.render_mode = self.saved_render_mode.unwrap_or(self.render_mode);
        self.debug_metrics.uptime_s = self.emu.uptime();
        self.debug_metrics.signal_queue_len = self.emu.signals.len();
        self.debug_metrics.buffer_resolution = (self.emu.buffer.width(), self.emu.buffer.height());
        self.debug_metrics.lua_steps = self.lua_steps_last_tick;
        if let Some(vk) = &self.vulkan {
            self.debug_metrics.gpu_name = vk.ctx.gpu_name.clone();
            self.debug_metrics.swapchain_extent =
                (vk.ctx.swapchain_extent.width, vk.ctx.swapchain_extent.height);
            self.debug_metrics.present_mode = format!("{:?}", vk.ctx.present_mode);
        }
    }

    fn count_frame(&mut self) {
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.fps = (self.frame_count as f64 / elapsed.as_secs_f64()).round() as u32;
            self.frame_count = 0;
            self.fps_timer = Instant::now();
            self.update_title();
        }
    }

    fn update_title(&self) {
        if let Some(win) = &self.window {
            let gpu_name = match &self.vulkan {
                Some(vk) => vk.ctx.gpu_name.as_str(),
                None => "N/A",
            };
            let mode = self.render_mode.label();
            let state = if self.running { "Running" } else { "Stopped" };
            win.set_title(&format!(
                "OC Emulator | {mode} | {state} | {fps} FPS | GPU: {gpu_name} | F5: cycle renderer",
                fps = self.fps,
            ));
        }
    }

    fn cycle_render_mode(&mut self) {
        let next = self.render_mode.next();
        let prev = self.render_mode;
        let next = if matches!(next, RenderMode::VulkanIndirect | RenderMode::VulkanDirect)
            && self.vulkan.is_none()
        {
            RenderMode::Software
        } else if next == RenderMode::VulkanDirect
            && self.vulkan.as_ref().map_or(true, |v| v.direct.is_none())
        {
            next.next()
        } else {
            next
        };

        self.render_mode = next;
        log::info!("Render mode: {} -> {}", prev.label(), self.render_mode.label());
        self.update_title();
    }
    
    /// Apply all current overlay widget values to config, emu state,
    /// and renderer. Called every frame while the overlay is open and
    /// a widget was changed. Also saves to disk.
    fn apply_settings_live(&mut self) {
        // Push widget values into config
        if let Some(m) = self.overlay.settings.apply_to_config(&mut self.config) {
            let valid = match m {
                RenderMode::VulkanIndirect => self.vulkan.is_some(),
                RenderMode::VulkanDirect => self.vulkan.as_ref()
                    .map_or(false, |v| v.direct.is_some()),
                RenderMode::Software => true,
            };
            if valid {
                self.render_mode = m;
            }
        }

        // Config -> emu state
        self.emu.timeout = self.config.timeout;
        self.emu.ignore_power = self.config.ignore_power;
        self.emu.allow_bytecode = self.config.allow_bytecode;
        self.emu.allow_gc = self.config.allow_gc;

        // Boot address from DynamicChoice
        if let Some(addr) = self.overlay.settings.get_boot_device() {
            self.emu.boot_address = addr.to_owned();
        }

        // VSync
        let new_vsync = self.overlay.settings.get_vsync();
        if new_vsync != self.vsync_enabled {
            self.vsync_enabled = new_vsync;
            if let Some(vk) = &mut self.vulkan {
                let modes = unsafe {
                    vk.ctx.surface_loader
                        .get_physical_device_surface_present_modes(
                            vk.ctx.physical_device, vk.ctx.surface)
                        .unwrap_or_default()
                };
                vk.ctx.present_mode = if new_vsync {
                    ash::vk::PresentModeKHR::FIFO
                } else if modes.contains(&ash::vk::PresentModeKHR::MAILBOX) {
                    ash::vk::PresentModeKHR::MAILBOX
                } else {
                    ash::vk::PresentModeKHR::FIFO
                };
                if let Some(win) = &self.window {
                    let sz = win.inner_size();
                    unsafe {
                        vk.ctx.recreate_swapchain(
                            sz.width.max(1), sz.height.max(1)).ok();
                    }
                }
                eprintln!("[settings] VSync: {} -> {:?}",
                    new_vsync, vk.ctx.present_mode);
            }
        }

        // FPS limit
        self.fps_limit = self.overlay.settings.get_fps_limit();

        // Sound volumes
        if let Some(snd) = &self.sound {
            snd.beep_volume.store(
                (self.config.beep_volume * 100.0) as u32,
                std::sync::atomic::Ordering::Relaxed);
            snd.effect_volume.store(
                (self.config.effect_volume * 100.0) as u32,
                std::sync::atomic::Ordering::Relaxed);
        }

        // Save to disk
        self.save_settings();

        self.update_title();
        eprintln!("[settings] Applied live (timeout={}, vsync={}, fps={:?}, mode={})",
            self.config.timeout, self.vsync_enabled,
            self.fps_limit, self.render_mode.label());
    }

    /// Persist current config to save/settings.json
    fn save_settings(&self) {
        let dir = std::path::Path::new("save");
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join("settings.json");
        match serde_json::to_string_pretty(&self.config) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, &json) {
                    eprintln!("[settings] Save failed: {e}");
                } else {
                    eprintln!("[settings] Saved to {}", path.display());
                }
            }
            Err(e) => eprintln!("[settings] Serialize failed: {e}"),
        }
    }

    /// Recreate the Vulkan swapchain with the appropriate present mode
    /// for the given VSync setting.
    fn apply_vsync(&mut self, vsync: bool) {
        let Some(vk) = &mut self.vulkan else { return };

        let modes = unsafe {
            vk.ctx.surface_loader
                .get_physical_device_surface_present_modes(
                    vk.ctx.physical_device, vk.ctx.surface)
                .unwrap_or_default()
        };

        vk.ctx.present_mode = if vsync {
            ash::vk::PresentModeKHR::FIFO
        } else if modes.contains(&ash::vk::PresentModeKHR::MAILBOX) {
            ash::vk::PresentModeKHR::MAILBOX
        } else if modes.contains(&ash::vk::PresentModeKHR::IMMEDIATE) {
            ash::vk::PresentModeKHR::IMMEDIATE
        } else {
            ash::vk::PresentModeKHR::FIFO
        };

        if let Some(win) = &self.window {
            let sz = win.inner_size();
            unsafe {
                vk.ctx.recreate_swapchain(sz.width.max(1), sz.height.max(1)).ok();
            }
        }

        log::info!("VSync {} -> present mode {:?}",
            if vsync { "ON" } else { "OFF" }, vk.ctx.present_mode);
    }

    fn close_settings(&mut self, apply: bool) {
        self.overlay.settings.visible = false;
        if apply {
            if let Some(m) = self.overlay.settings.apply_to_config(&mut self.config) {
                self.render_mode = m;
            } else if let Some(m) = self.saved_render_mode.take() {
                self.render_mode = m;
            }
            self.saved_render_mode = None;
            self.emu.timeout = self.config.timeout;
            self.emu.ignore_power = self.config.ignore_power;
            self.emu.allow_bytecode = self.config.allow_bytecode;
            self.emu.allow_gc = self.config.allow_gc;

            let new_vsync = self.overlay.settings.get_vsync();
            if new_vsync != self.vsync_enabled {
                self.vsync_enabled = new_vsync;
                if let Some(vk) = &mut self.vulkan {
                    let modes = unsafe {
                        vk.ctx.surface_loader
                            .get_physical_device_surface_present_modes(
                                vk.ctx.physical_device, vk.ctx.surface)
                            .unwrap_or_default()
                    };
                    vk.ctx.present_mode = if new_vsync {
                        ash::vk::PresentModeKHR::FIFO
                    } else if modes.contains(&ash::vk::PresentModeKHR::MAILBOX) {
                        ash::vk::PresentModeKHR::MAILBOX
                    } else {
                        ash::vk::PresentModeKHR::FIFO
                    };
                    if let Some(win) = &self.window {
                        let sz = win.inner_size();
                        unsafe { vk.ctx.recreate_swapchain(sz.width.max(1), sz.height.max(1)).ok(); }
                    }
                }
            }

            self.fps_limit = self.overlay.settings.get_fps_limit();
        } else {
            if let Some(m) = self.saved_render_mode.take() {
                self.render_mode = m;
            }
        }
        self.update_title();
    }

}

/// Show an OC-faithful Blue Screen of Death.
///
/// Layout (matching OC):
/// ```text
/// (blue background fills entire screen)
///
///   Unrecoverable Error
///
///   <error message, word-wrapped>
///
///   All components were disconnected.
///
///   Press any key to continue.
/// ```
fn show_bluescreen(buf: &mut display::TextBuffer, msg: &str) {
    use display::PackedColor;

    let fg = PackedColor::rgb(0xFFFFFF);
    let bg = PackedColor::rgb(0x0000AA);
    buf.set_foreground(fg);
    buf.set_background(bg);

    let w = buf.width();
    let h = buf.height();

    buf.fill(0, 0, w, h, ' ' as u32);

    let margin = 2u32;
    let text_w = (w as usize).saturating_sub(margin as usize * 2).max(1);

    let mut row = 2u32;

    buf.set(margin, row, "Unrecoverable Error", false);
    row += 2;

    let words: Vec<&str> = msg.split_whitespace().collect();
    let mut line = String::new();
    for word in &words {
        if !line.is_empty() && line.len() + 1 + word.len() > text_w {
            buf.set(margin, row, &line, false);
            row += 1;
            line.clear();
            if row >= h - 4 { break; }
        }
        if !line.is_empty() { line.push(' '); }
        line.push_str(word);
    }
    if !line.is_empty() && row < h - 4 {
        buf.set(margin, row, &line, false);
        row += 1;
    }

    row = row.max(h.saturating_sub(6));
    row += 1;
    buf.set(margin, row, "All components were disconnected.", false);
    row += 2;
    buf.set(margin, row, "Press any key to continue.", false);
}

// ═══════════════════════════════════════════════════════════════════════
// winit ApplicationHandler
// ═══════════════════════════════════════════════════════════════════════

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_none() {
            let monitor = el.primary_monitor().or_else(|| el.available_monitors().next());
            let (init_w, init_h) = match &monitor {
                Some(m) => {
                    let sz = m.size();
                    ((sz.width as f64 * 0.55) as u32, (sz.height as f64 * 0.7) as u32)
                }
                None => (1280, 800),
            };
            let attrs = WindowAttributes::default()
                .with_title("OpenComputers Emulator")
                .with_inner_size(winit::dpi::PhysicalSize::new(init_w.max(800), init_h.max(500)));
            let win = Arc::new(el.create_window(attrs).expect("window"));
            log::info!("Window created: {}x{}", init_w, init_h);
            // Softbuffer (always).
            let ctx = softbuffer::Context::new(win.clone()).expect("sb context");
            let srf = softbuffer::Surface::new(&ctx, win.clone()).expect("sb surface");
            self.sb_ctx = Some(ctx);
            self.sb_srf = Some(srf);
            log::info!("Softbuffer surface created");
            // Vulkan (optional).
            self.vulkan = VulkanState::try_new(&win, &self.atlas);
            if self.vulkan.is_some() {
                eprintln!("[oc-emu] Vulkan available — press F5 to cycle renderers");
            }
            match &self.vulkan {
                Some(vk) => log::info!("Vulkan context ready (GPU: {})", vk.ctx.gpu_name),
                None => log::warn!("Vulkan not available, software-only rendering"),
            }
            self.window = Some(win);
        }
        if !self.init_done {
            self.boot();
            self.update_title();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => {
                self.save_all();
                // Save settings on exit
                let saved = settings_file::SavedSettings::from_current(
                    &self.config, self.render_mode,
                    self.vsync_enabled, self.fps_limit,
                    self.overlay.settings.pinky,
                );
                settings_file::save_settings(&saved);
                el.exit();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.overlay.settings.handle_mouse_move(position.x, position.y);
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left && state.is_pressed() {
                    let mx = self.overlay.settings.mouse_x as f64;
                    let my = self.overlay.settings.mouse_y as f64;
                    self.overlay.settings.handle_click(mx, my);
                }
                // Play UI beep if requested
                if let Some((freq, dur)) = self.overlay.settings.pending_beep.take() {
                    if let Some(snd) = &self.sound {
                        snd.beep(freq, dur);
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if self.overlay.settings.visible {
                    let dy = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y / 18.0,
                    };
                    self.overlay.settings.handle_scroll(dy);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // Track modifier keys
                if let winit::keyboard::PhysicalKey::Code(kc) = event.physical_key {
                    match kc {
                        WinitKey::ControlLeft | WinitKey::ControlRight => {
                            self.ctrl_held = event.state.is_pressed();
                        }
                        WinitKey::AltLeft | WinitKey::AltRight => {
                            self.alt_held = event.state.is_pressed();
                        }
                        _ => {}
                    }
                }

                if !event.state.is_pressed() {
                    // Key up - only forward to OC
                    if self.running && !self.overlay.settings.visible {
                        if let winit::keyboard::PhysicalKey::Code(kc) = event.physical_key {
                            let ch = get_char_code(&event);
                            let code = winit_to_lwjgl(kc);
                            self.emu.signals.push(Signal::new("key_up")
                                .with_string(self.emu.keyboard.address.clone())
                                .with_int(ch as i64)
                                .with_int(code as i64));
                        }
                    }
                    return;
                }

                if let winit::keyboard::PhysicalKey::Code(kc) = event.physical_key {
                    // Global hotkeys
                    if profiler::is_visible() {
                        if profiler::handle_key(kc) {
                            return;
                        }
                    }
                    match kc {
                        WinitKey::F7 if !self.overlay.settings.visible => {
                            profiler::handle_key(kc);
                            return;
                        }
                        WinitKey::F5 if !self.overlay.settings.visible => {
                            self.cycle_render_mode();
                            return;
                        }
                        WinitKey::F8 => {
                            if !self.overlay.settings.visible {
                                self.overlay.settings.sync_from_config(
                                    &self.config, self.render_mode,
                                    self.vsync_enabled, self.fps_limit);

                                // Populate boot device list from registered filesystems
                                let devices: Vec<(String, String)> = self.emu.filesystems.iter()
                                    .map(|fs| {
                                        let name = match &fs.label {
                                            Some(l) => format!("{} ({}..)", l, &fs.address[..8]),
                                            None    => format!("{}...", &fs.address[..12]),
                                        };
                                        (name, fs.address.clone())
                                    })
                                    .collect();
                                self.overlay.settings.set_boot_devices(devices, &self.emu.boot_address);

                                self.overlay.settings.visible = true;
                                self.saved_render_mode = Some(self.render_mode);
                            } else {
                                self.close_settings(true);
                            }
                            return;
                        }
                        WinitKey::F9 => {
                            self.overlay.debug_bar_visible = !self.overlay.debug_bar_visible;
                            return;
                        }
                        WinitKey::F10 => {
                            // Save state
                            match crate::machine::persistence::save_state(&self.emu) {
                                Ok(()) => eprintln!("[oc-emu] State saved"),
                                Err(e) => eprintln!("[oc-emu] Save failed: {e}"),
                            }
                            return;
                        }
                        WinitKey::F11 => {
                            // Load state
                            if let Some(snap) = crate::machine::persistence::load_state(&self.emu.address) {
                                crate::machine::persistence::apply_snapshot(&mut self.emu, &snap);
                                eprintln!("[oc-emu] State restored (reboot required)");
                            }
                            return;
                        }
                        _ => {}
                    }

                    // Settings GUI captures all input
// Settings GUI captures all input when visible.
                    if self.overlay.settings.visible {
                        match self.overlay.settings.handle_key(
                            kc, self.ctrl_held, self.alt_held)
                        {
                            overlay::GuiAction::CloseApply  => self.close_settings(true),
                            overlay::GuiAction::CloseCancel => self.close_settings(false),
                            overlay::GuiAction::None => {}
                        }
                        return;
                    }
                }

                // Forward to OC
                if !self.running { return; }
                let ch = get_char_code(&event);
                let code = match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(k) => winit_to_lwjgl(k),
                    _ => 0,
                };
                self.emu.signals.push(Signal::new("key_down")
                    .with_string(self.emu.keyboard.address.clone())
                    .with_int(ch as i64)
                    .with_int(code as i64));
                self.sleep_until = None;
            }

            WindowEvent::Resized(sz) => {
                log::debug!("Window resized to {}x{}", sz.width, sz.height);
                if sz.width > 0 && sz.height > 0 {
                    if let Some(vk) = &mut self.vulkan {
                        log::debug!("Recreating swapchain for new size");
                        unsafe { vk.ctx.recreate_swapchain(sz.width, sz.height).ok(); }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                if now - self.last_tick >= TICK {
                    self.last_tick = now;
                    self.tick();
                }
                self.render();
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    logging::init();
    eprintln!("Molecular function of dichlorvos");
    let el = EventLoop::new().expect("event loop");
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    el.run_app(&mut app).expect("run");
}