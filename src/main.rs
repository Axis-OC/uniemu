//! # OpenComputers Emulator
//!
//! Entry point: creates a window, initialises Vulkan, boots the Lua kernel,
//! and runs the main event loop.
//!
//! ## Usage
//!
//! ```bash
//! # 1. Place Lua 5.4 source in arch/lua54/src/
//! # 2. Place font.hex in assets/font.hex
//! # 3. Compile shaders: (requires Vulkan SDK)
//! #    glslc shaders/text.vert -o shaders/compiled/text_vert.spv
//! #    glslc shaders/text.frag -o shaders/compiled/text_frag.spv
//! # 4. cargo run
//! ```
#![allow(warnings)]
pub mod config;
pub mod display;
pub mod machine;
pub mod components;
pub mod render;
pub mod lua;
pub mod fs;

use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId, WindowAttributes};

use config::OcConfig;
use display::{TextBuffer, ColorDepth};
use machine::{Machine, EmulationMode};
use components::eeprom::Eeprom;
use components::keyboard::Keyboard;
use components::screen::Screen;
use components::gpu::Gpu;

/// Tick interval: 50 ms (20 ticks/sec), matching Minecraft.
const TICK_INTERVAL: Duration = Duration::from_millis(50);

/// Main application state.
struct App {
    config: OcConfig,
    window: Option<Window>,
    machine: Machine,
    buffer: TextBuffer,
    gpu: Gpu,
    screen: Screen,
    keyboard: Keyboard,
    eeprom: Eeprom,
    lua_state: Option<lua::state::LuaState>,
    lua_ctx: lua::host::HostContext,
    last_tick: Instant,
    initialized: bool,
}

impl App {
    fn new() -> Self {
        let config = OcConfig::default();
        let machine = Machine::new(config.clone());
        let buffer = TextBuffer::new(80, 25, ColorDepth::EightBit);
        let gpu = Gpu::new(2, &config); // T3 GPU
        let screen = Screen::new(2); // T3 screen
        let keyboard = Keyboard::new();
        let mut eeprom = Eeprom::new(&config);

        // Flash a minimal BIOS that prints a greeting.
        eeprom.flash(r#"
            local gpu = component.list("gpu")()
            local screen = component.list("screen")()
            if gpu and screen then
                component.invoke(gpu, "bind", screen)
                component.invoke(gpu, "set", 1, 1, "OpenComputers Emulator booted!")
            end
            while true do
                computer.pullSignal(1)
            end
        "#);

        let lua_ctx = lua::host::HostContext {
            timeout: config.timeout,
            ignore_power: config.ignore_power,
            allow_bytecode: config.allow_bytecode,
            allow_gc: config.allow_gc,
            ..Default::default()
        };

        Self {
            config,
            window: None,
            machine,
            buffer,
            gpu,
            screen,
            keyboard,
            eeprom,
            lua_state: None,
            lua_ctx,
            last_tick: Instant::now(),
            initialized: false,
        }
    }

    /// Initialise the Lua VM and register components.
    fn init_lua(&mut self) {
        // Create Lua state.
        let lua = match lua::state::LuaState::new() {
            Some(l) => l,
            None => {
                eprintln!("[oc-emu] FATAL: failed to create Lua state.");
                eprintln!("[oc-emu] Make sure Lua 5.4 source is in arch/lua54/src/");
                return;
            }
        };

        // Register OC APIs.
        lua::host::register_apis(&lua, &mut self.lua_ctx);

        // Set computer address.
        self.lua_ctx.address = crate::components::new_address();

        // Register components in the machine.
        self.machine.max_components = 16;
        self.machine.add_component(self.gpu.address.clone(), "gpu".into());
        self.machine.add_component(self.screen.address.clone(), "screen".into());
        self.machine.add_component(self.keyboard.address.clone(), "keyboard".into());
        self.machine.add_component(self.eeprom.address.clone(), "eeprom".into());

        // Try to load machine.lua.
        let machine_lua = include_str!("../assets/machine.lua");
        match lua::host::load_kernel(&lua, machine_lua) {
            Ok(()) => {
                eprintln!("[oc-emu] Kernel loaded successfully.");
                self.machine.start();
            }
            Err(e) => {
                eprintln!("[oc-emu] Failed to load kernel: {e}");
            }
        }

        self.lua_state = Some(lua);
        self.initialized = true;
    }

    /// Run one tick of the machine.
    fn tick(&mut self) {
        self.machine.tick();

        if let Some(ref lua) = self.lua_state {
            if self.machine.is_running() {
                if let Some(result) = self.machine.step_lua(lua, &mut self.lua_ctx) {
                    match result {
                        lua::host::ExecResult::Error(msg) => {
                            eprintln!("[oc-emu] Lua error: {msg}");
                            // Display error on screen.
                            self.buffer.set_foreground(display::PackedColor::rgb(0xFF_FF_FF));
                            self.buffer.set_background(display::PackedColor::rgb(0x00_00_FF));
                            self.buffer.fill(0, 0, 80, 25, ' ' as u32);
                            self.buffer.set(1, 1, "Unrecoverable Error", false);
                            self.buffer.set(1, 3, &msg, false);
                        }
                        lua::host::ExecResult::Halted => {
                            eprintln!("[oc-emu] Computer halted.");
                        }
                        lua::host::ExecResult::Shutdown { reboot } => {
                            eprintln!("[oc-emu] Shutdown (reboot={})", reboot);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Handle keyboard input from the window.
    fn handle_key(&mut self, event: &winit::event::KeyEvent) {
        if !event.state.is_pressed() { return; }

        // Convert to OC key codes (simplified).
        let char_code = event.text.as_ref()
            .and_then(|t| t.chars().next())
            .map(|c| c as u16)
            .unwrap_or(0);

        // Use physical key as key code.
        let key_code = match event.physical_key {
            winit::keyboard::PhysicalKey::Code(code) => code as u32,
            _ => 0,
        };

        if event.state.is_pressed() {
            self.keyboard.key_down(&mut self.machine, char_code, key_code);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("OpenComputers Emulator")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    80 * 8,  // 640
                    25 * 16, // 400
                ));

            match event_loop.create_window(attrs) {
                Ok(w) => {
                    eprintln!("[oc-emu] Window created: {}×{}",
                        w.inner_size().width, w.inner_size().height);
                    self.window = Some(w);
                }
                Err(e) => {
                    eprintln!("[oc-emu] Failed to create window: {e}");
                    event_loop.exit();
                    return;
                }
            }
        }

        if !self.initialized {
            self.init_lua();
            // Perform initial yield (machine.lua expects this).
            if let Some(ref lua) = self.lua_state {
                let _ = self.machine.step_lua(lua, &mut self.lua_ctx);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(&event);
            }
            WindowEvent::RedrawRequested => {
                // Tick if enough time has passed.
                let now = Instant::now();
                if now - self.last_tick >= TICK_INTERVAL {
                    self.last_tick = now;
                    self.tick();
                }

                // TODO: Render frame via Vulkan.
                // For now, request another redraw.
                if let Some(ref w) = self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref w) = self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    eprintln!("╔══════════════════════════════════════╗");
    eprintln!("║  OpenComputers Emulator v0.1.0       ║");
    eprintln!("║  Rust + Vulkan (ash)                 ║");
    eprintln!("╚══════════════════════════════════════╝");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}