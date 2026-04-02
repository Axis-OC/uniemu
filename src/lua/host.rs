//! # OpenComputers Machine <-> Lua bridge
//!
//! This is the largest and most complex module in the emulator. It
//! contains [`EmulatorState`] (the single source of truth for all
//! host-side data accessible from Lua C callbacks) and every callback
//! function that the OC kernel expects.
//!
//! ## Callback registration
//!
//! During startup, [`register_apis`] creates Lua global tables and
//! populates them with C function closures:
//!
//! ```text
//! _G.computer = {
//!     realTime    = c_real_time,
//!     uptime      = c_uptime,
//!     address     = c_address,
//!     freeMemory  = c_free_memory,
//!     totalMemory = c_total_memory,
//!     pushSignal  = c_push_signal,
//!     ...
//! }
//!
//! _G.component = {
//!     list    = comp_list,
//!     type    = comp_type,
//!     invoke  = comp_invoke,
//!     methods = comp_methods,
//!     doc     = comp_doc,
//!     slot    = comp_slot,
//! }
//!
//! _G.system = {
//!     timeout       = sys_timeout,
//!     allowBytecode = sys_allow_bytecode,
//!     allowGC       = sys_allow_gc,
//! }
//!
//! _G.unicode = { ... }  -- built from Lua shim using utf8 stdlib
//!
//! _G.userdata = {}      -- stub (empty table)
//! ```
//!
//! ## EmulatorState access from C callbacks
//!
//! A raw `*mut EmulatorState` pointer is stashed in the Lua registry
//! under the key `"__oc_emu"`. Each C callback retrieves it via the
//! `st()` helper function:
//!
//! ```text
//! unsafe fn st(L: *mut lua_State) -> &'static mut EmulatorState {
//!     lua_getfield(L, LUA_REGISTRYINDEX, "__oc_emu");
//!     let p = lua_touserdata(L, -1) as *mut EmulatorState;
//!     lua_pop(L, 1);
//!     &mut *p
//! }
//! ```
//!
//! This is safe in practice because:
//! * Lua is driven single-threadedly from the main loop.
//! * The `EmulatorState` outlives the Lua state.
//! * No mutable aliasing occurs (callbacks run sequentially).
//!
//! ## component.invoke dispatch
//!
//! The `comp_invoke` callback is the central routing point for all
//! component method calls from Lua:
//!
//! ```text
//! Lua: component.invoke(addr, "set", 1, 1, "Hello")
//!   |
//!   v
//! comp_invoke(L) -- reads addr and method from stack
//!   |
//!   | looks up type_name in component_types
//!   v
//! dispatch_invoke(L, state, addr, type_name, method)
//!   |
//!   | match type_name {
//!   |   "eeprom"    -> match method { "get" -> ..., "set" -> ... }
//!   |   "gpu"       -> match method { "set" -> ..., "fill" -> ... }
//!   |   "screen"    -> ...
//!   |   "filesystem"-> dispatch_filesystem(L, fs, method)
//!   |   ...
//!   | }
//!   v
//! Push (true, result...) or (false, error_msg) onto Lua stack
//! ```
//!
//! ## Return value protocol
//!
//! All `component.invoke` results follow the OC protocol:
//!
//! * Success: push `true` followed by the method's return values.
//!   Return count = 1 + number_of_results.
//! * Error: push `false` followed by an error message string.
//!   Return count = 2.
//!
//! ## Kernel lifecycle
//!
//! [`load_kernel`] compiles `machine.lua`, wraps it in a coroutine,
//! and stores the coroutine thread at stack index 1 of the main state.
//!
//! [`step_kernel`] resumes the coroutine with any pending signal,
//! interprets the yield/return, and produces an [`ExecResult`].
//!
//! The host (in `main.rs`) calls `step_kernel` in a loop within each
//! tick, respecting the tick budget and sleep timers.

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_int;
use std::time::Instant;
use std::sync::Mutex;

use super::ffi;
use super::state::{LuaState, LuaResult};
use crate::config::OcConfig;
use crate::display::{TextBuffer, ColorDepth, PackedColor};
use crate::machine::signal::{Signal, SignalArg, SignalQueue};

// -----------------------------------------------------------------------
// Registry key for the EmulatorState pointer
// -----------------------------------------------------------------------

/// Key used to stash the `EmulatorState*` in the Lua registry.
///
/// NUL-terminated byte string for direct use as a C string pointer.
const STATE_KEY: &[u8] = b"__oc_emu\0";
pub static SOUND_SYSTEM: Mutex<Option<std::sync::Arc<crate::sound::SoundSystem>>> =
    Mutex::new(None);
// -----------------------------------------------------------------------
// Execution result enum
// -----------------------------------------------------------------------

/// Outcome of one kernel resume/yield cycle.
///
/// Returned by [`step_kernel`] to tell the host what to do next.
#[derive(Debug)]
pub enum ExecResult {
    /// Sleep for the given number of seconds before resuming.
    ///
    /// `f64::INFINITY` means "wait indefinitely for a signal" (equivalent
    /// to OC's `computer.pullSignal()` with no timeout).
    Sleep(f64),

    /// A synchronised call was requested by the kernel.
    ///
    /// The kernel yielded a function that must be executed on the main
    /// thread (in the main Lua state, not the coroutine). The result
    /// is pushed back and the coroutine is resumed immediately.
    ///
    /// This is handled recursively inside [`step_kernel`] (up to a
    /// nesting limit of 64).
    SynchronizedCall,

    /// The machine requested shutdown.
    ///
    /// * `reboot: true` -> the host should restart the machine.
    /// * `reboot: false` -> the host should stop the machine permanently.
    Shutdown { reboot: bool },

    /// The kernel coroutine returned (finished executing).
    ///
    /// This means `machine.lua` exited cleanly, which typically
    /// indicates a successful shutdown.
    Halted,

    /// An unrecoverable error occurred.
    ///
    /// The string contains the error message (which may include a
    /// stack traceback if one was available).
    Error(String),
}

// -----------------------------------------------------------------------
// EmulatorState
// -----------------------------------------------------------------------

/// All host-side state accessible from Lua C callbacks.
///
/// A raw `*mut EmulatorState` is stashed in the Lua registry so that
/// C callbacks can access it. This is safe because Lua is driven
/// single-threadedly from the main loop, and the `EmulatorState`
/// outlives the Lua state.
///
/// This struct is the "god object" of the emulator -- it holds
/// references to every component, the display buffer, the signal
/// queue, and all configuration scalars that Lua callbacks need.
///
/// # Why not pass individual references?
///
/// Lua C callbacks have a fixed signature (`fn(L) -> c_int`) and
/// cannot take extra parameters. The only way to access host data
/// is through a pointer stashed in the registry. Using a single
/// struct simplifies the registry access pattern (one getfield per
/// callback invocation instead of many).
pub struct EmulatorState {
    // -- Timing --

    /// Wall-clock time when the emulator started.
    ///
    /// Used by `computer.realTime()` to return elapsed seconds.
    pub start_time: Instant,

    /// Number of ticks since the machine started.
    ///
    /// Incremented once per tick (50 ms). Used by `computer.uptime()`
    /// which returns `uptime_ticks / 20.0` seconds.
    pub uptime_ticks: u64,

    // -- Identity --

    /// UUID address of the computer itself.
    ///
    /// Returned by `computer.address()`. In OC, this is the address
    /// of the computer case block.
    pub address: String,

    /// UUID address of the boot filesystem.
    ///
    /// Set/get via `computer.setBootAddress()` / `computer.getBootAddress()`.
    /// Used by the BIOS to know which filesystem to boot from.
    pub boot_address: String,

    // -- Config scalars --

    /// Maximum execution time before forced yield.
    pub timeout: f64,

    /// Whether to ignore power requirements.
    pub ignore_power: bool,

    /// Whether Lua bytecode loading is allowed.
    pub allow_bytecode: bool,

    /// Whether `__gc` metamethods are allowed.
    pub allow_gc: bool,

    // -- Component registry --

    /// Map from component address to type name.
    ///
    /// E.g. `("abc-def-...", "gpu")`, `("123-456-...", "filesystem")`.
    /// Populated during setup via `register_defaults()`.
    pub component_types: HashMap<String, String>,

    // -- Concrete components --

    /// The EEPROM (BIOS) component.
    pub eeprom: crate::components::eeprom::Eeprom,

    /// The GPU component.
    pub gpu: crate::components::gpu::Gpu,

    /// The screen component.
    pub screen: crate::components::screen::Screen,

    /// The keyboard component.
    pub keyboard: crate::components::keyboard::Keyboard,

    /// All filesystem components (ROM, tmpfs, disks).
    pub filesystems: Vec<crate::components::filesystem::FilesystemComponent>,

    // -- Display --

    /// The text buffer (shared between GPU and renderers).
    pub buffer: TextBuffer,

    /// Address of the screen the GPU is currently bound to.
    pub gpu_bound_screen: Option<String>,

    // -- Signals --

    /// The signal queue (bounded FIFO).
    pub signals: SignalQueue,

    /// Unmanaged drives (raw block devices).
    pub drives: Vec<crate::components::drive::Drive>,
    pub ram_bytes: i64,
}

impl EmulatorState {
    /// Create a new `EmulatorState` with defaults from the config.
    ///
    /// Components are created but not yet registered in `component_types`.
    /// Call `register_defaults()` after adding any filesystems.
    pub fn new(config: &OcConfig) -> Self {
        let ram_bytes = match config.ram_tier {
            0 => 192 * 1024,
            1 => 384 * 1024,
            2 => 768 * 1024,
            _ => 64 * 1024 * 1024,
        };
        Self {
            start_time: Instant::now(),
            uptime_ticks: 0,
            address: crate::components::new_address(),
            boot_address: String::new(),
            timeout: config.timeout,
            ignore_power: config.ignore_power,
            allow_bytecode: config.allow_bytecode,
            allow_gc: config.allow_gc,
            component_types: HashMap::new(),
            eeprom: crate::components::eeprom::Eeprom::new(config),
            gpu: crate::components::gpu::Gpu::new(2, config),
            screen: crate::components::screen::Screen::new(2),
            keyboard: crate::components::keyboard::Keyboard::new(),
            filesystems: Vec::new(),
            buffer: TextBuffer::new(80, 25, ColorDepth::EightBit),
            gpu_bound_screen: None,
            signals: SignalQueue::new(config.max_signal_queue_size), drives: Vec::new(), ram_bytes,
        }
    }

    /// Register a component address -> type name mapping.
    ///
    /// After this, `component.list()` and `component.type()` will
    /// see the component.
    pub fn register_component(&mut self, addr: &str, type_name: &str) {
        self.component_types.insert(addr.to_owned(), type_name.to_owned());
    }

    /// Register all built-in components.
    ///
    /// Must be called after adding any filesystems to `self.filesystems`.
    pub fn register_defaults(&mut self) {
        let pairs: Vec<(String, String)> = vec![
            (self.address.clone(), "computer".into()),
            (self.eeprom.address.clone(), "eeprom".into()),
            (self.gpu.address.clone(), "gpu".into()),
            (self.screen.address.clone(), "screen".into()),
            (self.keyboard.address.clone(), "keyboard".into()),
        ];
        for (a, t) in pairs {
            self.component_types.insert(a, t);
        }
        for fs in &self.filesystems {
            self.component_types.insert(fs.address.clone(), "filesystem".into());
        }
        for d in &self.drives {
            self.component_types.insert(d.address.clone(), "drive".into());
        }
    }

    /// Uptime in seconds (`uptime_ticks / 20.0`).
    #[inline]
    pub fn uptime(&self) -> f64 {
        self.uptime_ticks as f64 / 20.0
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Retrieve state from Lua registry
// ═══════════════════════════════════════════════════════════════════════

#[inline]
unsafe fn st(L: *mut ffi::lua_State) -> &'static mut EmulatorState {
    ffi::lua_getfield(L, ffi::LUA_REGISTRYINDEX, STATE_KEY.as_ptr() as *const _);
    let p = ffi::lua_touserdata(L, -1) as *mut EmulatorState;
    ffi::lua_pop(L, 1);
    &mut *p
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers: push / read Lua values
// ═══════════════════════════════════════════════════════════════════════

pub fn with_sound(f: impl FnOnce(&crate::sound::SoundSystem)) {
    if let Ok(guard) = SOUND_SYSTEM.lock() {
        if let Some(snd) = guard.as_ref() {
            f(snd);
        }
    }
}

unsafe fn push_str(L: *mut ffi::lua_State, s: &str) {
    ffi::lua_pushlstring(L, s.as_ptr() as *const _, s.len());
}

unsafe fn push_bytes(L: *mut ffi::lua_State, b: &[u8]) {
    ffi::lua_pushlstring(L, b.as_ptr() as *const _, b.len());
}

unsafe fn to_str<'a>(L: *mut ffi::lua_State, idx: c_int) -> Option<&'a str> {
    let mut len: usize = 0;
    let p = ffi::lua_tolstring(L, idx, &mut len);
    if p.is_null() { return None; }
    let bytes = std::slice::from_raw_parts(p as *const u8, len);
    std::str::from_utf8(bytes).ok()
}

/// Read bytes from the Lua stack at `idx` (string or nil → empty).
unsafe fn read_bytes(L: *mut ffi::lua_State, idx: c_int) -> Vec<u8> {
    let mut len: usize = 0;
    let p = ffi::lua_tolstring(L, idx, &mut len);
    if p.is_null() { return Vec::new(); }
    std::slice::from_raw_parts(p as *const u8, len).to_vec()
}

/// Push `false, msg` for invoke errors.
#[inline]
unsafe fn inv_err(L: *mut ffi::lua_State, msg: &str) -> c_int {
    ffi::lua_pushboolean(L, 0);
    push_str(L, msg);
    2
}

// ═══════════════════════════════════════════════════════════════════════
// computer.* callbacks
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" fn c_real_time(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    ffi::lua_pushnumber(L, s.start_time.elapsed().as_secs_f64());
    1
}

unsafe extern "C" fn c_uptime(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    ffi::lua_pushnumber(L, s.uptime());
    1
}

unsafe extern "C" fn c_address(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    push_str(L, &s.address);
    1
}

unsafe extern "C" fn c_free_memory(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    ffi::lua_pushinteger(L, s.ram_bytes);
    1
}

unsafe extern "C" fn c_total_memory(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    ffi::lua_pushinteger(L, s.ram_bytes);
    1
}

unsafe extern "C" fn c_push_signal(L: *mut ffi::lua_State) -> c_int {
    crate::profiler::instant(crate::profiler::Cat::Signal);
    let s = st(L);
    let name = to_str(L, 1).unwrap_or("");
    let mut sig = Signal::new(name);
    let top = ffi::lua_gettop(L);
    for i in 2..=top {
        match ffi::lua_type(L, i) {
            ffi::LUA_TBOOLEAN => sig.args.push(SignalArg::Bool(ffi::lua_toboolean(L, i) != 0)),
            ffi::LUA_TNUMBER => {
                if ffi::lua_isinteger(L, i) != 0 {
                    sig.args.push(SignalArg::Int(ffi::lua_tointeger(L, i)));
                } else {
                    sig.args.push(SignalArg::Float(ffi::lua_tonumber(L, i)));
                }
            }
            ffi::LUA_TSTRING => {
                if let Some(v) = to_str(L, i) {
                    sig.args.push(SignalArg::Str(v.to_owned()));
                }
            }
            _ => sig.args.push(SignalArg::Nil),
        }
    }
    ffi::lua_pushboolean(L, s.signals.push(sig) as c_int);
    1
}

unsafe extern "C" fn c_tmp_address(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnil(L); 1
}
unsafe extern "C" fn c_users(L: *mut ffi::lua_State) -> c_int { 0 }

unsafe extern "C" fn c_energy(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    ffi::lua_pushnumber(L, if s.ignore_power { f64::INFINITY } else { 10000.0 });
    1
}

unsafe extern "C" fn c_max_energy(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnumber(L, 10000.0); 1
}

unsafe extern "C" fn c_get_architectures(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_newtable(L);
    push_str(L, "Lua 5.4");
    ffi::lua_rawseti(L, -2, 1);
    1
}

unsafe extern "C" fn c_get_architecture(L: *mut ffi::lua_State) -> c_int {
    push_str(L, "Lua 5.4"); 1
}

unsafe extern "C" fn c_set_architecture(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushboolean(L, 0); 1
}

unsafe extern "C" fn c_is_robot(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushboolean(L, 0); 1
}

unsafe extern "C" fn c_get_boot_address(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    if s.boot_address.is_empty() { ffi::lua_pushnil(L); }
    else { push_str(L, &s.boot_address); }
    1
}

unsafe extern "C" fn c_set_boot_address(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    if let Some(a) = to_str(L, 1) { s.boot_address = a.to_owned(); }
    0
}

unsafe extern "C" fn c_add_user(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushboolean(L, 1); 1
}
unsafe extern "C" fn c_remove_user(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushboolean(L, 0); 1
}

// ═══════════════════════════════════════════════════════════════════════
// component.* callbacks
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" fn comp_list(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    let filter = to_str(L, 1);
    let exact = if ffi::lua_isboolean(L, 2) { ffi::lua_toboolean(L, 2) != 0 } else { true };

    ffi::lua_createtable(L, 0, s.component_types.len() as c_int);
    for (addr, name) in &s.component_types {
        let pass = match filter {
            Some(f) => if exact { name == f } else { name.contains(f) },
            None => true,
        };
        if pass {
            push_str(L, addr);
            push_str(L, name);
            ffi::lua_rawset(L, -3);
        }
    }
    1
}

unsafe extern "C" fn comp_type(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    let addr = to_str(L, 1).unwrap_or("");
    match s.component_types.get(addr) {
        Some(name) => { push_str(L, name); 1 }
        None => { ffi::lua_pushnil(L); push_str(L, "no such component"); 2 }
    }
}

unsafe extern "C" fn comp_slot(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    let addr = to_str(L, 1).unwrap_or("");
    if s.component_types.contains_key(addr) {
        ffi::lua_pushinteger(L, -1); 1
    } else {
        ffi::lua_pushnil(L); push_str(L, "no such component"); 2
    }
}

/// Build a methods-info table for the given component type.
unsafe fn push_methods_table(L: *mut ffi::lua_State, type_name: &str) {
    ffi::lua_newtable(L);
    let methods: &[&str] = match type_name {
        "eeprom" => &["get","set","getLabel","setLabel","getData","setData",
                       "getSize","getDataSize","getChecksum","makeReadonly"],
        "gpu" => &["bind","getScreen","setResolution","getResolution","maxResolution",
                    "setBackground","getBackground","setForeground","getForeground",
                    "set","fill","copy","get","setDepth","getDepth","maxDepth",
                    "setViewport","getViewport","setPaletteColor","getPaletteColor",
                    "freeMemory","totalMemory","getActiveBuffer","setActiveBuffer",
                    "buffers","allocateBuffer","freeBuffer","freeAllBuffers","getBufferSize"],
        "screen" => &["isOn","turnOn","turnOff","getAspectRatio","getKeyboards",
                       "setPrecise","isPrecise","setTouchModeInverted","isTouchModeInverted"],
        "keyboard" => &[],
        "filesystem" => &["isReadOnly","spaceTotal","spaceUsed","exists","isDirectory",
                          "size","list","makeDirectory","remove","rename","open",
                          "read","write","close","seek","getLabel","setLabel",
                          "lastModified"],
        "computer" => &["beep","getDeviceInfo","getProgramLocations"],
        _ => &[],
    };
    for &m in methods {
        push_str(L, m);
        ffi::lua_newtable(L);
        ffi::lua_pushboolean(L, 1);
        ffi::lua_setfield(L, -2, b"direct\0".as_ptr() as *const _);
        ffi::lua_pushboolean(L, 0);
        ffi::lua_setfield(L, -2, b"getter\0".as_ptr() as *const _);
        ffi::lua_pushboolean(L, 0);
        ffi::lua_setfield(L, -2, b"setter\0".as_ptr() as *const _);
        ffi::lua_rawset(L, -3);
    }
}

unsafe extern "C" fn comp_methods(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    let addr = to_str(L, 1).unwrap_or("");
    match s.component_types.get(addr) {
        Some(t) => { push_methods_table(L, t); 1 }
        None => { ffi::lua_pushnil(L); push_str(L, "no such component"); 2 }
    }
}

unsafe extern "C" fn comp_doc(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnil(L); 1
}

unsafe extern "C" fn comp_invoke(L: *mut ffi::lua_State) -> c_int {
    let s = st(L);
    let addr = match to_str(L, 1) { Some(a) => a.to_owned(), None => {
        ffi::lua_pushboolean(L, 0); push_str(L, "bad address"); return 2;
    }};
    let method = match to_str(L, 2) { Some(m) => m.to_owned(), None => {
        ffi::lua_pushboolean(L, 0); push_str(L, "bad method"); return 2;
    }};
    let type_name = match s.component_types.get(&addr) {
        Some(t) => t.clone(),
        None => { ffi::lua_pushboolean(L, 0); push_str(L, "no such component"); return 2; }
    };

    let cat = match type_name.as_str() {
        "gpu"        => crate::profiler::Cat::GpuCall,
        "filesystem" => crate::profiler::Cat::FsOp,
        "drive"      => crate::profiler::Cat::DriveOp,
        _            => crate::profiler::Cat::LuaStep,
    };
    let _prof = crate::profiler::scope(cat, "invoke");

    dispatch_invoke(L, s, &addr, &type_name, &method)
}

/// Push `true, nil, msg` for "soft" filesystem errors (file not found, I/O error).
#[inline]
unsafe fn fs_soft_err(L: *mut ffi::lua_State, msg: &str) -> c_int {
    ffi::lua_pushboolean(L, 1);
    ffi::lua_pushnil(L);
    push_str(L, msg);
    3
}

/// Dispatch `component.invoke` calls to a filesystem component.
///
/// Method arguments start at Lua stack index 3.
unsafe fn dispatch_filesystem(
    L: *mut ffi::lua_State,
    fs: &mut crate::components::filesystem::FilesystemComponent,
    method: &str,
) -> c_int {
    match method {
        "isReadOnly" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.is_read_only() as c_int);
            2
        }
        "spaceTotal" => {
            ffi::lua_pushboolean(L, 1);
            let t = fs.space_total();
            if t == 0 { ffi::lua_pushnumber(L, f64::INFINITY); }
            else      { ffi::lua_pushinteger(L, t as i64); }
            2
        }
        "spaceUsed" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, fs.space_used() as i64);
            2
        }
        "exists" => {
            let path = to_str(L, 3).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.exists(path) as c_int);
            2
        }
        "isDirectory" => {
            let path = to_str(L, 3).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.is_directory(path) as c_int);
            2
        }
        "size" => {
            let path = to_str(L, 3).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, fs.size(path) as i64);
            2
        }
        "lastModified" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, 0);
            2
        }
        "list" => {
            let path = to_str(L, 3).unwrap_or("");
            match fs.list(path) {
                Some(entries) => {
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_createtable(L, entries.len() as c_int, 0);
                    for (i, entry) in entries.iter().enumerate() {
                        push_str(L, entry);
                        ffi::lua_rawseti(L, -2, (i + 1) as i64);
                    }
                    2
                }
                None => {
                    // Not a directory or doesn't exist → return nil.
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushnil(L);
                    2
                }
            }
        }
        "makeDirectory" => {
            let path = to_str(L, 3).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.make_directory(path) as c_int);
            2
        }
        "remove" => {
            let path = to_str(L, 3).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.remove(path) as c_int);
            2
        }
        "rename" => {
            let from = to_str(L, 3).unwrap_or("");
            let to   = to_str(L, 4).unwrap_or("");
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushboolean(L, fs.rename(from, to) as c_int);
            2
        }
        "open" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let _prof = crate::profiler::scope(crate::profiler::Cat::FsOp, "fs.open");
            let handle = ffi::lua_tointeger(L, 3) as i32;
            let path = to_str(L, 3).unwrap_or("");
            let mode_str = to_str(L, 4).unwrap_or("r");
            log::debug!("fs.open({path:?}, {mode_str:?})");
            let mode = match mode_str {
                "r" | "rb"  => crate::fs::OpenMode::Read,
                "w" | "wb"  => crate::fs::OpenMode::Write,
                "a" | "ab"  => crate::fs::OpenMode::Append,
                _ => return inv_err(L, "unsupported mode"),
            };
            match fs.open(path, mode) {
                Ok(handle) => {
                    log::trace!("fs.open -> handle {handle}");
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushinteger(L, handle as i64);
                    2
                }
                Err(e) => { log::debug!("fs.open({path:?}) failed: {e}"); fs_soft_err(L, e) },
            }
        }
        "read" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let _prof = crate::profiler::scope(crate::profiler::Cat::FsOp, "fs.read");
            let handle = ffi::lua_tointeger(L, 3) as i32;
            
            let raw = ffi::lua_tonumber(L, 4);
            let count = if raw.is_infinite() || raw.is_nan() || raw > 2_000_000.0 {
                2048usize // clamp like OC's maxReadBuffer
            } else if raw < 0.0 {
                0usize
            } else {
                raw as usize
            };
            log::trace!("fs.read(handle={handle}, count={count})");
            match fs.read(handle, count) {
                Ok(Some(data)) => {
                    ffi::lua_pushboolean(L, 1);
                    push_bytes(L, &data);
                    2
                }
                Ok(None) => {
                    // EOF
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushnil(L);
                    2
                }
                Err(e) => fs_soft_err(L, e),
            }
            
        }
        "write" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let _prof = crate::profiler::scope(crate::profiler::Cat::FsOp, "fs.write");
            let handle = ffi::lua_tointeger(L, 3) as i32;
            let data = read_bytes(L, 4);
            log::trace!("fs.write(handle={handle}, {} bytes)", data.len());
            match fs.write(handle, &data) {
                Ok(()) => {
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushboolean(L, 1);
                    2
                }
                Err(e) => fs_soft_err(L, e),
            }
        }
        "seek" => {
            let handle = ffi::lua_tointeger(L, 3) as i32;
            let whence = to_str(L, 4).unwrap_or("cur");
            let offset = ffi::lua_tointeger(L, 5);
            match fs.seek(handle, whence, offset) {
                Ok(pos) => {
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushinteger(L, pos as i64);
                    2
                }
                Err(e) => fs_soft_err(L, e),
            }
        }
        "close" => {
            let handle = ffi::lua_tointeger(L, 3) as i32;
            let _ = fs.close(handle);
            ffi::lua_pushboolean(L, 1);
            1
        }
        "getLabel" => {
            ffi::lua_pushboolean(L, 1);
            match fs.label.as_deref() {
                Some(l) => push_str(L, l),
                None    => ffi::lua_pushnil(L),
            }
            2
        }
        "setLabel" => {
            if fs.is_read_only() {
                return fs_soft_err(L, "filesystem is read-only");
            }
            let label = to_str(L, 3);
            fs.label = label.map(|s| s.chars().take(16).collect());
            ffi::lua_pushboolean(L, 1);
            match fs.label.as_deref() {
                Some(l) => push_str(L, l),
                None    => ffi::lua_pushnil(L),
            }
            2
        }
        
        _ => inv_err(L, "no such method"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Component dispatch  — routes component.invoke calls
// ═══════════════════════════════════════════════════════════════════════

/// Dispatch a `component.invoke(addr, method, ...)` call.
///
/// Protocol: push `true, result…` on success, `false, msg` on error.
/// Returns the number of Lua values pushed (always ≥ 1).
unsafe fn dispatch_invoke(
    L: *mut ffi::lua_State,
    s: &mut EmulatorState,
    addr: &str,
    type_name: &str,
    method: &str,
) -> c_int {
    match type_name {
        // ─── EEPROM ─────────────────────────────────────────────
        "eeprom" => match method {
            "get" => {
                ffi::lua_pushboolean(L, 1);
                push_bytes(L, s.eeprom.get_code());
                2
            }
            "set" => {
                let data = read_bytes(L, 3);
                log::debug!("eeprom.set: {} bytes", data.len());
                match s.eeprom.set_code(data) {
                    Ok(()) => { ffi::lua_pushboolean(L, 1); 1 }
                    Err(e) => {
                        log::warn!("eeprom.set failed: {e}");
                        ffi::lua_pushboolean(L, 0); push_str(L, e); 2
                    }
                }
            }
            "getLabel" => {
                ffi::lua_pushboolean(L, 1);
                push_str(L, s.eeprom.get_label());
                2
            }
            "setLabel" => {
                let label = to_str(L, 3).unwrap_or("EEPROM");
                let _ = s.eeprom.set_label(label);
                ffi::lua_pushboolean(L, 1);
                push_str(L, s.eeprom.get_label());
                2
            }
            "getData" => {
                ffi::lua_pushboolean(L, 1);
                push_bytes(L, s.eeprom.get_data());
                2
            }
            "setData" => {
                let data = read_bytes(L, 3);
                match s.eeprom.set_data(data) {
                    Ok(()) => { ffi::lua_pushboolean(L, 1); 1 }
                    Err(e) => { ffi::lua_pushboolean(L, 0); push_str(L, e); 2 }
                }
            }
            "getSize" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.eeprom.max_code_size() as i64);
                2
            }
            "getDataSize" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.eeprom.max_data_size() as i64);
                2
            }
            "getChecksum" => {
                ffi::lua_pushboolean(L, 1);
                push_str(L, &s.eeprom.checksum());
                2
            }
            "makeReadonly" => {
                let cs = to_str(L, 3).unwrap_or("");
                match s.eeprom.make_readonly(cs) {
                    Ok(()) => { ffi::lua_pushboolean(L, 1); ffi::lua_pushboolean(L, 1); 2 }
                    Err(e) => { ffi::lua_pushboolean(L, 0); push_str(L, e); 2 }
                }
            }
            _ => inv_err(L, "no such method"),
        }

        // ─── GPU ────────────────────────────────────────────────
        "gpu" => match method {
            "bind" => {
                let screen_addr = to_str(L, 3).unwrap_or("").to_owned();
                let reset = if ffi::lua_gettop(L) >= 4 && ffi::lua_isboolean(L, 4) {
                    ffi::lua_toboolean(L, 4) != 0
                } else { true };
                if !s.component_types.contains_key(&screen_addr) {
                    return inv_err(L, "no such component");
                }
                s.gpu_bound_screen = Some(screen_addr);
                if reset {
                    let (mw, mh) = s.gpu.max_resolution();
                    s.buffer.set_resolution(mw, mh);
                    s.buffer.set_color_depth(s.gpu.max_depth());
                    s.buffer.set_foreground(PackedColor::rgb(0xFFFFFF));
                    s.buffer.set_background(PackedColor::rgb(0x000000));
                }
                
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "getScreen" => {
                ffi::lua_pushboolean(L, 1);
                match &s.gpu_bound_screen {
                    Some(a) => push_str(L, a),
                    None => ffi::lua_pushnil(L),
                }
                2
            }
            "setResolution" => {
                let w = ffi::lua_tointeger(L, 3) as u32;
                let h = ffi::lua_tointeger(L, 4) as u32;
                let changed = s.buffer.set_resolution(w, h);
                log::debug!("gpu.setResolution({w}, {h})");
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, changed as c_int);
                2
            }
            "getResolution" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.buffer.width() as i64);
                ffi::lua_pushinteger(L, s.buffer.height() as i64);
                3
            }
            "maxResolution" => {
                let (w, h) = s.gpu.max_resolution();
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, w as i64);
                ffi::lua_pushinteger(L, h as i64);
                3
            }
            "setBackground" => {
                let color = ffi::lua_tointeger(L, 3) as u32;
                let pal = ffi::lua_gettop(L) >= 4 && ffi::lua_toboolean(L, 4) != 0;
                let old = s.buffer.background();
                let c = if pal { PackedColor::palette(color as u8) } else { PackedColor::rgb(color) };
                s.buffer.set_background(c);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, old.resolve(s.buffer.palette()) as i64);
                2
            }
            "getBackground" => {
                let bg = s.buffer.background();
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, bg.resolve(s.buffer.palette()) as i64);
                ffi::lua_pushboolean(L, bg.is_from_palette() as c_int);
                3
            }
            "setForeground" => {
                let color = ffi::lua_tointeger(L, 3) as u32;
                let pal = ffi::lua_gettop(L) >= 4 && ffi::lua_toboolean(L, 4) != 0;
                let old = s.buffer.foreground();
                let c = if pal { PackedColor::palette(color as u8) } else { PackedColor::rgb(color) };
                s.buffer.set_foreground(c);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, old.resolve(s.buffer.palette()) as i64);
                2
            }
            "getForeground" => {
                let fg = s.buffer.foreground();
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, fg.resolve(s.buffer.palette()) as i64);
                ffi::lua_pushboolean(L, fg.is_from_palette() as c_int);
                3
            }
            "set" => {
                let x = ffi::lua_tointeger(L, 3) as u32;
                let y = ffi::lua_tointeger(L, 4) as u32;
                let text = to_str(L, 5).unwrap_or("");
                let vert = ffi::lua_gettop(L) >= 6 && ffi::lua_toboolean(L, 6) != 0;
                s.buffer.set(x.saturating_sub(1), y.saturating_sub(1), text, vert);
                log::trace!("gpu.set({x}, {y}, {:?})", &text[..text.len().min(40)]);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "fill" => {
                let x = ffi::lua_tointeger(L, 3) as u32;
                let y = ffi::lua_tointeger(L, 4) as u32;
                let w = ffi::lua_tointeger(L, 5) as u32;
                let h = ffi::lua_tointeger(L, 6) as u32;
                let ch = to_str(L, 7).and_then(|s| s.chars().next()).unwrap_or(' ');
                s.buffer.fill(x.saturating_sub(1), y.saturating_sub(1), w, h, ch as u32);
                log::trace!("gpu.fill({x}, {y}, {w}, {h}, {:?})", ch);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "copy" => {
                let x  = ffi::lua_tointeger(L, 3) as u32;
                let y  = ffi::lua_tointeger(L, 4) as u32;
                let w  = ffi::lua_tointeger(L, 5) as u32;
                let h  = ffi::lua_tointeger(L, 6) as u32;
                let tx = ffi::lua_tointeger(L, 7) as i32;
                let ty = ffi::lua_tointeger(L, 8) as i32;
                s.buffer.copy(x.saturating_sub(1), y.saturating_sub(1), w, h, tx, ty);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "get" => {
                let x = (ffi::lua_tointeger(L, 3) as u32).saturating_sub(1);
                let y = (ffi::lua_tointeger(L, 4) as u32).saturating_sub(1);
                ffi::lua_pushboolean(L, 1);
                if let Some(cell) = s.buffer.get(x, y) {
                    let ch_str: String = char::from_u32(cell.codepoint).unwrap_or(' ').to_string();
                    push_str(L, &ch_str);
                    ffi::lua_pushinteger(L, cell.foreground.resolve(s.buffer.palette()) as i64);
                    ffi::lua_pushinteger(L, cell.background.resolve(s.buffer.palette()) as i64);
                    // fgIndex, bgIndex (nil for non-palette)
                    ffi::lua_pushnil(L);
                    ffi::lua_pushnil(L);
                    6
                } else {
                    push_str(L, " ");
                    ffi::lua_pushinteger(L, 0xFFFFFF);
                    ffi::lua_pushinteger(L, 0x000000);
                    ffi::lua_pushnil(L);
                    ffi::lua_pushnil(L);
                    6
                }
            }
            "setDepth" => {
                let bits = ffi::lua_tointeger(L, 3) as u8;
                let depth = match bits {
                    1 => ColorDepth::OneBit,
                    4 => ColorDepth::FourBit,
                    8 => ColorDepth::EightBit,
                    _ => return inv_err(L, "unsupported depth"),
                };
                let changed = s.buffer.set_color_depth(depth);
                log::debug!("gpu.setDepth({bits})");
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, changed as c_int);
                2
            }
            "getDepth" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.buffer.palette().depth().bits() as i64);
                2
            }
            "maxDepth" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.gpu.max_depth().bits() as i64);
                2
            }
            "setViewport" => {
                let w = ffi::lua_tointeger(L, 3) as u32;
                let h = ffi::lua_tointeger(L, 4) as u32;
                let changed = s.buffer.set_viewport(w, h);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, changed as c_int);
                2
            }
            "getViewport" => {
                let (vw, vh) = s.buffer.viewport();
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, vw as i64);
                ffi::lua_pushinteger(L, vh as i64);
                3
            }
            "setPaletteColor" => {
                let idx = ffi::lua_tointeger(L, 3) as usize;
                let color = ffi::lua_tointeger(L, 4) as u32;
                let old = s.buffer.palette().get(idx);
                s.buffer.palette_mut().set(idx, color);
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, old as i64);
                2
            }
            "getPaletteColor" => {
                let idx = ffi::lua_tointeger(L, 3) as usize;
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, s.buffer.palette().get(idx) as i64);
                2
            }
            "freeMemory" | "totalMemory" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, 65536);
                2
            }
            "getActiveBuffer" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, 0);
                2
            }
            "setActiveBuffer" | "buffers" | "allocateBuffer"
            | "freeBuffer" | "freeAllBuffers" | "getBufferSize" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushinteger(L, 0);
                2
            }
            _ => inv_err(L, "no such method"),
        }

        // ─── Screen ─────────────────────────────────────────────
        "screen" => match method {
            "isOn" | "turnOn" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "turnOff" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 1);
                2
            }
            "getAspectRatio" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushnumber(L, 1.0);
                ffi::lua_pushnumber(L, 1.0);
                3
            }
            "getKeyboards" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_newtable(L);
                push_str(L, &s.keyboard.address);
                ffi::lua_rawseti(L, -2, 1);
                2
            }
            "isPrecise" | "isTouchModeInverted" => {
                ffi::lua_pushboolean(L, 1);
                ffi::lua_pushboolean(L, 0);
                2
            }
            "setPrecise" | "setTouchModeInverted" => {
                ffi::lua_pushboolean(L, 1);
                1
            }
            _ => inv_err(L, "no such method"),
        }

        // ─── Keyboard ───────────────────────────────────────────
        "keyboard" => inv_err(L, "no such method"),

        // ─── Computer ───────────────────────────────────────────
        "computer" => match method {
            "beep" => {
                let top = ffi::lua_gettop(L);
                if top >= 3
                    && ffi::lua_type(L, 3) == ffi::LUA_TSTRING
                    && ffi::lua_isnumber(L, 3) == 0
                {
                    let pattern = to_str(L, 3).unwrap_or("");
                    with_sound(|snd| snd.beep_pattern(pattern));
                } else {
                    let freq = if top >= 3 && ffi::lua_isnumber(L, 3) != 0 {
                        ffi::lua_tonumber(L, 3)
                    } else { 440.0 };
                    let dur = if top >= 4 && ffi::lua_isnumber(L, 4) != 0 {
                        ffi::lua_tonumber(L, 4)
                    } else { 0.1 };
                    with_sound(|snd| snd.beep(freq, dur));
                }
                ffi::lua_pushboolean(L, 1);
                1
            }
            _ => inv_err(L, "no such method"),
        }

        // ─── Filesystem (stub for future OpenOS) ────────────────
        "filesystem" => {
            let fs_idx = s.filesystems.iter().position(|f| f.address == addr);
            match fs_idx {
                Some(idx) => dispatch_filesystem(L, &mut s.filesystems[idx], &method),
                None => inv_err(L, "filesystem not found"),
            }
        }
        "drive" => {
            let drive_idx = s.drives.iter().position(|d| d.address == addr);
            match drive_idx {
                Some(idx) => dispatch_drive(L, &mut s.drives[idx], method),
                None => inv_err(L, "drive not found"),
            }
        }
        _ => inv_err(L, "no such component type"),
    }
}

/// Dispatch `component.invoke` calls to an unmanaged drive.
unsafe fn dispatch_drive(
    L: *mut ffi::lua_State,
    drive: &mut crate::components::drive::Drive,
    method: &str,
) -> c_int {
    match method {
        "getLabel" => {
            ffi::lua_pushboolean(L, 1);
            match drive.get_label() {
                Some(l) => push_str(L, l),
                None    => ffi::lua_pushnil(L),
            }
            2
        }
        "setLabel" => {
            let label = to_str(L, 3);
            match drive.set_label(label) {
                Ok(()) => {
                    ffi::lua_pushboolean(L, 1);
                    match drive.get_label() {
                        Some(l) => push_str(L, l),
                        None    => ffi::lua_pushnil(L),
                    }
                    2
                }
                Err(e) => inv_err(L, e),
            }
        }
        "getCapacity" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, drive.capacity() as i64);
            2
        }
        "getSectorSize" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, drive.sector_size() as i64);
            2
        }
        "getPlatterCount" => {
            ffi::lua_pushboolean(L, 1);
            ffi::lua_pushinteger(L, drive.platter_count() as i64);
            2
        }
        "readSector" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let _prof = crate::profiler::scope(crate::profiler::Cat::DriveOp, "readSector");
            // Lua: readSector(sector) -- 1-based
            let sector = (ffi::lua_tointeger(L, 3) - 1) as usize;
            log::trace!("drive.readSector({})", sector);
            match drive.read_sector(sector) {
                Ok((data, _seek)) => {
                    ffi::lua_pushboolean(L, 1);
                    push_bytes(L, &data);
                    2
                }
                Err(e) => inv_err(L, e),
            }
        }
        "writeSector" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let _prof = crate::profiler::scope(crate::profiler::Cat::DriveOp, "writeSector");
            let sector = (ffi::lua_tointeger(L, 3) - 1) as usize;
            let data = read_bytes(L, 4);
            log::trace!("drive.writeSector({}, {} bytes)", sector, data.len());
            match drive.write_sector(sector, &data) {
                Ok(_seek) => {
                    ffi::lua_pushboolean(L, 1);
                    1
                }
                Err(e) => inv_err(L, e),
            }
        }
        "readByte" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            // Lua: readByte(offset) -- 1-based
            let offset = (ffi::lua_tointeger(L, 3) - 1) as usize;
            match drive.read_byte(offset) {
                Ok((val, _seek)) => {
                    ffi::lua_pushboolean(L, 1);
                    ffi::lua_pushinteger(L, val as i64);
                    2
                }
                Err(e) => inv_err(L, e),
            }
        }
        "writeByte" => {
            with_sound(|snd| snd.play_disk_sound(crate::sound::DiskSound::random_hdd()));
            let offset = (ffi::lua_tointeger(L, 3) - 1) as usize;
            let value = ffi::lua_tointeger(L, 4) as u8;
            match drive.write_byte(offset, value) {
                Ok(_seek) => {
                    ffi::lua_pushboolean(L, 1);
                    1
                }
                Err(e) => inv_err(L, e),
            }
        }
        _ => inv_err(L, "no such method"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// system.* callbacks
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" fn sys_timeout(L: *mut ffi::lua_State) -> c_int {
    let s = st(L); ffi::lua_pushnumber(L, s.timeout); 1
}
unsafe extern "C" fn sys_allow_bytecode(L: *mut ffi::lua_State) -> c_int {
    let s = st(L); ffi::lua_pushboolean(L, s.allow_bytecode as c_int); 1
}
unsafe extern "C" fn sys_allow_gc(L: *mut ffi::lua_State) -> c_int {
    let s = st(L); ffi::lua_pushboolean(L, s.allow_gc as c_int); 1
}

// ═══════════════════════════════════════════════════════════════════════
// Registration — inject all host APIs into the Lua state
// ═══════════════════════════════════════════════════════════════════════

/// `computer.beep(frequency, duration)` or `computer.beep(pattern)`
unsafe extern "C" fn c_beep(L: *mut ffi::lua_State) -> c_int {
    let top = ffi::lua_gettop(L);

    // Detect pattern mode: first arg is a string that is NOT a number
    if top >= 1
        && ffi::lua_type(L, 1) == ffi::LUA_TSTRING
        && ffi::lua_isnumber(L, 1) == 0
    {
        let pattern = to_str(L, 1).unwrap_or("");
        with_sound(|snd| snd.beep_pattern(pattern));
    } else {
        // Frequency mode
        let freq = if top >= 1 && ffi::lua_isnumber(L, 1) != 0 {
            ffi::lua_tonumber(L, 1)
        } else {
            440.0
        };
        let dur = if top >= 2 && ffi::lua_isnumber(L, 2) != 0 {
            ffi::lua_tonumber(L, 2)
        } else {
            0.1
        };
        with_sound(|snd| snd.beep(freq, dur));
    }

    ffi::lua_pushboolean(L, 1);
    1
}
/// Helper: register a `name → fn` in the table at stack top.
unsafe fn reg(L: *mut ffi::lua_State, name: &str, f: ffi::lua_CFunction) {
    let c = CString::new(name).unwrap();
    ffi::lua_pushcfunction(L, f);
    ffi::lua_setfield(L, -2, c.as_ptr());
}

/// Inject all host APIs into the Lua state and stash the state pointer.
pub fn register_apis(lua: &LuaState, state: &mut EmulatorState) {
    log::debug!("Registering host APIs...");
    unsafe {
        let L = lua.ptr();

        // Stash EmulatorState pointer in the registry.
        ffi::lua_pushlightuserdata(L, state as *mut EmulatorState as *mut _);
        ffi::lua_setfield(L, ffi::LUA_REGISTRYINDEX, STATE_KEY.as_ptr() as *const _);

        // ── computer ──
        ffi::lua_newtable(L);
        reg(L, "realTime",          c_real_time);
        reg(L, "uptime",            c_uptime);
        reg(L, "address",           c_address);
        reg(L, "freeMemory",        c_free_memory);
        reg(L, "totalMemory",       c_total_memory);
        reg(L, "pushSignal",        c_push_signal);
        reg(L, "tmpAddress",        c_tmp_address);
        reg(L, "users",             c_users);
        reg(L, "energy",            c_energy);
        reg(L, "maxEnergy",         c_max_energy);
        reg(L, "getArchitectures",  c_get_architectures);
        reg(L, "getArchitecture",   c_get_architecture);
        reg(L, "setArchitecture",   c_set_architecture);
        reg(L, "isRobot",           c_is_robot);
        reg(L, "getBootAddress",    c_get_boot_address);
        reg(L, "setBootAddress",    c_set_boot_address);
        reg(L, "addUser",           c_add_user);
        reg(L, "removeUser",        c_remove_user);
        ffi::lua_setglobal(L, b"computer\0".as_ptr() as *const _);

        // ── component ──
        ffi::lua_newtable(L);
        reg(L, "list",    comp_list);
        reg(L, "type",    comp_type);
        reg(L, "invoke",  comp_invoke);
        reg(L, "methods", comp_methods);
        reg(L, "doc",     comp_doc);
        reg(L, "slot",    comp_slot);
        ffi::lua_setglobal(L, b"component\0".as_ptr() as *const _);

        // ── system ──
        ffi::lua_newtable(L);
        reg(L, "timeout",        sys_timeout);
        reg(L, "allowBytecode",  sys_allow_bytecode);
        reg(L, "allowGC",        sys_allow_gc);
        reg(L, "beep", c_beep);
        ffi::lua_setglobal(L, b"system\0".as_ptr() as *const _);

        // ── unicode (Lua shim over utf8 stdlib) ──
        lua.exec(UNICODE_SHIM, "=unicode_init").ok();

        // ── userdata (stub) ──
        ffi::lua_newtable(L);
        ffi::lua_setglobal(L, b"userdata\0".as_ptr() as *const _);

        // Remove dangerous globals that machine.lua doesn't expect.
        ffi::lua_pushnil(L); ffi::lua_setglobal(L, b"dofile\0".as_ptr() as *const _);
        ffi::lua_pushnil(L); ffi::lua_setglobal(L, b"loadfile\0".as_ptr() as *const _);
    }
    log::debug!("Host APIs registered: computer, component, system, unicode, userdata");
}

/// Lua shim that builds the `unicode` global from Lua's `utf8` stdlib.
const UNICODE_SHIM: &str = r#"
unicode = {}
unicode.char = utf8.char
unicode.len = function(s)
    local ok, n = pcall(utf8.len, s)
    return ok and n or #s
end
unicode.lower = string.lower
unicode.upper = string.upper
unicode.reverse = function(s)
    local t = {}; for _,c in utf8.codes(s) do table.insert(t, 1, utf8.char(c)) end
    return table.concat(t)
end
unicode.sub = function(s, i, j)
    local len = unicode.len(s)
    if i < 0 then i = len + i + 1 end
    if i < 1 then i = 1 end
    j = j or len
    if j < 0 then j = len + j + 1 end
    if j > len then j = len end
    if j < i then return "" end
    local si = utf8.offset(s, i) or 1
    local sj = (utf8.offset(s, j + 1) or (#s + 1)) - 1
    return s:sub(si, sj)
end
unicode.isWide = function() return false end
unicode.charWidth = function() return 1 end
unicode.wlen = unicode.len
unicode.wtrunc = function(s, n)
    if unicode.len(s) <= n then return s end
    return unicode.sub(s, 1, n)
end
"#;

// ═══════════════════════════════════════════════════════════════════════
// Kernel lifecycle
// ═══════════════════════════════════════════════════════════════════════

/// Load `machine.lua`, wrap it in a coroutine, leave the thread at index 1.
pub fn load_kernel(lua: &LuaState, machine_lua_src: &str) -> LuaResult {
    log::debug!("Loading kernel ({} bytes)...", machine_lua_src.len());
    lua.load_string(machine_lua_src, "=machine")?;
    let thread = lua.new_thread();
    lua.push_value(1);
    lua.xmove_to(&thread, 1);
    unsafe { ffi::lua_rotate(lua.ptr(), 1, 1); }
    lua.pop(1);
    log::debug!("Kernel loaded into coroutine thread");
    Ok(())
}

/// Resume the kernel coroutine one step.
///
/// Pushes pending signals onto the thread stack before resuming.
pub fn step_kernel(lua: &LuaState, emu: &mut EmulatorState) -> ExecResult {
    let _prof = crate::profiler::scope(crate::profiler::Cat::LuaStep, "step_kernel");
    let thread = match lua.get_thread(1) {
        Some(t) => t,
        None => {
            log::error!("Kernel thread missing from Lua stack");
            return ExecResult::Error("kernel thread missing".into());
        }
    };

    let nargs = if let Some(sig) = emu.signals.pop() {
        log::trace!("kernel: delivering signal '{}' ({} args)",
            sig.name, sig.args.len());
        push_signal(&thread, &sig);
        1 + sig.args.len() as c_int
    } else {
        0
    };

    let (status, nres) = thread.resume(nargs);
    log::trace!("kernel: resume -> status={status}, nres={nres}");
    dispatch_yield(lua, &thread, status, nres, 0)
}

/// Interpret the result of a `lua_resume`.  Recursive for sync-call chains.
fn dispatch_yield(
    lua: &LuaState,
    thread: &LuaState,
    status: i32,
    nres: i32,
    depth: u32,
) -> ExecResult {
    if depth > 64 {
        return ExecResult::Error("sync call nesting limit".into());
    }
    match status {
        ffi::LUA_YIELD => {
            if nres >= 1 && thread.is_bool(1) {
                let reboot = thread.to_bool(1);
                thread.set_top(0);
                ExecResult::Shutdown { reboot }
            } else if nres >= 1 && thread.is_function(1) {
                // Synchronized call — execute on main state, return result.
                thread.xmove_to(lua, 1);
                let rc = unsafe { ffi::lua_pcall(lua.ptr(), 0, 1, 0) };
                if rc != ffi::LUA_OK {
                    let e = lua.to_string_owned(-1).unwrap_or_default();
                    lua.pop(1);
                    return ExecResult::Error(e);
                }
                lua.xmove_to(thread, 1);
                let (s2, n2) = thread.resume(1);
                dispatch_yield(lua, thread, s2, n2, depth + 1)
            } else if nres >= 1 && thread.is_number(1) {
                let secs = thread.to_number(1);
                thread.set_top(0);
                ExecResult::Sleep(secs)
            } else {
                // Nil yield or no values → "sleep until signal" (like OC's Int.MaxValue ticks).
                thread.set_top(0);
                ExecResult::Sleep(f64::INFINITY)
            }
        }
        ffi::LUA_OK => {
            // machine.lua returned: pcallTimeoutCheck(pcall(main))
            //   → (false, "error msg") on crash
            //   → (true)               on clean exit (shouldn't happen)
            let nres = thread.top();
            if nres >= 2 && thread.is_bool(1) && !thread.to_bool(1) {
                let msg = thread.to_string_owned(2)
                    .unwrap_or_else(|| "unknown error".into());
                thread.set_top(0);
                ExecResult::Error(msg)
            } else if nres >= 1 && thread.is_bool(1) && thread.to_bool(1) {
                thread.set_top(0);
                ExecResult::Shutdown { reboot: false }
            } else {
                thread.set_top(0);
                ExecResult::Halted
            }
        }
        _ => {
            let msg = thread.to_string_owned(-1).unwrap_or_default();
            thread.set_top(0);
            ExecResult::Error(msg)
        }
    }
}

/// Push a signal's name + args onto a Lua thread's stack.
fn push_signal(thread: &LuaState, sig: &Signal) {
    thread.push_string(&sig.name);
    for arg in &sig.args {
        match arg {
            SignalArg::Nil       => thread.push_nil(),
            SignalArg::Bool(b)   => thread.push_bool(*b),
            SignalArg::Int(n)    => thread.push_integer(*n),
            SignalArg::Float(n)  => thread.push_number(*n),
            SignalArg::Str(s)    => thread.push_string(s),
            SignalArg::Bytes(b)  => thread.push_bytes(b),
        }
    }
}