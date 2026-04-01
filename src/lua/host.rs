//! # OpenComputers Machine ↔ Lua bridge
//!
//! Implements the host-side API that `machine.lua` expects:
//!
//! | Lua global     | Provider                 |
//! |----------------|--------------------------|
//! | `computer.*`   | [`ComputerApi`]          |
//! | `component.*`  | [`ComponentApi`]         |
//! | `unicode.*`    | Standard `utf8` lib      |
//! | `system.*`     | [`SystemApi`]            |
//!
//! The host creates a Lua state, injects these APIs, loads `machine.lua`
//! as the kernel, and drives execution via coroutine resume/yield.

use std::ffi::CString;
use std::os::raw::c_int;
use std::time::Instant;

use super::ffi;
use super::state::{LuaState, LuaResult, LuaError};

/// Opaque context pointer stashed in the Lua registry, allowing
/// C callbacks to access the `HostContext` without global state.
const CONTEXT_REGISTRY_KEY: &str = "__oc_host_ctx";

/// Execution result from one step of the machine.
#[derive(Debug)]
pub enum ExecResult {
    /// Lua yielded, wants to sleep for N seconds.
    Sleep(f64),
    /// Lua yielded a function for synchronized execution.
    SynchronizedCall,
    /// Lua requested shutdown (reboot = true → restart).
    Shutdown { reboot: bool },
    /// Lua execution finished (machine halted).
    Halted,
    /// An error occurred.
    Error(String),
}

/// Host-side context accessible from Lua callbacks.
///
/// This is stashed in the Lua registry as a light userdata pointer.
/// **Lifetime**: must live at least as long as the `LuaState`.
pub struct HostContext {
    /// Wall-clock time at which the machine was started.
    pub start_time: Instant,
    /// Monotonic uptime in seconds (updated from [`Machine::uptime`]).
    pub uptime: f64,
    /// CPU time accumulated.
    pub cpu_time: f64,
    /// Address of this computer's node.
    pub address: String,
    /// Timeout in seconds (from config).
    pub timeout: f64,
    /// World time in ticks (for `os.time` / `os.date`).
    pub world_time: u64,
    /// Whether power is ignored.
    pub ignore_power: bool,
    /// Allow bytecode loading.
    pub allow_bytecode: bool,
    /// Allow `__gc` metamethods.
    pub allow_gc: bool,
}

impl Default for HostContext {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            uptime: 0.0,
            cpu_time: 0.0,
            address: String::new(),
            timeout: 5.0,
            world_time: 0,
            ignore_power: true,
            allow_bytecode: false,
            allow_gc: false,
        }
    }
}

/// Retrieve the [`HostContext`] from a Lua state.
///
/// # Safety
/// Requires a valid `lua_State` with the context registered.
unsafe fn get_context(L: *mut ffi::lua_State) -> &'static mut HostContext {
    let key = CString::new(CONTEXT_REGISTRY_KEY).unwrap();
    ffi::lua_getfield(L, ffi::LUA_REGISTRYINDEX, key.as_ptr());
    let ptr = ffi::lua_touserdata(L, -1) as *mut HostContext;
    ffi::lua_pop(L, 1);
    &mut *ptr
}

// ─── C callback implementations ─────────────────────────────────────────

/// `computer.realTime() -> number` / wall-clock seconds.
unsafe extern "C" fn computer_real_time(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    let elapsed = ctx.start_time.elapsed().as_secs_f64();
    ffi::lua_pushnumber(L, elapsed);
    1
}

/// `computer.uptime() -> number` / machine uptime in seconds.
unsafe extern "C" fn computer_uptime(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    ffi::lua_pushnumber(L, ctx.uptime);
    1
}

/// `computer.address() -> string` / machine's network address.
unsafe extern "C" fn computer_address(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    let s = &ctx.address;
    ffi::lua_pushlstring(L, s.as_ptr() as *const _, s.len());
    1
}

/// `computer.freeMemory() -> number`.
unsafe extern "C" fn computer_free_memory(L: *mut ffi::lua_State) -> c_int {
    // In our emulator we don't limit Lua memory, return a large value.
    ffi::lua_pushinteger(L, 2 * 1024 * 1024); // 2 MiB
    1
}

/// `computer.totalMemory() -> number`.
unsafe extern "C" fn computer_total_memory(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushinteger(L, 2 * 1024 * 1024);
    1
}

/// `computer.pushSignal(name, ...) -> boolean`.
unsafe extern "C" fn computer_push_signal(L: *mut ffi::lua_State) -> c_int {
    // For now, signals pushed from Lua are a no-op (would need signal queue access).
    ffi::lua_pushboolean(L, 1);
    1
}

/// `computer.tmpAddress() -> string | nil`.
unsafe extern "C" fn computer_tmp_address(L: *mut ffi::lua_State) -> c_int {
    // Return nil if no tmpfs is configured.
    ffi::lua_pushnil(L);
    1
}

/// `computer.users() -> string...`.
unsafe extern "C" fn computer_users(L: *mut ffi::lua_State) -> c_int {
    0 // no users in the emulator
}

/// `computer.energy() -> number`.
unsafe extern "C" fn computer_energy(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    if ctx.ignore_power {
        ffi::lua_pushnumber(L, f64::INFINITY);
    } else {
        ffi::lua_pushnumber(L, 10000.0);
    }
    1
}

/// `computer.maxEnergy() -> number`.
unsafe extern "C" fn computer_max_energy(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnumber(L, 10000.0);
    1
}

/// `computer.getArchitectures() -> table`.
unsafe extern "C" fn computer_get_architectures(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_newtable(L);
    ffi::lua_pushstring(L, b"Lua 5.4\0".as_ptr() as _);
    ffi::lua_rawseti(L, -2, 1);
    1
}

/// `computer.getArchitecture() -> string`.
unsafe extern "C" fn computer_get_architecture(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushstring(L, b"Lua 5.4\0".as_ptr() as _);
    1
}

/// `system.timeout() -> number`.
unsafe extern "C" fn system_timeout(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    ffi::lua_pushnumber(L, ctx.timeout);
    1
}

/// `system.allowBytecode() -> boolean`.
unsafe extern "C" fn system_allow_bytecode(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    ffi::lua_pushboolean(L, ctx.allow_bytecode as c_int);
    1
}

/// `system.allowGC() -> boolean`.
unsafe extern "C" fn system_allow_gc(L: *mut ffi::lua_State) -> c_int {
    let ctx = get_context(L);
    ffi::lua_pushboolean(L, ctx.allow_gc as c_int);
    1
}

/// `component.list(filter?, exact?) -> table`.
unsafe extern "C" fn component_list(L: *mut ffi::lua_State) -> c_int {
    // Stub: return empty table.
    ffi::lua_newtable(L);
    1
}

/// `component.type(address) -> string | nil`.
unsafe extern "C" fn component_type(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnil(L);
    ffi::lua_pushstring(L, b"no such component\0".as_ptr() as _);
    2
}

/// `component.invoke(addr, method, ...) -> ...`.
unsafe extern "C" fn component_invoke(L: *mut ffi::lua_State) -> c_int {
    // Stub.
    ffi::lua_pushboolean(L, 0);
    ffi::lua_pushstring(L, b"not implemented\0".as_ptr() as _);
    2
}

/// `component.methods(addr) -> table`.
unsafe extern "C" fn component_methods(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_newtable(L);
    1
}

/// `component.doc(addr, method) -> string | nil`.
unsafe extern "C" fn component_doc(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnil(L);
    1
}

/// `component.slot(addr) -> number | nil`.
unsafe extern "C" fn component_slot(L: *mut ffi::lua_State) -> c_int {
    ffi::lua_pushnil(L);
    ffi::lua_pushstring(L, b"no such component\0".as_ptr() as _);
    2
}

// ─── Host setup ─────────────────────────────────────────────────────────

/// Register all OC APIs into the given Lua state.
///
/// The `ctx` must outlive the Lua state.
pub fn register_apis(lua: &LuaState, ctx: &mut HostContext) {
    unsafe {
        let L = lua.ptr();

        // Stash context pointer in registry.
        let key = CString::new(CONTEXT_REGISTRY_KEY).unwrap();
        ffi::lua_pushlightuserdata(L, ctx as *mut HostContext as *mut _);
        ffi::lua_setfield(L, ffi::LUA_REGISTRYINDEX, key.as_ptr());

        // ── computer ────────────────────────────────────────────────
        ffi::lua_newtable(L);
        register(L, "realTime", computer_real_time);
        register(L, "uptime", computer_uptime);
        register(L, "address", computer_address);
        register(L, "freeMemory", computer_free_memory);
        register(L, "totalMemory", computer_total_memory);
        register(L, "pushSignal", computer_push_signal);
        register(L, "tmpAddress", computer_tmp_address);
        register(L, "users", computer_users);
        register(L, "energy", computer_energy);
        register(L, "maxEnergy", computer_max_energy);
        register(L, "getArchitectures", computer_get_architectures);
        register(L, "getArchitecture", computer_get_architecture);
        ffi::lua_setglobal(L, CString::new("computer").unwrap().as_ptr());

        // ── component ───────────────────────────────────────────────
        ffi::lua_newtable(L);
        register(L, "list", component_list);
        register(L, "type", component_type);
        register(L, "invoke", component_invoke);
        register(L, "methods", component_methods);
        register(L, "doc", component_doc);
        register(L, "slot", component_slot);
        ffi::lua_setglobal(L, CString::new("component").unwrap().as_ptr());

        // ── system ──────────────────────────────────────────────────
        ffi::lua_newtable(L);
        register(L, "timeout", system_timeout);
        register(L, "allowBytecode", system_allow_bytecode);
        register(L, "allowGC", system_allow_gc);
        ffi::lua_setglobal(L, CString::new("system").unwrap().as_ptr());

        // ── unicode (delegate to Lua's utf8 lib) ────────────────────
        ffi::lua_newtable(L);
        ffi::lua_setglobal(L, CString::new("unicode").unwrap().as_ptr());

        // ── userdata (stub) ─────────────────────────────────────────
        ffi::lua_newtable(L);
        ffi::lua_setglobal(L, CString::new("userdata").unwrap().as_ptr());

        // Remove dangerous globals (matches machine.lua expectations).
        ffi::lua_pushnil(L);
        ffi::lua_setglobal(L, CString::new("dofile").unwrap().as_ptr());
        ffi::lua_pushnil(L);
        ffi::lua_setglobal(L, CString::new("loadfile").unwrap().as_ptr());
    }
}

/// Helper: register a C function in the table at stack top.
unsafe fn register(L: *mut ffi::lua_State, name: &str, f: ffi::lua_CFunction) {
    let cname = CString::new(name).unwrap();
    ffi::lua_pushcfunction(L, f);
    ffi::lua_setfield(L, -2, cname.as_ptr());
}

/// Load and initialise `machine.lua`, returning the kernel coroutine.
///
/// After this call, the Lua stack of `lua` has:
///   `[1]` = kernel thread (coroutine)
///
/// The caller must then resume the thread to begin execution.
pub fn load_kernel(lua: &LuaState, machine_lua_src: &str) -> LuaResult {
    // Load machine.lua as a function.
    lua.load_string(machine_lua_src, "=machine")?;

    // Create the kernel coroutine from the loaded chunk.
    // The coroutine is left at stack position 1.
    let _thread = lua.new_thread(); // pushes thread onto parent stack
    lua.push_value(-2);             // copy the chunk
    lua.xmove_to(&_thread, 1);     // move chunk into thread
    lua.pop(1);                     // pop the original chunk
    // Now stack: [1] = thread

    Ok(())
}

/// Resume the kernel coroutine one step.
///
/// `lua` must have the kernel thread at stack index 1.
/// `signal_args` is the number of signal arguments pushed onto the
/// thread's stack before calling this.
pub fn step_kernel(lua: &LuaState, signal_args: i32) -> ExecResult {
    let thread = match lua.get_thread(1) {
        Some(t) => t,
        None => return ExecResult::Error("kernel thread not found at stack index 1".into()),
    };

    let (status, nresults) = thread.resume(signal_args);

    match status {
        ffi::LUA_YIELD => {
            if nresults >= 1 && thread.is_bool(1) {
                // Shutdown: `coroutine.yield(reboot)`.
                let reboot = thread.to_bool(1);
                thread.pop(nresults);
                ExecResult::Shutdown { reboot }
            } else if nresults >= 1 && thread.is_function(1) {
                // Synchronized call.
                thread.pop(nresults);
                ExecResult::SynchronizedCall
            } else if nresults >= 1 && thread.is_number(1) {
                // Sleep for N seconds.
                let secs = thread.to_number(1);
                thread.pop(nresults);
                ExecResult::Sleep(secs)
            } else {
                thread.set_top(0);
                ExecResult::Sleep(0.05) // default: wake next tick
            }
        }
        ffi::LUA_OK => {
            // Coroutine finished = halted.
            ExecResult::Halted
        }
        _ => {
            // Error.
            let msg = thread.to_string_owned(-1)
                .unwrap_or_else(|| "unknown error".into());
            thread.set_top(0);
            ExecResult::Error(msg)
        }
    }
}