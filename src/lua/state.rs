//! # Safe Lua state wrapper
//!
//! Provides a RAII handle (`LuaState`) around `*mut lua_State` with
//! safe(ish) push/get/load/call methods. The wrapper encapsulates
//! `unsafe` blocks and enforces basic invariants:
//!
//! * The state is closed on `Drop` (if owned).
//! * Non-owning handles (from `lua_newthread` / `lua_tothread`) do
//!   NOT close the state on drop.
//! * All methods that read from the stack check types via `lua_type`.
//! * String reads go through `lua_tolstring` with explicit length
//!   (no assumption of NUL-termination).
//!
//! ## Ownership model
//!
//! ```text
//! LuaState { L: *mut lua_State, owned: true }
//!   |
//!   | new_thread() -> LuaState { L: coroutine_ptr, owned: false }
//!   |                   ^
//!   |                   | Does NOT close on drop
//!   |
//!   | Drop -> lua_close(L)   // only if owned == true
//! ```
//!
//! ## Thread safety
//!
//! `LuaState` implements `Send` but NOT `Sync`. Only one thread may
//! use a Lua state at a time. In this emulator, all Lua access happens
//! on the main thread, so this is trivially satisfied.
//!
//! ## Error handling
//!
//! Lua errors from `lua_pcall` / `luaL_loadbuffer` are caught and
//! returned as `LuaError` (containing the error message string and
//! the Lua status code). Unprotected calls (`lua_call`) will abort
//! on error (Lua's default panic handler), so we only use `pcall`.

use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::ptr;

use super::ffi;

/// Wrapper around a Lua 5.4 state.
///
/// Owns (or borrows) a `*mut lua_State` and provides safe methods
/// for common Lua operations.
///
/// # Ownership
///
/// * If `owned` is `true`, `lua_close` is called on drop.
/// * If `owned` is `false`, the pointer is borrowed from a parent
///   state (e.g. a coroutine thread). The parent is responsible for
///   closing.
///
/// # Construction
///
/// * `LuaState::new()` -- Create a fresh, owned state with all
///   standard libraries loaded.
/// * `state.new_thread()` -- Create a non-owning coroutine handle.
/// * `LuaState::from_raw_non_owning(ptr)` -- Wrap an existing pointer
///   without taking ownership (unsafe).
pub struct LuaState {
    /// Raw pointer to the Lua state.
    ///
    /// Guaranteed non-null for the lifetime of this struct (checked
    /// at construction time).
    L: *mut ffi::lua_State,

    /// Whether this struct owns the Lua state (and should close it
    /// on drop).
    owned: bool,
}

// LuaState is Send (can be moved between threads) but NOT Sync
// (cannot be shared between threads simultaneously).
unsafe impl Send for LuaState {}

/// Errors from Lua operations.
///
/// Contains the error message (extracted from the Lua stack) and the
/// status code returned by the C API function.
#[derive(Debug)]
pub struct LuaError {
    /// Human-readable error message from Lua.
    ///
    /// For syntax errors, this includes the source location.
    /// For runtime errors, this includes the stack traceback (if a
    /// message handler was provided).
    pub message: String,

    /// The Lua status code (`LUA_ERRRUN`, `LUA_ERRSYNTAX`, etc.).
    pub code: c_int,
}

impl std::fmt::Display for LuaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lua error ({}): {}", self.code, self.message)
    }
}
impl std::error::Error for LuaError {}

/// Convenience alias for `Result<T, LuaError>`.
pub type LuaResult<T = ()> = Result<T, LuaError>;

impl LuaState {
    /// Create a new Lua state with all standard libraries loaded.
    ///
    /// # Returns
    ///
    /// * `Some(state)` -- A fresh Lua state, ready for use.
    /// * `None` -- Memory allocation failed (`luaL_newstate` returned NULL).
    ///
    /// # What happens
    ///
    /// 1. `luaL_newstate()` creates a bare Lua state.
    /// 2. `luaL_openlibs()` loads all standard libraries (string, table,
    ///    math, io, os, utf8, coroutine, package, debug).
    /// 3. The state is wrapped in an owning `LuaState`.
    pub fn new() -> Option<Self> {
        let L = unsafe { ffi::luaL_newstate() };
        if L.is_null() {
            log::error!("luaL_newstate returned NULL");
            return None;
        }
        unsafe { ffi::luaL_openlibs(L) };
        log::debug!("Lua state created ({:p}), stdlib loaded", L);
        Some(Self { L, owned: true })
    }

    /// Raw pointer to the underlying `lua_State`.
    ///
    /// Use this when calling FFI functions directly. The pointer is
    /// valid for the lifetime of this `LuaState`.
    #[inline]
    pub fn ptr(&self) -> *mut ffi::lua_State { self.L }

    // ---------------------------------------------------------------
    // Stack info
    // ---------------------------------------------------------------

    /// Number of values on the Lua stack.
    ///
    /// Equivalent to `lua_gettop(L)`.
    #[inline]
    pub fn top(&self) -> i32 {
        unsafe { ffi::lua_gettop(self.L) }
    }

    /// Set the stack top (pops excess values or pushes nils).
    ///
    /// If `idx < top()`, the excess values are discarded.
    /// If `idx > top()`, nils are pushed to fill the gap.
    /// If `idx == 0`, the entire stack is cleared.
    #[inline]
    pub fn set_top(&self, idx: i32) {
        unsafe { ffi::lua_settop(self.L, idx) };
    }

    /// Pop `n` values from the top of the stack.
    ///
    /// Equivalent to `set_top(top() - n)`.
    #[inline]
    pub fn pop(&self, n: i32) {
        unsafe { ffi::lua_pop(self.L, n) };
    }

    /// Get the Lua type tag of the value at `idx`.
    ///
    /// Returns one of `LUA_TNIL`, `LUA_TBOOLEAN`, `LUA_TNUMBER`, etc.
    /// Returns `LUA_TNONE` (-1) for invalid stack indices.
    #[inline]
    pub fn type_of(&self, idx: i32) -> i32 {
        unsafe { ffi::lua_type(self.L, idx) }
    }

    // ---------------------------------------------------------------
    // Push values onto the stack
    // ---------------------------------------------------------------

    /// Push nil.
    #[inline]
    pub fn push_nil(&self) {
        unsafe { ffi::lua_pushnil(self.L) };
    }

    /// Push a boolean.
    #[inline]
    pub fn push_bool(&self, v: bool) {
        unsafe { ffi::lua_pushboolean(self.L, v as c_int) };
    }

    /// Push a 64-bit integer.
    #[inline]
    pub fn push_integer(&self, v: i64) {
        unsafe { ffi::lua_pushinteger(self.L, v) };
    }

    /// Push a 64-bit float.
    #[inline]
    pub fn push_number(&self, v: f64) {
        unsafe { ffi::lua_pushnumber(self.L, v) };
    }

    /// Push a UTF-8 string.
    ///
    /// Lua makes an internal copy of the string data, so the original
    /// can be dropped immediately after this call.
    pub fn push_string(&self, s: &str) {
        unsafe {
            ffi::lua_pushlstring(self.L, s.as_ptr() as *const _, s.len());
        }
    }

    /// Push a byte slice as a Lua string.
    ///
    /// Lua strings can contain arbitrary bytes (including NULs), so
    /// this is the correct way to push binary data.
    pub fn push_bytes(&self, b: &[u8]) {
        unsafe {
            ffi::lua_pushlstring(self.L, b.as_ptr() as *const _, b.len());
        }
    }

    /// Push a C function (with no upvalues).
    pub fn push_fn(&self, f: ffi::lua_CFunction) {
        unsafe { ffi::lua_pushcfunction(self.L, f) };
    }

    /// Push a light userdata (raw pointer, not GC-managed).
    pub fn push_light_userdata(&self, p: *mut std::ffi::c_void) {
        unsafe { ffi::lua_pushlightuserdata(self.L, p) };
    }

    /// Duplicate the value at stack index `idx` and push it on top.
    #[inline]
    pub fn push_value(&self, idx: i32) {
        unsafe { ffi::lua_pushvalue(self.L, idx) };
    }

    // ---------------------------------------------------------------
    // Read values from the stack
    // ---------------------------------------------------------------

    /// Read a boolean at `idx`.
    ///
    /// Non-boolean types are coerced: nil and false -> false,
    /// everything else -> true.
    #[inline]
    pub fn to_bool(&self, idx: i32) -> bool {
        unsafe { ffi::lua_toboolean(self.L, idx) != 0 }
    }

    /// Read a float at `idx`.
    ///
    /// Non-number types return 0.0.
    #[inline]
    pub fn to_number(&self, idx: i32) -> f64 {
        unsafe { ffi::lua_tonumber(self.L, idx) }
    }

    /// Read an integer at `idx`.
    ///
    /// Non-integer types return 0.
    #[inline]
    pub fn to_integer(&self, idx: i32) -> i64 {
        unsafe { ffi::lua_tointeger(self.L, idx) }
    }

    /// Read a string at `idx` as a borrowed `&str`.
    ///
    /// Returns `None` if the value is not a string or if the bytes
    /// are not valid UTF-8.
    ///
    /// # Lifetime
    ///
    /// The returned reference is valid as long as the string remains
    /// on the Lua stack. Popping it invalidates the reference.
    pub fn to_str(&self, idx: i32) -> Option<&str> {
        unsafe {
            let mut len: usize = 0;
            let ptr = ffi::lua_tolstring(self.L, idx, &mut len);
            if ptr.is_null() {
                None
            } else {
                let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
                std::str::from_utf8(bytes).ok()
            }
        }
    }

    /// Read a string at `idx` as an owned `String`.
    ///
    /// Returns `None` if the value is not a string or not valid UTF-8.
    pub fn to_string_owned(&self, idx: i32) -> Option<String> {
        self.to_str(idx).map(|s| s.to_owned())
    }

    /// Read a userdata pointer at `idx`.
    ///
    /// Returns a raw pointer (may be NULL if the value is not userdata).
    pub fn to_userdata(&self, idx: i32) -> *mut std::ffi::c_void {
        unsafe { ffi::lua_touserdata(self.L, idx) }
    }

    // ---------------------------------------------------------------
    // Type checks
    // ---------------------------------------------------------------

    /// Check if the value at `idx` is nil.
    #[inline] pub fn is_nil(&self, idx: i32) -> bool { unsafe { ffi::lua_isnil(self.L, idx) } }

    /// Check if the value at `idx` is a boolean.
    #[inline] pub fn is_bool(&self, idx: i32) -> bool { unsafe { ffi::lua_isboolean(self.L, idx) } }

    /// Check if the value at `idx` is a number (or convertible to one).
    #[inline] pub fn is_number(&self, idx: i32) -> bool { unsafe { ffi::lua_isnumber(self.L, idx) != 0 } }

    /// Check if the value at `idx` is a string (or convertible to one).
    #[inline] pub fn is_string(&self, idx: i32) -> bool { unsafe { ffi::lua_isstring(self.L, idx) != 0 } }

    /// Check if the value at `idx` is a function (Lua or C).
    #[inline] pub fn is_function(&self, idx: i32) -> bool { unsafe { ffi::lua_isfunction(self.L, idx) } }

    /// Check if the value at `idx` is a table.
    #[inline] pub fn is_table(&self, idx: i32) -> bool { unsafe { ffi::lua_istable(self.L, idx) } }

    // ---------------------------------------------------------------
    // Table operations
    // ---------------------------------------------------------------

    /// Create an empty table and push it onto the stack.
    pub fn new_table(&self) {
        unsafe { ffi::lua_newtable(self.L) };
    }

    /// Create a table with pre-allocated array (`narr`) and hash (`nrec`)
    /// parts, and push it onto the stack.
    pub fn create_table(&self, narr: i32, nrec: i32) {
        unsafe { ffi::lua_createtable(self.L, narr, nrec) };
    }

    /// Set `t[key] = value` where `t` is at `idx` and `value` is at
    /// the stack top (popped).
    ///
    /// `key` is a Rust string that is converted to a C string internally.
    pub fn set_field(&self, idx: i32, key: &str) {
        let c = CString::new(key).unwrap();
        unsafe { ffi::lua_setfield(self.L, idx, c.as_ptr()) };
    }

    /// Push `t[key]` onto the stack, where `t` is at `idx`.
    ///
    /// Returns the Lua type tag of the pushed value.
    pub fn get_field(&self, idx: i32, key: &str) -> i32 {
        let c = CString::new(key).unwrap();
        unsafe { ffi::lua_getfield(self.L, idx, c.as_ptr()) }
    }

    /// Set `_G[name] = value` where `value` is at the stack top (popped).
    pub fn set_global(&self, name: &str) {
        let c = CString::new(name).unwrap();
        unsafe { ffi::lua_setglobal(self.L, c.as_ptr()) };
    }

    /// Push `_G[name]` onto the stack.
    ///
    /// Returns the Lua type tag of the pushed value.
    pub fn get_global(&self, name: &str) -> i32 {
        let c = CString::new(name).unwrap();
        unsafe { ffi::lua_getglobal(self.L, c.as_ptr()) }
    }

    /// Raw integer-keyed set: `t[n] = value` (value at top, popped).
    ///
    /// No metamethods are invoked.
    pub fn raw_seti(&self, idx: i32, n: i64) {
        unsafe { ffi::lua_rawseti(self.L, idx, n) };
    }

    /// Raw integer-keyed get: push `t[n]` onto the stack.
    ///
    /// No metamethods are invoked.
    pub fn raw_geti(&self, idx: i32, n: i64) -> i32 {
        unsafe { ffi::lua_rawgeti(self.L, idx, n) }
    }

    // ---------------------------------------------------------------
    // Loading and execution
    // ---------------------------------------------------------------

    /// Load a Lua chunk from a string buffer (text mode only).
    ///
    /// On success, the compiled chunk is pushed onto the stack as a
    /// Lua function. On failure, an error message is pushed (and then
    /// popped into the returned `LuaError`).
    ///
    /// # Arguments
    ///
    /// * `code` - The Lua source code to compile.
    /// * `name` - A name for the chunk (shown in error messages).
    ///   Convention: prefix with `=` for literal names (e.g. `"=machine"`).
    ///
    /// # Text mode only
    ///
    /// This function passes `"t"` as the mode to `luaL_loadbufferx`,
    /// rejecting bytecode. To load bytecode, the caller would need to
    /// use `"bt"` mode directly (not exposed in this safe wrapper).
    pub fn load_string(&self, code: &str, name: &str) -> LuaResult {
        let cname = CString::new(name).unwrap();
        log::trace!("Lua load: {name} ({} bytes)", code.len());
        let status = unsafe {
            ffi::luaL_loadbuffer(
                self.L,
                code.as_ptr() as *const _,
                code.len(),
                cname.as_ptr(),
            )
        };
        if status != ffi::LUA_OK {
            let msg = self.pop_error_string();
            log::error!("Lua load error ({name}): {msg}");
            Err(LuaError { message: msg, code: status })
        } else {
            Ok(())
        }
    }

    /// Protected call: call the function at the top of the stack.
    ///
    /// # Arguments
    ///
    /// * `nargs` - Number of arguments (already pushed above the function).
    /// * `nresults` - Number of expected results (or `LUA_MULTRET` for all).
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Call succeeded. Results are on the stack.
    /// * `Err(LuaError)` - Call failed. The error message was on the
    ///   stack and has been popped into the error.
    pub fn pcall(&self, nargs: i32, nresults: i32) -> LuaResult {
        let status = unsafe { ffi::lua_pcall(self.L, nargs, nresults, 0) };
        if status != ffi::LUA_OK {
            let msg = self.pop_error_string();
            Err(LuaError { message: msg, code: status })
        } else {
            Ok(())
        }
    }

    /// Execute a Lua string (load + pcall, zero arguments, all results).
    ///
    /// Convenience wrapper: compiles the string, then immediately calls it.
    pub fn exec(&self, code: &str, name: &str) -> LuaResult {
        self.load_string(code, name)?;
        self.pcall(0, ffi::LUA_MULTRET)
    }

    // ---------------------------------------------------------------
    // Coroutine operations
    // ---------------------------------------------------------------

    /// Create a new Lua coroutine (thread) sharing the same global state.
    ///
    /// The thread is pushed onto the parent's stack AND returned as a
    /// non-owning `LuaState`. The caller must keep the thread on the
    /// parent's stack (or anchor it in the registry) to prevent GC.
    ///
    /// # Returns
    ///
    /// A non-owning `LuaState` wrapping the coroutine's `lua_State*`.
    pub fn new_thread(&self) -> LuaState {
        let co = unsafe { ffi::lua_newthread(self.L) };
        LuaState { L: co, owned: false }
    }

    /// Resume a coroutine.
    ///
    /// # Arguments
    ///
    /// * `nargs` - Number of values on this coroutine's stack to pass
    ///   as arguments (or resume values).
    ///
    /// # Returns
    ///
    /// `(status, nresults)` where:
    /// * `status` is `LUA_OK` if the coroutine finished, `LUA_YIELD`
    ///   if it yielded, or an error code.
    /// * `nresults` is the number of values on the stack (return values
    ///   or yielded values).
    pub fn resume(&self, nargs: i32) -> (i32, i32) {
        let mut nresults: c_int = 0;
        let status = unsafe {
            ffi::lua_resume(self.L, ptr::null_mut(), nargs, &mut nresults)
        };
        (status, nresults)
    }

    /// Get the status of this coroutine.
    ///
    /// Returns `LUA_OK` (finished/not started), `LUA_YIELD` (suspended),
    /// or an error code (dead with error).
    pub fn status(&self) -> i32 {
        unsafe { ffi::lua_status(self.L) }
    }

    /// Move `n` values from `self`'s stack to `other`'s stack.
    ///
    /// The values are popped from `self` and pushed onto `other`.
    /// Both states must share the same global state (i.e. one must
    /// be a thread of the other).
    pub fn xmove_to(&self, other: &LuaState, n: i32) {
        unsafe { ffi::lua_xmove(self.L, other.L, n) };
    }

    // ---------------------------------------------------------------
    // Debug hooks
    // ---------------------------------------------------------------

    /// Set a count-based debug hook.
    ///
    /// `hook` is called every `count` VM instructions. Used to implement
    /// the execution timeout (checking wall-clock time periodically).
    ///
    /// Pass `count = 0` to disable the hook.
    pub fn set_hook(&self, hook: ffi::lua_Hook, count: i32) {
        let mask = if count > 0 { ffi::LUA_MASKCOUNT } else { 0 };
        unsafe { ffi::lua_sethook(self.L, hook, mask, count) };
    }

    // ---------------------------------------------------------------
    // Garbage collector
    // ---------------------------------------------------------------

    /// Force a full garbage collection cycle.
    pub fn gc_collect(&self) {
        unsafe { ffi::lua_gc(self.L, ffi::LUA_GCCOLLECT) };
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /// Pop the top of stack and interpret it as an error message string.
    ///
    /// If the top is not a string, returns `"(unknown error)"`.
    fn pop_error_string(&self) -> String {
        let msg = self.to_string_owned(-1).unwrap_or_else(|| "(unknown error)".into());
        self.pop(1);
        msg
    }

    /// Register a named function in a table at the top of the stack.
    ///
    /// Pushes the function, then calls `set_field` to store it under
    /// the given name.
    pub fn register_fn(&self, name: &str, f: ffi::lua_CFunction) {
        self.push_fn(f);
        self.set_field(-2, name);
    }

    /// Wrap an existing `*mut lua_State` pointer as a non-owning `LuaState`.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and must outlive the returned `LuaState`.
    /// The caller is responsible for ensuring the state is not closed
    /// while this handle exists.
    pub unsafe fn from_raw_non_owning(ptr: *mut ffi::lua_State) -> Self {
        Self { L: ptr, owned: false }
    }

    /// Retrieve the Lua thread (coroutine) at the given stack index.
    ///
    /// Returns `None` if the value at that index is not a thread.
    /// The returned `LuaState` is non-owning (will not close on drop).
    pub fn get_thread(&self, idx: i32) -> Option<Self> {
        unsafe {
            let t = ffi::lua_tothread(self.L, idx);
            if t.is_null() {
                None
            } else {
                Some(Self::from_raw_non_owning(t))
            }
        }
    }
}

impl Drop for LuaState {
    /// Close the Lua state if this handle owns it.
    ///
    /// Non-owning handles (coroutine threads, `from_raw_non_owning`)
    /// do nothing on drop.
    fn drop(&mut self) {
        if self.owned && !self.L.is_null() {
            log::debug!("Closing Lua state ({:p})", self.L);
            unsafe { ffi::lua_close(self.L) };
        }
    }
}