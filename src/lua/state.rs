//! # Safe Lua state wrapper
//!
//! Provides a RAII handle around `lua_State*` with safe push/get methods.
//! Not all operations can be fully safe (Lua's API is inherently stateful),
//! but we encapsulate the `unsafe` blocks and enforce basic invariants.

use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::ptr;

use super::ffi;

/// Wrapper around a Lua 5.4 state.
///
/// Owns the state and closes it on drop. Provides safe methods for
/// common operations.
pub struct LuaState {
    /// Raw pointer to the main Lua state.
    L: *mut ffi::lua_State,
    /// Whether we own this state (and should close it on drop).
    owned: bool,
}

// LuaState is Send but NOT Sync / only one thread may use it at a time.
unsafe impl Send for LuaState {}

/// Errors from Lua operations.
#[derive(Debug)]
pub struct LuaError {
    pub message: String,
    pub code: c_int,
}

impl std::fmt::Display for LuaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lua error ({}): {}", self.code, self.message)
    }
}
impl std::error::Error for LuaError {}

pub type LuaResult<T = ()> = Result<T, LuaError>;

impl LuaState {
    /// Create a new Lua state with all standard libraries loaded.
    pub fn new() -> Option<Self> {
        let L = unsafe { ffi::luaL_newstate() };
        if L.is_null() {
            return None;
        }
        unsafe { ffi::luaL_openlibs(L) };
        Some(Self { L, owned: true })
    }

    /// Raw pointer, for passing to FFI functions.
    #[inline]
    pub fn ptr(&self) -> *mut ffi::lua_State { self.L }

    // ── Stack info ──────────────────────────────────────────────────────

    /// Number of values on the stack.
    #[inline]
    pub fn top(&self) -> i32 {
        unsafe { ffi::lua_gettop(self.L) }
    }

    /// Set the stack top (pops or pushes nils).
    #[inline]
    pub fn set_top(&self, idx: i32) {
        unsafe { ffi::lua_settop(self.L, idx) };
    }

    /// Pop `n` values.
    #[inline]
    pub fn pop(&self, n: i32) {
        unsafe { ffi::lua_pop(self.L, n) };
    }

    /// Get the Lua type of the value at `idx`.
    #[inline]
    pub fn type_of(&self, idx: i32) -> i32 {
        unsafe { ffi::lua_type(self.L, idx) }
    }

    // ── Push values ─────────────────────────────────────────────────────

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

    /// Push an integer.
    #[inline]
    pub fn push_integer(&self, v: i64) {
        unsafe { ffi::lua_pushinteger(self.L, v) };
    }

    /// Push a float.
    #[inline]
    pub fn push_number(&self, v: f64) {
        unsafe { ffi::lua_pushnumber(self.L, v) };
    }

    /// Push a string (copies the data into Lua's heap).
    pub fn push_string(&self, s: &str) {
        unsafe {
            ffi::lua_pushlstring(self.L, s.as_ptr() as *const _, s.len());
        }
    }

    /// Push a byte slice as a Lua string.
    pub fn push_bytes(&self, b: &[u8]) {
        unsafe {
            ffi::lua_pushlstring(self.L, b.as_ptr() as *const _, b.len());
        }
    }

    /// Push a C function.
    pub fn push_fn(&self, f: ffi::lua_CFunction) {
        unsafe { ffi::lua_pushcfunction(self.L, f) };
    }

    /// Push a light userdata pointer.
    pub fn push_light_userdata(&self, p: *mut std::ffi::c_void) {
        unsafe { ffi::lua_pushlightuserdata(self.L, p) };
    }

    /// Duplicate the value at `idx`.
    #[inline]
    pub fn push_value(&self, idx: i32) {
        unsafe { ffi::lua_pushvalue(self.L, idx) };
    }

    // ── Read values ─────────────────────────────────────────────────────

    /// Read a boolean at `idx`.
    #[inline]
    pub fn to_bool(&self, idx: i32) -> bool {
        unsafe { ffi::lua_toboolean(self.L, idx) != 0 }
    }

    /// Read a number at `idx`.
    #[inline]
    pub fn to_number(&self, idx: i32) -> f64 {
        unsafe { ffi::lua_tonumber(self.L, idx) }
    }

    /// Read an integer at `idx`.
    #[inline]
    pub fn to_integer(&self, idx: i32) -> i64 {
        unsafe { ffi::lua_tointeger(self.L, idx) }
    }

    /// Read a string at `idx`. Returns `None` if not a string.
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

    /// Read a string at `idx` as owned `String`. Returns `None` if not a string.
    pub fn to_string_owned(&self, idx: i32) -> Option<String> {
        self.to_str(idx).map(|s| s.to_owned())
    }

    /// Read userdata at `idx`.
    pub fn to_userdata(&self, idx: i32) -> *mut std::ffi::c_void {
        unsafe { ffi::lua_touserdata(self.L, idx) }
    }

    // ── Type checks ─────────────────────────────────────────────────────

    #[inline] pub fn is_nil(&self, idx: i32) -> bool { unsafe { ffi::lua_isnil(self.L, idx) } }
    #[inline] pub fn is_bool(&self, idx: i32) -> bool { unsafe { ffi::lua_isboolean(self.L, idx) } }
    #[inline] pub fn is_number(&self, idx: i32) -> bool { unsafe { ffi::lua_isnumber(self.L, idx) != 0 } }
    #[inline] pub fn is_string(&self, idx: i32) -> bool { unsafe { ffi::lua_isstring(self.L, idx) != 0 } }
    #[inline] pub fn is_function(&self, idx: i32) -> bool { unsafe { ffi::lua_isfunction(self.L, idx) } }
    #[inline] pub fn is_table(&self, idx: i32) -> bool { unsafe { ffi::lua_istable(self.L, idx) } }

    // ── Table operations ────────────────────────────────────────────────

    /// Create an empty table and push it.
    pub fn new_table(&self) {
        unsafe { ffi::lua_newtable(self.L) };
    }

    /// Create a table with pre-allocated space.
    pub fn create_table(&self, narr: i32, nrec: i32) {
        unsafe { ffi::lua_createtable(self.L, narr, nrec) };
    }

    /// `t[key] = value` where `t` is at `idx` and `value` is at stack top.
    pub fn set_field(&self, idx: i32, key: &str) {
        let c = CString::new(key).unwrap();
        unsafe { ffi::lua_setfield(self.L, idx, c.as_ptr()) };
    }

    /// Push `t[key]` where `t` is at `idx`.
    pub fn get_field(&self, idx: i32, key: &str) -> i32 {
        let c = CString::new(key).unwrap();
        unsafe { ffi::lua_getfield(self.L, idx, c.as_ptr()) }
    }

    /// `_G[name] = value` (value at stack top, popped).
    pub fn set_global(&self, name: &str) {
        let c = CString::new(name).unwrap();
        unsafe { ffi::lua_setglobal(self.L, c.as_ptr()) };
    }

    /// Push `_G[name]`.
    pub fn get_global(&self, name: &str) -> i32 {
        let c = CString::new(name).unwrap();
        unsafe { ffi::lua_getglobal(self.L, c.as_ptr()) }
    }

    /// Raw integer-keyed set: `t[n] = value` (value at top, popped).
    pub fn raw_seti(&self, idx: i32, n: i64) {
        unsafe { ffi::lua_rawseti(self.L, idx, n) };
    }

    /// Raw integer-keyed get: push `t[n]`.
    pub fn raw_geti(&self, idx: i32, n: i64) -> i32 {
        unsafe { ffi::lua_rawgeti(self.L, idx, n) }
    }

    // ── Loading & execution ─────────────────────────────────────────────

    /// Load a Lua chunk from a string buffer (text mode only).
    ///
    /// On success, the compiled chunk is pushed as a function.
    pub fn load_string(&self, code: &str, name: &str) -> LuaResult {
        let cname = CString::new(name).unwrap();
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
            Err(LuaError { message: msg, code: status })
        } else {
            Ok(())
        }
    }

    /// Protected call: call the function at the top of the stack.
    ///
    /// Pushes `nresults` results (or an error message on failure).
    pub fn pcall(&self, nargs: i32, nresults: i32) -> LuaResult {
        let status = unsafe { ffi::lua_pcall(self.L, nargs, nresults, 0) };
        if status != ffi::LUA_OK {
            let msg = self.pop_error_string();
            Err(LuaError { message: msg, code: status })
        } else {
            Ok(())
        }
    }

    /// Execute a Lua string (load + pcall).
    pub fn exec(&self, code: &str, name: &str) -> LuaResult {
        self.load_string(code, name)?;
        self.pcall(0, ffi::LUA_MULTRET)
    }

    // ── Coroutine operations ────────────────────────────────────────────

    /// Create a new Lua thread (coroutine) sharing the same state.
    ///
    /// The thread is pushed onto the parent stack and also returned
    /// as a non-owning `LuaState`.
    pub fn new_thread(&self) -> LuaState {
        let co = unsafe { ffi::lua_newthread(self.L) };
        LuaState { L: co, owned: false }
    }

    /// Resume a coroutine.
    ///
    /// Returns `(status, nresults)`. Status is `LUA_OK` if finished
    /// or `LUA_YIELD` if yielded.
    pub fn resume(&self, nargs: i32) -> (i32, i32) {
        let mut nresults: c_int = 0;
        let status = unsafe {
            ffi::lua_resume(self.L, ptr::null_mut(), nargs, &mut nresults)
        };
        (status, nresults)
    }

    /// Get the status of a coroutine.
    pub fn status(&self) -> i32 {
        unsafe { ffi::lua_status(self.L) }
    }

    /// Move `n` values from `self` to `other`.
    pub fn xmove_to(&self, other: &LuaState, n: i32) {
        unsafe { ffi::lua_xmove(self.L, other.L, n) };
    }

    // ── Debug hooks ─────────────────────────────────────────────────────

    /// Set a count-based debug hook on this state.
    ///
    /// `hook` is called every `count` instructions. Pass `count=0` to disable.
    pub fn set_hook(&self, hook: ffi::lua_Hook, count: i32) {
        let mask = if count > 0 { ffi::LUA_MASKCOUNT } else { 0 };
        unsafe { ffi::lua_sethook(self.L, hook, mask, count) };
    }

    // ── GC ──────────────────────────────────────────────────────────────

    /// Force a full garbage collection cycle.
    pub fn gc_collect(&self) {
        unsafe { ffi::lua_gc(self.L, ffi::LUA_GCCOLLECT) };
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Pop the top of stack and interpret it as an error string.
    fn pop_error_string(&self) -> String {
        let msg = self.to_string_owned(-1).unwrap_or_else(|| "(unknown error)".into());
        self.pop(1);
        msg
    }

    /// Register a named function in a table at the top of the stack.
    pub fn register_fn(&self, name: &str, f: ffi::lua_CFunction) {
        self.push_fn(f);
        self.set_field(-2, name);
    }
        pub unsafe fn from_raw_non_owning(ptr: *mut ffi::lua_State) -> Self {
        Self { L: ptr, owned: false }
    }

    /// Retrieve the Lua thread (coroutine) at the given stack index.
    ///
    /// Returns `None` if the value at that index is not a thread.
    /// The returned `LuaState` is non-owning.
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
    fn drop(&mut self) {
        if self.owned && !self.L.is_null() {
            unsafe { ffi::lua_close(self.L) };
        }
    }
}