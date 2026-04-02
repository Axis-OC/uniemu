//! # Raw Lua 5.4 C API bindings
//!
//! This module provides the minimal set of Lua C API function
//! declarations needed by the emulator. These are linked against
//! `liblua54.a`, which is built from the Lua 5.4 source code by
//! the `build.rs` build script.
//!
//! ## Why raw bindings?
//!
//! Existing Lua binding crates (rlua, mlua, hlua) add significant
//! abstraction overhead and do not easily support the coroutine-based
//! execution model that OpenComputers requires (where machine.lua
//! runs as a coroutine that yields back to the host). By using raw
//! FFI, we have full control over the Lua stack, coroutine lifecycle,
//! debug hooks, and memory management.
//!
//! ## Safety contract
//!
//! All functions in this module are `unsafe` because they require:
//!
//! * A valid, non-null `*mut lua_State` pointer.
//! * Correct argument counts on the Lua stack (the C API does not
//!   bounds-check in release builds).
//! * Correct pairing of push/pop operations to avoid stack overflow
//!   or underflow.
//! * Strings pushed via `lua_pushlstring` must have valid lengths.
//!
//! The safe wrapper in [`state`](super::state) encapsulates most of
//! these invariants.
//!
//! ## Naming conventions
//!
//! Function names match the Lua C API exactly (e.g. `lua_pushinteger`,
//! `luaL_loadbufferx`). Type aliases also match (`lua_Number`,
//! `lua_Integer`, `lua_CFunction`).
//!
//! ## What is NOT included
//!
//! This is a minimal binding. Notable omissions:
//! * `luaL_dofile` / `luaL_dostring` (we use load+pcall explicitly)
//! * `lua_upvalueindex` (not needed for our closure-free callbacks)
//! * `luaL_newmetatable` / `luaL_setmetatable` (we use raw metatable ops)
//! * `lua_dump` (bytecode dumping is not needed)
//! * Coroutine transfer functions beyond `lua_resume`/`lua_yieldk`
#![allow(warnings)]
#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

/// Opaque Lua state handle.
///
/// This enum has no variants and is only used as a pointer target
/// (`*mut lua_State`). It is never constructed on the Rust side;
/// the Lua C library manages its internal layout.
///
/// All Lua API functions take a `*mut lua_State` as their first argument.
pub enum lua_State {}

/// Lua number type.
///
/// In standard Lua 5.4, this is `f64` (double). The Lua configure
/// step in `build.rs` does not override this.
pub type lua_Number = f64;

/// Lua integer type.
///
/// In standard Lua 5.4 on 64-bit platforms, this is `i64`.
/// On 32-bit platforms it would be `i32`, but this emulator targets
/// 64-bit only.
pub type lua_Integer = i64;

/// C function callable from Lua.
///
/// Signature: takes a `*mut lua_State`, returns the number of return
/// values pushed onto the stack.
///
/// # Safety
///
/// The function must:
/// * Only access the Lua stack through the provided state pointer.
/// * Return a non-negative integer equal to the number of values
///   it pushed onto the stack.
/// * Not longjmp past Rust stack frames (Lua errors from C functions
///   are handled via `lua_error` / `luaL_error`, which internally
///   do a longjmp in standard Lua).
pub type lua_CFunction = unsafe extern "C" fn(*mut lua_State) -> c_int;

/// Debug hook function signature.
///
/// Called by the Lua VM at configured hook points (every N instructions,
/// on function call/return, on each line). Used to implement execution
/// timeouts.
pub type lua_Hook = unsafe extern "C" fn(*mut lua_State, *mut lua_Debug);

// -----------------------------------------------------------------------
// Type tags (returned by lua_type)
// -----------------------------------------------------------------------

/// Type tag for nil values.
pub const LUA_TNIL: c_int = 0;
/// Type tag for boolean values.
pub const LUA_TBOOLEAN: c_int = 1;
/// Type tag for light userdata (raw pointers).
pub const LUA_TLIGHTUSERDATA: c_int = 2;
/// Type tag for numbers (both integer and float sub-types).
pub const LUA_TNUMBER: c_int = 3;
/// Type tag for strings (including binary data).
pub const LUA_TSTRING: c_int = 4;
/// Type tag for tables.
pub const LUA_TTABLE: c_int = 5;
/// Type tag for functions (Lua closures and C functions).
pub const LUA_TFUNCTION: c_int = 6;
/// Type tag for full userdata (heap-allocated, GC-managed blocks).
pub const LUA_TUSERDATA: c_int = 7;
/// Type tag for threads (coroutines).
pub const LUA_TTHREAD: c_int = 8;

// -----------------------------------------------------------------------
// Status codes (returned by lua_pcall, lua_resume, etc.)
// -----------------------------------------------------------------------

/// No errors.
pub const LUA_OK: c_int = 0;
/// Coroutine yielded.
pub const LUA_YIELD: c_int = 1;
/// Runtime error.
pub const LUA_ERRRUN: c_int = 2;
/// Syntax error during compilation.
pub const LUA_ERRSYNTAX: c_int = 3;
/// Memory allocation error.
pub const LUA_ERRMEM: c_int = 4;
/// Error in the error handler function.
pub const LUA_ERRERR: c_int = 5;

// -----------------------------------------------------------------------
// Special stack indices
// -----------------------------------------------------------------------

/// Pseudo-index for the Lua registry table.
///
/// The registry is a global table accessible from C code but not from
/// Lua code. We use it to stash the `EmulatorState*` pointer so that
/// C callbacks can retrieve it.
pub const LUA_REGISTRYINDEX: c_int = -1_001_000;

// -----------------------------------------------------------------------
// Hook masks (bitwise OR for lua_sethook)
// -----------------------------------------------------------------------

/// Hook called on every function call.
pub const LUA_MASKCALL: c_int = 1 << 0;
/// Hook called on every function return.
pub const LUA_MASKRET: c_int = 1 << 1;
/// Hook called on every new source line.
pub const LUA_MASKLINE: c_int = 1 << 2;
/// Hook called every N VM instructions.
pub const LUA_MASKCOUNT: c_int = 1 << 3;

// -----------------------------------------------------------------------
// GC control constants
// -----------------------------------------------------------------------

/// Perform a full garbage collection cycle.
pub const LUA_GCCOLLECT: c_int = 2;
/// Stop the garbage collector.
pub const LUA_GCSTOP: c_int = 0;
/// Restart the garbage collector.
pub const LUA_GCRESTART: c_int = 1;

// -----------------------------------------------------------------------
// Multi-return sentinel
// -----------------------------------------------------------------------

/// When passed as `nresults` to `lua_pcall` or `lua_call`, means
/// "return all results".
pub const LUA_MULTRET: c_int = -1;

// -----------------------------------------------------------------------
// Debug info structure
// -----------------------------------------------------------------------

/// Debug information structure.
///
/// Passed to hook functions by Lua. We treat it as opaque (never read
/// its fields). The size is over-allocated to 256 bytes to be safe
/// across Lua builds with different struct sizes.
#[repr(C)]
pub struct lua_Debug {
    _private: [u8; 256],
}

// -----------------------------------------------------------------------
// Core state management
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Create a new, independent Lua state.
    ///
    /// Returns `NULL` on memory allocation failure.
    /// The state must be closed with `lua_close` when no longer needed.
    pub fn luaL_newstate() -> *mut lua_State;

    /// Close a Lua state, freeing all associated memory.
    ///
    /// After this call, the `lua_State` pointer is invalid.
    pub fn lua_close(L: *mut lua_State);

    /// Open all standard Lua libraries (string, table, math, io, etc.).
    ///
    /// Must be called after `luaL_newstate` if standard library
    /// functions are needed (which they always are for us).
    pub fn luaL_openlibs(L: *mut lua_State);
}

// -----------------------------------------------------------------------
// Stack manipulation
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Return the number of elements on the stack (= index of the top).
    pub fn lua_gettop(L: *mut lua_State) -> c_int;

    /// Set the stack top to `idx`.
    ///
    /// If `idx` is less than the current top, values are popped.
    /// If greater, nils are pushed to fill the gap.
    pub fn lua_settop(L: *mut lua_State, idx: c_int);

    /// Push a copy of the value at `idx` onto the top.
    pub fn lua_pushvalue(L: *mut lua_State, idx: c_int);

    /// Rotate the `n` elements between `idx` and the top.
    pub fn lua_rotate(L: *mut lua_State, idx: c_int, n: c_int);

    /// Copy the value at `from` to `to` without removing either.
    pub fn lua_copy(L: *mut lua_State, from: c_int, to: c_int);

    /// Convert a relative stack index to an absolute one.
    pub fn lua_absindex(L: *mut lua_State, idx: c_int) -> c_int;

    /// Ensure the stack has room for `n` more elements.
    ///
    /// Returns 0 on failure (out of memory).
    pub fn lua_checkstack(L: *mut lua_State, n: c_int) -> c_int;

    /// Return the type tag of the value at `idx`.
    ///
    /// Returns one of `LUA_TNIL`, `LUA_TBOOLEAN`, etc.
    /// Returns `LUA_TNONE` (-1) for invalid indices.
    pub fn lua_type(L: *mut lua_State, idx: c_int) -> c_int;
}

/// Pop `n` values from the stack.
///
/// Convenience wrapper around `lua_settop(L, -n - 1)`.
#[inline]
pub unsafe fn lua_pop(L: *mut lua_State, n: c_int) {
    lua_settop(L, -n - 1);
}

// -----------------------------------------------------------------------
// Push values onto the stack
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Push a nil value.
    pub fn lua_pushnil(L: *mut lua_State);

    /// Push a boolean value (0 = false, non-zero = true).
    pub fn lua_pushboolean(L: *mut lua_State, b: c_int);

    /// Push an integer value.
    pub fn lua_pushinteger(L: *mut lua_State, n: lua_Integer);

    /// Push a floating-point number value.
    pub fn lua_pushnumber(L: *mut lua_State, n: lua_Number);

    /// Push a string of `len` bytes (may contain embedded NULs).
    ///
    /// Lua makes an internal copy of the data.
    /// Returns a pointer to the internal copy (valid until the string
    /// is garbage collected).
    pub fn lua_pushlstring(L: *mut lua_State, s: *const c_char, len: usize) -> *const c_char;

    /// Push a NUL-terminated C string.
    ///
    /// Lua makes an internal copy.
    pub fn lua_pushstring(L: *mut lua_State, s: *const c_char) -> *const c_char;

    /// Push a C closure with `n` upvalues.
    ///
    /// The upvalues must already be on the stack (they are popped).
    /// For n=0, this is equivalent to `lua_pushcfunction`.
    pub fn lua_pushcclosure(L: *mut lua_State, f: lua_CFunction, n: c_int);

    /// Push a light userdata (raw pointer, not GC-managed).
    pub fn lua_pushlightuserdata(L: *mut lua_State, p: *mut c_void);
}

/// Push a C function with no upvalues.
///
/// Convenience wrapper around `lua_pushcclosure(L, f, 0)`.
#[inline]
pub unsafe fn lua_pushcfunction(L: *mut lua_State, f: lua_CFunction) {
    lua_pushcclosure(L, f, 0);
}

// -----------------------------------------------------------------------
// Read values from the stack
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Read a boolean value at `idx` (0 = false, 1 = true).
    ///
    /// Non-boolean types are coerced: nil and false -> 0, everything else -> 1.
    pub fn lua_toboolean(L: *mut lua_State, idx: c_int) -> c_int;

    /// Read an integer at `idx`, with optional success flag.
    ///
    /// If `isnum` is non-null, `*isnum` is set to 1 if the value was
    /// an integer (or convertible to one), 0 otherwise.
    pub fn lua_tointegerx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Integer;

    /// Read a number (float) at `idx`, with optional success flag.
    pub fn lua_tonumberx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Number;

    /// Read a string at `idx`, with length output.
    ///
    /// If `len` is non-null, `*len` is set to the string length.
    /// Returns a pointer to the internal string data (valid while the
    /// string is on the stack).
    /// Returns NULL if the value is not a string or number.
    pub fn lua_tolstring(L: *mut lua_State, idx: c_int, len: *mut usize) -> *const c_char;

    /// Read a userdata pointer at `idx`.
    ///
    /// Returns the raw pointer for both light and full userdata.
    /// Returns NULL if the value is not userdata.
    pub fn lua_touserdata(L: *mut lua_State, idx: c_int) -> *mut c_void;

    /// Read a thread (coroutine) at `idx`.
    ///
    /// Returns a `*mut lua_State` for the thread.
    /// Returns NULL if the value is not a thread.
    pub fn lua_tothread(L: *mut lua_State, idx: c_int) -> *mut lua_State;
}

/// Read a NUL-terminated string at `idx` (no length output).
#[inline]
pub unsafe fn lua_tostring(L: *mut lua_State, idx: c_int) -> *const c_char {
    lua_tolstring(L, idx, std::ptr::null_mut())
}

/// Read a number at `idx` (no success flag).
#[inline]
pub unsafe fn lua_tonumber(L: *mut lua_State, idx: c_int) -> lua_Number {
    lua_tonumberx(L, idx, std::ptr::null_mut())
}

/// Read an integer at `idx` (no success flag).
#[inline]
pub unsafe fn lua_tointeger(L: *mut lua_State, idx: c_int) -> lua_Integer {
    lua_tointegerx(L, idx, std::ptr::null_mut())
}

// -----------------------------------------------------------------------
// Type checking
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Check if the value at `idx` is an integer (not just a number).
    pub fn lua_isinteger(L: *mut lua_State, idx: c_int) -> c_int;

    /// Check if the value at `idx` is a number (or convertible to one).
    pub fn lua_isnumber(L: *mut lua_State, idx: c_int) -> c_int;

    /// Check if the value at `idx` is a string (or convertible to one).
    pub fn lua_isstring(L: *mut lua_State, idx: c_int) -> c_int;
}

/// Check if the value at `idx` is nil.
#[inline]
pub unsafe fn lua_isnil(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TNIL
}

/// Check if the value at `idx` is a boolean.
#[inline]
pub unsafe fn lua_isboolean(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TBOOLEAN
}

/// Check if the value at `idx` is a function (Lua or C).
#[inline]
pub unsafe fn lua_isfunction(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TFUNCTION
}

/// Check if the value at `idx` is a table.
#[inline]
pub unsafe fn lua_istable(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TTABLE
}

// -----------------------------------------------------------------------
// Table operations
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Create a new table with pre-allocated array and hash parts.
    pub fn lua_createtable(L: *mut lua_State, narr: c_int, nrec: c_int);

    /// Push `t[k]` where `t` is at `idx` and `k` is a C string.
    pub fn lua_getfield(L: *mut lua_State, idx: c_int, k: *const c_char) -> c_int;

    /// Set `t[k] = v` where `t` is at `idx`, `k` is a C string, and
    /// `v` is at the stack top (popped).
    pub fn lua_setfield(L: *mut lua_State, idx: c_int, k: *const c_char);

    /// Push `t[k]` where both `t` (at `idx`) and `k` (at top) are on the stack.
    pub fn lua_gettable(L: *mut lua_State, idx: c_int) -> c_int;

    /// Set `t[k] = v` where `t` is at `idx`, `k` is at top-1, `v` is at top.
    pub fn lua_settable(L: *mut lua_State, idx: c_int);

    /// Raw (no metamethod) version of `lua_gettable`.
    pub fn lua_rawget(L: *mut lua_State, idx: c_int) -> c_int;

    /// Raw (no metamethod) version of `lua_settable`.
    pub fn lua_rawset(L: *mut lua_State, idx: c_int);

    /// Push `t[n]` (integer key, raw access).
    pub fn lua_rawgeti(L: *mut lua_State, idx: c_int, n: lua_Integer) -> c_int;

    /// Set `t[n] = v` (integer key, raw access). `v` is at top, popped.
    pub fn lua_rawseti(L: *mut lua_State, idx: c_int, n: lua_Integer);

    /// Table iterator. Pops key, pushes next key-value pair.
    ///
    /// Returns 0 when no more elements.
    pub fn lua_next(L: *mut lua_State, idx: c_int) -> c_int;

    /// Return the raw length of a table or string (no metamethods).
    pub fn lua_rawlen(L: *mut lua_State, idx: c_int) -> usize;
}

/// Create an empty table and push it.
#[inline]
pub unsafe fn lua_newtable(L: *mut lua_State) {
    lua_createtable(L, 0, 0);
}

// -----------------------------------------------------------------------
// Global table
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Push `_G[name]`.
    pub fn lua_getglobal(L: *mut lua_State, name: *const c_char) -> c_int;

    /// Set `_G[name] = v` where `v` is at the stack top (popped).
    pub fn lua_setglobal(L: *mut lua_State, name: *const c_char);
}

// -----------------------------------------------------------------------
// Metatables
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Push the metatable of the value at `idx`.
    ///
    /// Returns 0 if the value has no metatable (pushes nothing).
    pub fn lua_getmetatable(L: *mut lua_State, idx: c_int) -> c_int;

    /// Set the metatable of the value at `idx` to the table at the top
    /// of the stack (popped).
    pub fn lua_setmetatable(L: *mut lua_State, idx: c_int) -> c_int;
}

// -----------------------------------------------------------------------
// Loading and calling
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Load a Lua chunk from a memory buffer.
    ///
    /// `mode` controls what is accepted:
    /// * `"t"` - text only (source code)
    /// * `"b"` - binary only (bytecode)
    /// * `"bt"` - both
    /// * NULL - both
    ///
    /// On success, pushes the compiled chunk as a function.
    /// On failure, pushes an error message.
    pub fn luaL_loadbufferx(
        L: *mut lua_State,
        buff: *const c_char,
        sz: usize,
        name: *const c_char,
        mode: *const c_char,
    ) -> c_int;

    /// Protected call with continuation support.
    ///
    /// `msgh` is the stack index of an error handler function (0 for none).
    /// `ctx` and `k` are for continuation (we pass 0 and None).
    pub fn lua_pcallk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        msgh: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    ) -> c_int;

    /// Unprotected call with continuation support.
    pub fn lua_callk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    );
}

/// Protected call (no continuation).
#[inline]
pub unsafe fn lua_pcall(L: *mut lua_State, nargs: c_int, nresults: c_int, msgh: c_int) -> c_int {
    lua_pcallk(L, nargs, nresults, msgh, 0, None)
}

/// Unprotected call (no continuation).
#[inline]
pub unsafe fn lua_call(L: *mut lua_State, nargs: c_int, nresults: c_int) {
    lua_callk(L, nargs, nresults, 0, None);
}

/// Load a Lua chunk from a buffer in text mode.
#[inline]
pub unsafe fn luaL_loadbuffer(
    L: *mut lua_State,
    buff: *const c_char,
    sz: usize,
    name: *const c_char,
) -> c_int {
    luaL_loadbufferx(L, buff, sz, name, b"t\0".as_ptr() as *const c_char)
}

// -----------------------------------------------------------------------
// Coroutines
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Create a new coroutine thread sharing the same global state.
    ///
    /// The new thread is pushed onto the stack of `L`.
    pub fn lua_newthread(L: *mut lua_State) -> *mut lua_State;

    /// Resume a coroutine.
    ///
    /// `from` is the thread that is resuming (NULL for the main thread).
    /// `nargs` values on the coroutine's stack are passed as arguments.
    /// `*nresults` is set to the number of values returned/yielded.
    ///
    /// Returns `LUA_OK` if the coroutine finished, `LUA_YIELD` if it
    /// yielded, or an error code.
    pub fn lua_resume(
        L: *mut lua_State,
        from: *mut lua_State,
        nargs: c_int,
        nresults: *mut c_int,
    ) -> c_int;

    /// Get the status of a coroutine (OK, YIELD, or error).
    pub fn lua_status(L: *mut lua_State) -> c_int;

    /// Yield from a C function with continuation support.
    ///
    /// `nresults` values on the stack are yielded.
    pub fn lua_yieldk(
        L: *mut lua_State,
        nresults: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    ) -> c_int;

    /// Move `n` values from one thread's stack to another.
    pub fn lua_xmove(from: *mut lua_State, to: *mut lua_State, n: c_int);
}

// -----------------------------------------------------------------------
// Debug hooks
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Set a debug hook on a Lua state.
    ///
    /// `mask` is a bitwise OR of `LUA_MASKCALL`, `LUA_MASKRET`,
    /// `LUA_MASKLINE`, `LUA_MASKCOUNT`.
    /// `count` is the instruction count for `LUA_MASKCOUNT` hooks.
    pub fn lua_sethook(L: *mut lua_State, f: lua_Hook, mask: c_int, count: c_int) -> c_int;
}

// -----------------------------------------------------------------------
// Garbage collector
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Control the garbage collector.
    ///
    /// `what` is one of `LUA_GCCOLLECT`, `LUA_GCSTOP`, `LUA_GCRESTART`, etc.
    pub fn lua_gc(L: *mut lua_State, what: c_int, ...) -> c_int;
}

// -----------------------------------------------------------------------
// Auxiliary library
// -----------------------------------------------------------------------

unsafe extern "C" {
    /// Create a reference to the value at the top of the stack in table `t`.
    ///
    /// Returns an integer key that can be passed to `luaL_unref` later.
    pub fn luaL_ref(L: *mut lua_State, t: c_int) -> c_int;

    /// Release a reference created by `luaL_ref`.
    pub fn luaL_unref(L: *mut lua_State, t: c_int, r: c_int);

    /// Raise a Lua error with a formatted message.
    ///
    /// Does not return (longjmps).
    pub fn luaL_error(L: *mut lua_State, fmt: *const c_char, ...) -> c_int;

    /// Check that argument `arg` is an integer, or raise a Lua error.
    pub fn luaL_checkinteger(L: *mut lua_State, arg: c_int) -> lua_Integer;

    /// Check that argument `arg` is a number, or raise a Lua error.
    pub fn luaL_checknumber(L: *mut lua_State, arg: c_int) -> lua_Number;

    /// Check that argument `arg` is a string, or raise a Lua error.
    ///
    /// If `len` is non-null, `*len` is set to the string length.
    pub fn luaL_checklstring(
        L: *mut lua_State,
        arg: c_int,
        len: *mut usize,
    ) -> *const c_char;

    /// Get argument `arg` as an integer, or `def` if absent/nil.
    pub fn luaL_optinteger(L: *mut lua_State, arg: c_int, def: lua_Integer) -> lua_Integer;
}

/// Reference constant meaning "nil reference" (the referenced value is nil).
pub const LUA_REFNIL: c_int = -1;

/// Reference constant meaning "no reference" (invalid/freed reference).
pub const LUA_NOREF: c_int = -2;