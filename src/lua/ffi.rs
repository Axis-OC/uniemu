//! # Raw Lua 5.4 C API bindings
//!
//! Minimal set of Lua functions needed by the emulator.
//! These are linked against `liblua54.a` built by `build.rs`.
//!
//! # Safety
//!
//! All functions in this module are `unsafe` / they require a valid
//! `*mut lua_State` and correct argument counts on the stack.
#![allow(warnings)]
#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

/// Opaque Lua state handle.
pub enum lua_State {}

/// Lua number type (f64).
pub type lua_Number = f64;

/// Lua integer type (i64 on 64-bit).
pub type lua_Integer = i64;

/// C function callable from Lua.
pub type lua_CFunction = unsafe extern "C" fn(*mut lua_State) -> c_int;

/// Debug hook function.
pub type lua_Hook = unsafe extern "C" fn(*mut lua_State, *mut lua_Debug);

/// Lua type tags.
pub const LUA_TNIL: c_int = 0;
pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TLIGHTUSERDATA: c_int = 2;
pub const LUA_TNUMBER: c_int = 3;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;
pub const LUA_TFUNCTION: c_int = 6;
pub const LUA_TUSERDATA: c_int = 7;
pub const LUA_TTHREAD: c_int = 8;

/// Status codes.
pub const LUA_OK: c_int = 0;
pub const LUA_YIELD: c_int = 1;
pub const LUA_ERRRUN: c_int = 2;
pub const LUA_ERRSYNTAX: c_int = 3;
pub const LUA_ERRMEM: c_int = 4;
pub const LUA_ERRERR: c_int = 5;

/// Special stack indices.
pub const LUA_REGISTRYINDEX: c_int = -1_001_000;

/// Hook masks.
pub const LUA_MASKCALL: c_int = 1 << 0;
pub const LUA_MASKRET: c_int = 1 << 1;
pub const LUA_MASKLINE: c_int = 1 << 2;
pub const LUA_MASKCOUNT: c_int = 1 << 3;

/// GC options.
pub const LUA_GCCOLLECT: c_int = 2;
pub const LUA_GCSTOP: c_int = 0;
pub const LUA_GCRESTART: c_int = 1;

/// Multi-return sentinel.
pub const LUA_MULTRET: c_int = -1;

/// Debug info structure (opaque, we only pass pointers).
#[repr(C)]
pub struct lua_Debug {
    _private: [u8; 256], // oversized to be safe
}

// ── Core state management ───────────────────────────────────────────────

unsafe extern "C" {
    pub fn luaL_newstate() -> *mut lua_State;
    pub fn lua_close(L: *mut lua_State);
    pub fn luaL_openlibs(L: *mut lua_State);
}

// ── Stack manipulation ──────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_gettop(L: *mut lua_State) -> c_int;
    pub fn lua_settop(L: *mut lua_State, idx: c_int);
    pub fn lua_pushvalue(L: *mut lua_State, idx: c_int);
    pub fn lua_rotate(L: *mut lua_State, idx: c_int, n: c_int);
    pub fn lua_copy(L: *mut lua_State, from: c_int, to: c_int);
    pub fn lua_absindex(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_checkstack(L: *mut lua_State, n: c_int) -> c_int;
    pub fn lua_type(L: *mut lua_State, idx: c_int) -> c_int;
}

/// `lua_pop(L, n)` / pop n values from the stack.
#[inline]
pub unsafe fn lua_pop(L: *mut lua_State, n: c_int) {
    lua_settop(L, -n - 1);
}

// ── Push values ─────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_pushnil(L: *mut lua_State);
    pub fn lua_pushboolean(L: *mut lua_State, b: c_int);
    pub fn lua_pushinteger(L: *mut lua_State, n: lua_Integer);
    pub fn lua_pushnumber(L: *mut lua_State, n: lua_Number);
    pub fn lua_pushlstring(L: *mut lua_State, s: *const c_char, len: usize) -> *const c_char;
    pub fn lua_pushstring(L: *mut lua_State, s: *const c_char) -> *const c_char;
    pub fn lua_pushcclosure(L: *mut lua_State, f: lua_CFunction, n: c_int);
    pub fn lua_pushlightuserdata(L: *mut lua_State, p: *mut c_void);
}

/// Push a C function (zero upvalues).
#[inline]
pub unsafe fn lua_pushcfunction(L: *mut lua_State, f: lua_CFunction) {
    lua_pushcclosure(L, f, 0);
}

// ── Read values ─────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_toboolean(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_tointegerx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Integer;
    pub fn lua_tonumberx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Number;
    pub fn lua_tolstring(L: *mut lua_State, idx: c_int, len: *mut usize) -> *const c_char;
    pub fn lua_touserdata(L: *mut lua_State, idx: c_int) -> *mut c_void;
    pub fn lua_tothread(L: *mut lua_State, idx: c_int) -> *mut lua_State;
}

/// Convenience: `lua_tostring` without length output.
#[inline]
pub unsafe fn lua_tostring(L: *mut lua_State, idx: c_int) -> *const c_char {
    lua_tolstring(L, idx, std::ptr::null_mut())
}

/// Convenience: `lua_tonumber` without isnum output.
#[inline]
pub unsafe fn lua_tonumber(L: *mut lua_State, idx: c_int) -> lua_Number {
    lua_tonumberx(L, idx, std::ptr::null_mut())
}

/// Convenience: `lua_tointeger` without isnum output.
#[inline]
pub unsafe fn lua_tointeger(L: *mut lua_State, idx: c_int) -> lua_Integer {
    lua_tointegerx(L, idx, std::ptr::null_mut())
}

// ── Type checking ───────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_isinteger(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_isnumber(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_isstring(L: *mut lua_State, idx: c_int) -> c_int;
}

/// `lua_isnil(L, idx)`
#[inline]
pub unsafe fn lua_isnil(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TNIL
}

/// `lua_isboolean(L, idx)`
#[inline]
pub unsafe fn lua_isboolean(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TBOOLEAN
}

/// `lua_isfunction(L, idx)`
#[inline]
pub unsafe fn lua_isfunction(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TFUNCTION
}

/// `lua_istable(L, idx)`
#[inline]
pub unsafe fn lua_istable(L: *mut lua_State, idx: c_int) -> bool {
    lua_type(L, idx) == LUA_TTABLE
}

// ── Table operations ────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_createtable(L: *mut lua_State, narr: c_int, nrec: c_int);
    pub fn lua_getfield(L: *mut lua_State, idx: c_int, k: *const c_char) -> c_int;
    pub fn lua_setfield(L: *mut lua_State, idx: c_int, k: *const c_char);
    pub fn lua_gettable(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_settable(L: *mut lua_State, idx: c_int);
    pub fn lua_rawget(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_rawset(L: *mut lua_State, idx: c_int);
    pub fn lua_rawgeti(L: *mut lua_State, idx: c_int, n: lua_Integer) -> c_int;
    pub fn lua_rawseti(L: *mut lua_State, idx: c_int, n: lua_Integer);
    pub fn lua_next(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_rawlen(L: *mut lua_State, idx: c_int) -> usize;
}

/// `lua_newtable(L)` / creates an empty table.
#[inline]
pub unsafe fn lua_newtable(L: *mut lua_State) {
    lua_createtable(L, 0, 0);
}

// ── Global table ────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_getglobal(L: *mut lua_State, name: *const c_char) -> c_int;
    pub fn lua_setglobal(L: *mut lua_State, name: *const c_char);
}

// ── Metatables ──────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_getmetatable(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_setmetatable(L: *mut lua_State, idx: c_int) -> c_int;
}

// ── Loading & calling ───────────────────────────────────────────────────

unsafe extern "C" {
    pub fn luaL_loadbufferx(
        L: *mut lua_State,
        buff: *const c_char,
        sz: usize,
        name: *const c_char,
        mode: *const c_char,
    ) -> c_int;
    pub fn lua_pcallk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        msgh: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    ) -> c_int;
    pub fn lua_callk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    );
}

/// `lua_pcall(L, n, r, msgh)`.
#[inline]
pub unsafe fn lua_pcall(L: *mut lua_State, nargs: c_int, nresults: c_int, msgh: c_int) -> c_int {
    lua_pcallk(L, nargs, nresults, msgh, 0, None)
}

/// `lua_call(L, n, r)`.
#[inline]
pub unsafe fn lua_call(L: *mut lua_State, nargs: c_int, nresults: c_int) {
    lua_callk(L, nargs, nresults, 0, None);
}

/// `luaL_loadbuffer(L, s, sz, name)` (text mode).
#[inline]
pub unsafe fn luaL_loadbuffer(
    L: *mut lua_State,
    buff: *const c_char,
    sz: usize,
    name: *const c_char,
) -> c_int {
    luaL_loadbufferx(L, buff, sz, name, b"t\0".as_ptr() as *const c_char)
}

// ── Coroutines ──────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_newthread(L: *mut lua_State) -> *mut lua_State;
    pub fn lua_resume(
        L: *mut lua_State,
        from: *mut lua_State,
        nargs: c_int,
        nresults: *mut c_int,
    ) -> c_int;
    pub fn lua_status(L: *mut lua_State) -> c_int;
    pub fn lua_yieldk(
        L: *mut lua_State,
        nresults: c_int,
        ctx: isize,
        k: Option<unsafe extern "C" fn(*mut lua_State, c_int, isize) -> c_int>,
    ) -> c_int;
    pub fn lua_xmove(from: *mut lua_State, to: *mut lua_State, n: c_int);
}

// ── Debug / hooks ───────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_sethook(L: *mut lua_State, f: lua_Hook, mask: c_int, count: c_int) -> c_int;
}

// ── GC ──────────────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn lua_gc(L: *mut lua_State, what: c_int, ...) -> c_int;
}

// ── Aux library ─────────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn luaL_ref(L: *mut lua_State, t: c_int) -> c_int;
    pub fn luaL_unref(L: *mut lua_State, t: c_int, r: c_int);
    pub fn luaL_error(L: *mut lua_State, fmt: *const c_char, ...) -> c_int;
    pub fn luaL_checkinteger(L: *mut lua_State, arg: c_int) -> lua_Integer;
    pub fn luaL_checknumber(L: *mut lua_State, arg: c_int) -> lua_Number;
    pub fn luaL_checklstring(
        L: *mut lua_State,
        arg: c_int,
        len: *mut usize,
    ) -> *const c_char;
    pub fn luaL_optinteger(L: *mut lua_State, arg: c_int, def: lua_Integer) -> lua_Integer;
}

/// Nil reference constant.
pub const LUA_REFNIL: c_int = -1;
pub const LUA_NOREF: c_int = -2;