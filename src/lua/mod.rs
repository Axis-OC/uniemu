//! # Lua VM integration
//!
//! This module is the bridge between the Rust emulator and the Lua 5.4
//! virtual machine that runs the OpenComputers kernel (`machine.lua`)
//! and all guest programs on top of it.
//!
//! ## Architecture overview
//!
//! ```text
//! +-------------------+
//! | Guest Lua code    |  <-- OpenOS, user programs, BIOS
//! +-------------------+
//!         |
//!         | (runs inside a coroutine)
//!         v
//! +-------------------+
//! | machine.lua       |  <-- OC kernel, manages component calls,
//! |                   |      sandboxing, signal dispatch, timeouts
//! +-------------------+
//!         |
//!         | yield / resume
//!         v
//! +-------------------+
//! | host.rs           |  <-- Rust side: dispatches component.invoke,
//! |                   |      computer.*, system.*, unicode.* callbacks
//! +-------------------+
//!         |
//!         | calls into
//!         v
//! +-------------------+
//! | state.rs          |  <-- Safe(ish) wrapper around lua_State*
//! +-------------------+
//!         |
//!         | raw FFI
//!         v
//! +-------------------+
//! | ffi.rs            |  <-- extern "C" declarations for liblua54.a
//! +-------------------+
//!         |
//!         | linked by build.rs
//!         v
//! +-------------------+
//! | liblua54.a        |  <-- Lua 5.4 C source compiled from vendor/lua54/
//! +-------------------+
//! ```
//!
//! ## Sub-modules
//!
//! * [`ffi`] -- Raw `extern "C"` bindings to the Lua 5.4 C API.
//!   Every function is `unsafe`. Types, constants, and convenience
//!   wrappers (e.g. `lua_pop`, `lua_newtable`) live here.
//!
//! * [`state`] -- A RAII wrapper (`LuaState`) around `*mut lua_State`.
//!   Provides safe push/get/call/load methods. Owns the state and
//!   closes it on `Drop` (unless non-owning, e.g. coroutine handles).
//!
//! * [`host`] -- The OpenComputers machine <-> Lua bridge. Contains:
//!   - `EmulatorState`: all host-side data accessible from C callbacks
//!     (stashed as a light userdata in the Lua registry).
//!   - Every `computer.*`, `component.*`, `system.*`, `unicode.*`
//!     callback function.
//!   - `load_kernel()` / `step_kernel()` for the coroutine lifecycle.
//!   - `ExecResult` enum describing yield/sleep/shutdown/error outcomes.
//!
//! ## Execution model
//!
//! The OC kernel (`machine.lua`) runs as a Lua coroutine. The host
//! (Rust) drives it in a loop:
//!
//! ```text
//! 1. Create Lua state, open libs, register host APIs.
//! 2. Load machine.lua, wrap it in a coroutine (lua_newthread).
//! 3. Loop:
//!    a. Pop pending signal from the queue (if any).
//!    b. Push signal name + args onto the coroutine stack.
//!    c. lua_resume(coroutine, nargs) -> status, nresults.
//!    d. Interpret the yield:
//!       - Number -> sleep for that many seconds
//!       - Boolean -> shutdown (true = reboot)
//!       - Function -> synchronised call (execute, push result, resume again)
//!       - Nil / nothing -> sleep until signal
//!    e. If status == LUA_OK, the coroutine finished (machine halted).
//!    f. If status is an error, the machine crashed.
//! ```
//!
//! ## Why not mlua / rlua?
//!
//! These crates are excellent for embedding Lua in typical applications,
//! but they abstract away the coroutine stack manipulation that we need.
//! OC's kernel requires:
//!
//! * Resuming a coroutine with an arbitrary number of typed arguments.
//! * Interpreting yielded values as sleep durations, shutdown requests,
//!   or synchronised call functions.
//! * Stashing a raw pointer in the Lua registry for C callback access.
//! * Fine-grained control over `lua_sethook` for timeout enforcement.
//!
//! Raw FFI gives us all of this with zero abstraction overhead.

pub mod ffi;
pub mod state;
pub mod host;