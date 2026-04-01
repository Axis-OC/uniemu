//! # Lua VM integration
//!
//! Provides a safe-ish wrapper around the Lua 5.4 C API,
//! compiled from source via `build.rs`.
//!
//! Sub-modules:
//! - [`ffi`] / raw `extern "C"` bindings
//! - [`state`] / safe `LuaState` wrapper
//! - [`host`] / OpenComputers machine ↔ Lua bridge

pub mod ffi;
pub mod state;
pub mod host;