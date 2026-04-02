//! # Logging subsystem
//!
//! Initialises structured logging via `env_logger` with module-level
//! filtering. Default level is INFO; override with `RUST_LOG` env var.
//!
//! ## Typical usage
//!
//! ```text
//! RUST_LOG=debug cargo run                       # everything at debug
//! RUST_LOG=uniemu::render=trace cargo run        # trace-level Vulkan
//! RUST_LOG=uniemu::lua=debug,uniemu::render=info # per-module
//! ```
//!
//! ## Log level conventions
//!
//! | Level | Usage                                                      |
//! |-------|------------------------------------------------------------|
//! | ERROR | Unrecoverable failures (crash, Vulkan device lost)         |
//! | WARN  | Recoverable issues (fallback path, missing asset)          |
//! | INFO  | Lifecycle events (init, shutdown, mode switch, state)      |
//! | DEBUG | Per-operation detail (component invoke, frame stats)       |
//! | TRACE | Per-frame/per-cell granularity (SSBO upload, every signal) |

use std::io::Write;

/// Initialise the global logger. Call once at the start of `main()`.
///
/// Format: `[LEVEL][module] message`
///
/// If `RUST_LOG` is not set, defaults to `info`.
pub fn init() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format(|buf, record| {
        let level = record.level();
        let module = record.module_path().unwrap_or("?");
        // Strip crate name prefix for brevity
        let short = module
            .strip_prefix("uniemu::")
            .unwrap_or(module);
        writeln!(
            buf,
            "[{level:5}][{short}] {}",
            record.args()
        )
    })
    .init();

    log::info!("Logger initialised (RUST_LOG={:?})",
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()));
}