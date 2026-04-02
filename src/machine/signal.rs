//! # Signal queue
//!
//! Signals are the primary IPC mechanism between the host (Rust) and
//! the guest (Lua) code in OpenComputers. They are stored in a bounded
//! FIFO queue.
//!
//! ## Signal model
//!
//! Each signal has:
//! * A **name** (string): identifies the event type (e.g. `"key_down"`,
//!   `"component_added"`, `"redstone_changed"`).
//! * Zero or more **arguments**: typed values (nil, bool, int, float,
//!   string, or raw bytes) that provide event-specific data.
//!
//! ## Queue semantics
//!
//! * **Bounded**: The queue has a configurable maximum size (default 256).
//!   When full, new signals are silently discarded (matching OC behaviour).
//! * **FIFO**: Signals are delivered in the order they were pushed.
//! * **Non-blocking**: `push()` never blocks; it returns `false` if the
//!   queue is full.
//!
//! ## Delivery
//!
//! Signals are delivered to Lua by pushing them onto the coroutine's
//! stack before `lua_resume`. The kernel's `computer.pullSignal()`
//! function yields, and the host resumes with the signal's name and
//! arguments.
//!
//! ## Common signals
//!
//! ```text
//! Signal name          Produced by               Arguments
//! -------------------- ------------------------- --------------------------------
//! key_down             Keyboard                  addr, char, code
//! key_up               Keyboard                  addr, char, code
//! clipboard            Keyboard                  addr, line
//! component_added      Machine                   addr, type_name
//! component_removed    Machine                   addr, type_name
//! redstone_changed     Redstone I/O              addr, side, old, new
//! modem_message        Network card              addr, sender, port, ...
//! touch                Screen                    addr, x, y, button, player
//! ```

use std::collections::VecDeque;

/// A single signal argument.
///
/// Signals can carry heterogeneous arguments. This enum covers all
/// types that OC signals use.
#[derive(Debug, Clone)]
pub enum SignalArg {
    /// Lua `nil`.
    Nil,
    /// Lua boolean.
    Bool(bool),
    /// Lua integer (64-bit).
    Int(i64),
    /// Lua float (64-bit).
    Float(f64),
    /// Lua string (UTF-8 text).
    Str(String),
    /// Raw binary data (pushed as a Lua string, but may contain non-UTF-8 bytes).
    Bytes(Vec<u8>),
}

/// A signal waiting to be delivered to the Lua VM.
///
/// Signals are created by host code (component methods, the machine
/// itself, or user input handlers) and delivered to Lua by the kernel
/// stepping logic.
///
/// # Builder pattern
///
/// Signals are constructed using a builder pattern:
///
/// ```text
/// Signal::new("key_down")
///     .with_string(keyboard_addr)
///     .with_int(char_code as i64)
///     .with_int(key_code as i64)
/// ```
#[derive(Debug, Clone)]
pub struct Signal {
    /// The event name (e.g. `"key_down"`).
    pub name: String,

    /// Zero or more typed arguments.
    pub args: Vec<SignalArg>,
}

impl Signal {
    /// Create a new signal with no arguments.
    ///
    /// # Arguments
    ///
    /// * `name` - The signal name. Any type implementing `Into<String>`.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), args: Vec::new() }
    }

    /// Append a string argument (builder pattern).
    pub fn with_string(mut self, s: String) -> Self {
        self.args.push(SignalArg::Str(s));
        self
    }

    /// Append an integer argument (builder pattern).
    pub fn with_int(mut self, n: i64) -> Self {
        self.args.push(SignalArg::Int(n));
        self
    }

    /// Append a float argument (builder pattern).
    pub fn with_float(mut self, n: f64) -> Self {
        self.args.push(SignalArg::Float(n));
        self
    }

    /// Append a boolean argument (builder pattern).
    pub fn with_bool(mut self, b: bool) -> Self {
        self.args.push(SignalArg::Bool(b));
        self
    }
}

/// Bounded FIFO signal queue.
///
/// When the queue is full, [`push`](SignalQueue::push) returns `false`
/// and the signal is discarded. This matches the behaviour described
/// in `Machine.signal()` from the OC Scala source.
///
/// # Capacity
///
/// The capacity is set at construction time and cannot be changed.
/// Default: 256 (from `OcConfig::max_signal_queue_size`).
/// The internal `VecDeque` is pre-allocated to `min(capacity, 1024)`.
#[derive(Debug)]
pub struct SignalQueue {
    /// The underlying double-ended queue.
    queue: VecDeque<Signal>,

    /// Maximum number of signals the queue can hold.
    capacity: usize,
}

impl SignalQueue {
    /// Create a new queue with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Push a signal onto the queue.
    ///
    /// # Returns
    ///
    /// `true` if the signal was added.
    /// `false` if the queue was full (signal is discarded).
    #[inline]
    pub fn push(&mut self, signal: Signal) -> bool {
        if self.queue.len() >= self.capacity {
            log::trace!("Signal queue overflow: dropping '{}' (cap={})",
                signal.name, self.capacity);
            return false;
        }
        log::trace!("Signal pushed: '{}' (queue len={})", signal.name, self.queue.len() + 1);
        self.queue.push_back(signal);
        true
    }

    /// Pop the oldest signal from the queue.
    ///
    /// Returns `None` if the queue is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<Signal> {
        let sig = self.queue.pop_front();
        if let Some(ref s) = sig {
            log::trace!("Signal popped: '{}' (remaining={})", s.name, self.queue.len());
        }
        sig
    }
    
    /// Whether the queue has no pending signals.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of pending signals.
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Discard all pending signals.
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}