//! # Signal queue
//!
//! Signals are the primary IPC mechanism between the host (Rust) and
//! guest (Lua) code.  They are stored in a bounded FIFO queue.
//!
//! Each signal has a name (e.g. `"key_down"`, `"component_added"`) and
//! zero or more arguments that can be booleans, numbers, or strings.

use std::collections::VecDeque;

/// A single signal argument.
#[derive(Debug, Clone)]
pub enum SignalArg {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
}

/// A signal waiting to be delivered to the Lua VM.
#[derive(Debug, Clone)]
pub struct Signal {
    pub name: String,
    pub args: Vec<SignalArg>,
}

impl Signal {
    /// Create a new signal with no arguments.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), args: Vec::new() }
    }

    /// Builder: append a string argument.
    pub fn with_string(mut self, s: String) -> Self {
        self.args.push(SignalArg::Str(s));
        self
    }

    /// Builder: append an integer argument.
    pub fn with_int(mut self, n: i64) -> Self {
        self.args.push(SignalArg::Int(n));
        self
    }

    /// Builder: append a float argument.
    pub fn with_float(mut self, n: f64) -> Self {
        self.args.push(SignalArg::Float(n));
        self
    }

    /// Builder: append a bool argument.
    pub fn with_bool(mut self, b: bool) -> Self {
        self.args.push(SignalArg::Bool(b));
        self
    }
}

/// Bounded FIFO signal queue.
///
/// When the queue is full, [`push`] returns `false` and the signal is
/// discarded / matching the behaviour described in `Machine.signal()`.
#[derive(Debug)]
pub struct SignalQueue {
    queue: VecDeque<Signal>,
    capacity: usize,
}

impl SignalQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Push a signal.  Returns `false` if the queue is full.
    #[inline]
    pub fn push(&mut self, signal: Signal) -> bool {
        if self.queue.len() >= self.capacity {
            return false;
        }
        self.queue.push_back(signal);
        true
    }

    /// Pop the oldest signal.
    #[inline]
    pub fn pop(&mut self) -> Option<Signal> {
        self.queue.pop_front()
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