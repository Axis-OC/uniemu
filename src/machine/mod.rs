//! # Machine (Computer) state machine
//!
//! Implements the execution lifecycle from `Machine.scala`, including:
//! - State transitions (Stopped → Starting → Running → …)
//! - Signal queue
//! - Call budget per tick
//! - Component tracking (address → type name)
//! - Emulation mode selection (INDIRECT / DIRECT)
//!
//! The Lua VM integration is **not** part of this module / it provides the
//! hooks that a Lua host calls into.

pub mod signal;
pub mod persistence;

use std::collections::HashMap;
use signal::{Signal, SignalQueue};
use crate::config::OcConfig;

/// Execution states, matching `Machine.State` from `Machine.scala`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// No Lua state exists.  Power off.
    Stopped,
    /// First tick after `start()`, initialising kernel.
    Starting,
    /// Shutting down, then restarting.
    Restarting,
    /// Actively shutting down.
    Stopping,
    /// Paused (game paused, or explicit `computer.pause()`).
    Paused,
    /// Waiting for the host to execute a synchronised call on the main thread.
    SynchronizedCall,
    /// About to resume the Lua coroutine with the result of a sync call.
    SynchronizedReturn,
    /// Ready to resume immediately (signal available or zero-sleep).
    Yielded,
    /// Sleeping for a number of ticks (may be interrupted by signals).
    Sleeping,
    /// Lua coroutine is actively executing.
    Running,
}

/// Controls whether the GPU pipeline uses tick-accurate emulation or
/// bypasses budgets for maximum throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmulationMode {
    /// Faithful to OpenComputers: call budgets, tick-rate rendering,
    /// energy costs enforced.
    Indirect,
    /// Zero-budget GPU calls, no energy costs, immediate rendering.
    /// Lua still runs in the normal VM / only GPU throttling is removed.
    Direct,
}

/// The computer itself.
///
/// This struct owns the signal queue, component map, and timing state.
/// It does **not** own the Lua VM or any display buffers / those are
/// passed in via method parameters to keep ownership clear.
pub struct Machine {
    /// Current execution state.
    state: State,
    /// Emulation mode for GPU calls.
    pub mode: EmulationMode,

    // Components 
    /// `address → component_type_name` (e.g. `"gpu"`, `"filesystem"`).
    components: HashMap<String, String>,
    /// Maximum number of components this machine supports (CPU-dependent).
    pub max_components: usize,

    // Signals 
    signals: SignalQueue,

    // Call budget 
    /// Budget remaining for direct calls this tick.
    call_budget: f64,
    /// Maximum budget per tick (derived from CPU + memory tiers).
    max_call_budget: f64,

    // Timing 
    /// Ticks since the machine started (for `computer.uptime()`).
    uptime_ticks: u64,
    /// Ticks remaining to sleep before auto-resume.
    remain_idle: u32,
    /// Ticks remaining in an explicit pause.
    remain_pause: u32,

    // Energy 
    /// Local energy buffer.
    energy: f64,
    /// Maximum energy buffer size.
    energy_max: f64,
    /// Per-tick energy cost.
    cost_per_tick: f64,

    /// Config snapshot.
    config: OcConfig,
}

impl Machine {
    /// Create a new, stopped machine with the given configuration.
    pub fn new(config: OcConfig) -> Self {
        Self {
            state: State::Stopped,
            mode: EmulationMode::Indirect,
            components: HashMap::new(),
            max_components: 0,
            signals: SignalQueue::new(config.max_signal_queue_size),
            call_budget: 0.0,
            max_call_budget: 1.0,
            uptime_ticks: 0,
            remain_idle: 0,
            remain_pause: 0,
            energy: config.buffer_computer,
            energy_max: config.buffer_computer,
            cost_per_tick: config.computer_cost * config.tick_frequency as f64,
            config,
        }
    }

    // State queries 

    #[inline] pub fn state(&self) -> State { self.state }

    #[inline]
    pub fn is_running(&self) -> bool {
        !matches!(self.state, State::Stopped | State::Stopping)
    }

    #[inline]
    pub fn is_paused(&self) -> bool {
        self.state == State::Paused && self.remain_pause > 0
    }

    // Lifecycle 

    /// Attempt to start the machine. Returns `true` if state changed.
    ///
    /// Mirrors `Machine.start()` from `Machine.scala`.
    pub fn start(&mut self) -> bool {
        match self.state {
            State::Stopped => {
                if !self.config.ignore_power && self.energy < self.cost_per_tick {
                    log::warn!("Machine start failed: not enough energy ({:.1}/{:.1})",
                        self.energy, self.cost_per_tick);
                    return false;
                }
                if self.max_components == 0 {
                    log::warn!("Machine start failed: no CPU (max_components=0)");
                    return false;
                }
                self.state = State::Starting;
                self.uptime_ticks = 0;
                log::info!("Machine starting (mode={:?})", self.mode);
                true
            }
            State::Paused if self.remain_pause > 0 => {
                self.remain_pause = 0;
                true
            }
            State::Stopping => {
                self.state = State::Restarting;
                true
            }
            _ => false,
        }
    }

    /// Request the machine to stop.  Returns `true` if state changed.
    pub fn stop(&mut self) -> bool {
        match self.state {
            State::Stopped | State::Stopping => false,
            _ => {
                log::info!("Machine stopping (was {:?})", self.state);
                self.state = State::Stopping;
                true
            }
        }
    }

    /// Pause for the given number of seconds.
    pub fn pause(&mut self, seconds: f64) -> bool {
        let ticks = (seconds * 20.0).max(0.0) as u32;
        match self.state {
            State::Stopping | State::Stopped => false,
            State::Paused if ticks <= self.remain_pause => false,
            _ => {
                if self.state != State::Paused {
                    self.state = State::Paused;
                }
                self.remain_pause = ticks;
                true
            }
        }
    }

    /// Crash with an error message.  Triggers a stop.
    pub fn crash(&mut self, message: &str) -> bool {
        log::error!("Machine crash: {message}");
        self.stop()
    }


    // Per-tick update 

    /// Called once per server tick (50 ms in Minecraft).
    ///
    /// Handles state transitions, energy consumption, and budget reset.
    pub fn tick(&mut self) {
        if self.state == State::Stopped { return; }

        self.uptime_ticks += 1;

        if self.remain_idle > 0 {
            self.remain_idle -= 1;
        }

        // Reset call budget every tick.
        self.call_budget = self.max_call_budget;
        let old_state = self.state;
        // Energy consumption.
        if !self.config.ignore_power {
            let cost = match self.state {
                State::Sleeping if self.remain_idle > 0 && self.signals.is_empty() =>
                    self.cost_per_tick * self.config.sleep_cost_factor,
                State::Paused | State::Restarting | State::Stopping | State::Stopped =>
                    0.0,
                _ => self.cost_per_tick,
            };
            self.energy -= cost;
            if self.energy < 0.0 {
                self.crash("not enough energy");
                return;
            }
        }
        self.call_budget = self.max_call_budget;
        crate::profiler::budget(1.0); 

        // State transitions.
        match self.state {
            State::Starting => self.state = State::Yielded,
            State::Restarting => {
                self.state = State::Stopped;
                self.start();
            }
            State::Sleeping if self.remain_idle <= 0 || !self.signals.is_empty() => {
                self.state = State::Yielded;
            }
            State::Paused => {
                if self.remain_pause > 0 {
                    self.remain_pause -= 1;
                }
                // Stays paused until remain_pause hits 0.
            }
            _ => {}
        }
        
        if self.state != old_state {
            log::debug!("Machine state: {:?} -> {:?}", old_state, self.state);
        }
    }

    // Signal API 

    /// Push a signal into the queue.  Returns `false` if the queue is full.
    ///
    /// Mirrors `Machine.signal()` from `Machine.scala`.
    pub fn push_signal(&mut self, signal: Signal) -> bool {
        if matches!(self.state, State::Stopped | State::Stopping) {
            log::trace!("Signal '{}' dropped (machine {:?})", signal.name, self.state);
            return false;
        }
        let ok = self.signals.push(signal.clone());
        if !ok {
            log::warn!("Signal queue full, dropping '{}' (len={})",
                signal.name, self.signals.len());
        } else {
            log::trace!("Signal '{}' queued (len={})", signal.name, self.signals.len());
        }
        ok
    }
    /// Pop the next signal (FIFO).  Returns `None` if queue is empty.
    pub fn pop_signal(&mut self) -> Option<Signal> {
        self.signals.pop()
    }

    // Call budget 

    /// Consume call budget for a direct call.
    ///
    /// In **DIRECT** mode this is a no-op (budget is infinite).
    /// In **INDIRECT** mode, returns `Err(())` if budget is exhausted
    /// (equivalent to `LimitReachedException`).
    #[inline]
    pub fn consume_call_budget(&mut self, cost: f64) -> Result<(), ()> {
        if self.mode == EmulationMode::Direct {
            return Ok(());
        }
        let clamped = cost.max(0.0);
        if clamped > self.call_budget {
            crate::profiler::budget(0.0);
            return Err(());
        }
        self.call_budget -= clamped;

        let pct = if self.max_call_budget > 0.0 {
            (self.call_budget / self.max_call_budget) as f32
        } else { 1.0 };
        crate::profiler::budget(pct);

        Ok(())
    }

    /// Check if we should bypass call budget entirely (DIRECT mode).
    #[inline]
    pub fn is_direct_mode(&self) -> bool {
        self.mode == EmulationMode::Direct
    }

    // Components 

    /// Register a component.  Queues `component_added` signal if running.
    pub fn add_component(&mut self, address: String, type_name: String) {
        if self.is_running() {
            self.push_signal(Signal::new("component_added")
                .with_string(address.clone())
                .with_string(type_name.clone()));
            log::info!("Component added: {type_name} -> {}...", &address[..8.min(address.len())]);
        }
        self.components.insert(address, type_name);
        
    }

    /// Remove a component.  Queues `component_removed` signal if running.
    pub fn remove_component(&mut self, address: &str) {
        if let Some(name) = self.components.remove(address) {
            if self.is_running() {
                self.push_signal(Signal::new("component_removed")
                    .with_string(address.to_owned())
                    .with_string(name));
            }
        }
    }

    /// Iterate over all registered components.
    pub fn components(&self) -> &HashMap<String, String> {
        &self.components
    }

    /// Current uptime in seconds (ticks / 20).
    #[inline]
    pub fn uptime(&self) -> f64 {
        self.uptime_ticks as f64 / 20.0
    }
}

