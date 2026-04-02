//! # Sound system
//!
//! Plays OGG sound effects from `assets/sounds/` and generates
//! square-wave beeps for `computer.beep()`.
//!
//! Uses `cpal` for raw PCM audio output and `lewton` for OGG/Vorbis
//! decoding. All disk/ambient sounds are loaded from actual OGG files
//! matching the OpenComputers originals:
//!
//! ```text
//! assets/sounds/
//!   computer_running.ogg     (44100 Hz, loops while machine is on)
//!   hdd_access1..7.ogg       (24000 Hz, random pick on disk I/O)
//!   floppy_insert.ogg        (24000 Hz)
//!   floppy_eject.ogg         (24000 Hz)
//! ```
//!
//! `computer.beep(freq, duration)` generates a square wave matching
//! the OC Java implementation.
//!
//! ## Disk sound debouncing
//!
//! During boot and heavy I/O, hundreds of filesystem operations can
//! occur within a single tick (50 ms). Without throttling, each
//! `open`/`read`/`write` would trigger a HDD access sound, causing
//! dozens of overlapping voices in the mixer -- an ear-splitting wall
//! of noise.
//!
//! To prevent this, [`play_disk_sound`](SoundSystem::play_disk_sound)
//! enforces a cooldown of [`DISK_SOUND_COOLDOWN`]: if less than 80 ms
//! have elapsed since the last disk sound, the new request is silently
//! dropped. This limits disk sounds to ~12 per second, which sounds
//! like natural hard-drive activity without being overwhelming.
//!
//! ## Volume model
//!
//! Three independent volume controls are provided:
//!
//! * **Master volume** -- global multiplier applied to everything.
//! * **Effect volume** -- controls disk access clicks and the ambient
//!   `computer_running` loop.
//! * **Beep volume** -- controls `computer.beep()` square-wave tones.
//!
//! The effective volume for a sound is `master * category`. All three
//! are stored as [`AtomicU32`] (bit-cast `f32`) so they can be
//! updated from any thread without locking.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "sound")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

// -----------------------------------------------------------------------
// Disk sound cooldown
// -----------------------------------------------------------------------

/// Minimum interval between consecutive disk access sounds.
///
/// During boot, hundreds of filesystem operations happen in quick
/// succession. Without this cooldown, every `open`/`read`/`write`
/// would fire a HDD click simultaneously, creating deafening noise.
/// 80 ms limits playback to ~12 sounds per second, which sounds
/// like realistic hard-drive activity.
const DISK_SOUND_COOLDOWN: Duration = Duration::from_millis(80);

// -----------------------------------------------------------------------
// Atomic f32 helpers
// -----------------------------------------------------------------------

/// Store an `f32` in an `AtomicU32` via bit reinterpretation.
#[inline]
fn atomic_f32(v: f32) -> AtomicU32 {
    AtomicU32::new(v.to_bits())
}

/// Load an `f32` from an `AtomicU32` via bit reinterpretation.
#[inline]
fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Store an `f32` into an `AtomicU32`.
#[inline]
fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Disk access sound effect identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiskSound {
    HddAccess1,
    HddAccess2,
    HddAccess3,
    HddAccess4,
    HddAccess5,
    HddAccess6,
    HddAccess7,
    FloppyInsert,
    FloppyEject,
}

impl DiskSound {
    pub fn filename(self) -> &'static str {
        match self {
            Self::HddAccess1   => "hdd_access1.ogg",
            Self::HddAccess2   => "hdd_access2.ogg",
            Self::HddAccess3   => "hdd_access3.ogg",
            Self::HddAccess4   => "hdd_access4.ogg",
            Self::HddAccess5   => "hdd_access5.ogg",
            Self::HddAccess6   => "hdd_access6.ogg",
            Self::HddAccess7   => "hdd_access7.ogg",
            Self::FloppyInsert => "floppy_insert.ogg",
            Self::FloppyEject  => "floppy_eject.ogg",
        }
    }

    /// Pick a random HDD access sound (1-7).
    pub fn random_hdd() -> Self {
        let r = crate::components::fastrand_byte_pub() % 7;
        [
            Self::HddAccess1, Self::HddAccess2, Self::HddAccess3,
            Self::HddAccess4, Self::HddAccess5, Self::HddAccess6,
            Self::HddAccess7,
        ][r as usize]
    }
}

// -----------------------------------------------------------------------
// Voice / Mixer (shared between main thread and audio thread)
// -----------------------------------------------------------------------

/// A single playing sound.
struct Voice {
    /// Mono f32 samples at the output device sample rate.
    samples: Arc<Vec<f32>>,
    /// Current playback position in samples.
    position: usize,
    /// Volume multiplier (0.0 - 1.0).
    volume: f32,
    /// Whether this voice loops back to the start on completion.
    looping: bool,
    /// Unique ID for stopping specific voices.
    id: u64,
}

/// Additive mixer. Accessed from both the audio callback thread
/// and the main thread via `Arc<Mutex<Mixer>>`.
struct Mixer {
    voices: Vec<Voice>,
    next_id: u64,
}

impl Mixer {
    fn new() -> Self {
        Self { voices: Vec::new(), next_id: 1 }
    }

    /// Add a voice. Returns its ID.
    fn play(&mut self, samples: Arc<Vec<f32>>, volume: f32, looping: bool) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.voices.push(Voice { samples, position: 0, volume, looping, id });
        id
    }

    /// Remove a voice by ID.
    fn stop(&mut self, id: u64) {
        self.voices.retain(|v| v.id != id);
    }

    /// Called from the audio callback. Fills interleaved output buffer.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        // Silence first
        out.fill(0.0);

        if channels == 0 { return; }
        let num_frames = out.len() / channels;

        self.voices.retain_mut(|voice| {
            for frame in 0..num_frames {
                if voice.position >= voice.samples.len() {
                    if voice.looping {
                        voice.position = 0;
                    } else {
                        return false; // voice finished, remove it
                    }
                }

                let s = voice.samples[voice.position] * voice.volume;
                let base = frame * channels;
                for ch in 0..channels {
                    out[base + ch] += s;
                }
                voice.position += 1;
            }
            true
        });

        // Hard clamp to prevent clipping
        for s in out.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }
    }
}

// -----------------------------------------------------------------------
// SoundSystem
// -----------------------------------------------------------------------

/// The sound system. All methods are no-ops if audio is unavailable.
pub struct SoundSystem {
    mixer: Arc<Mutex<Mixer>>,
    /// Device output sample rate.
    output_sr: u32,
    /// Number of output channels.
    output_channels: usize,
    /// Pre-decoded sound cache: filename -> mono samples at output SR.
    cache: HashMap<String, Arc<Vec<f32>>>,
    /// cpal stream handle (must be kept alive).
    #[cfg(feature = "sound")]
    _stream: Option<cpal::Stream>,
    /// Voice ID of the looping computer_running sound, if playing.
    running_voice: Mutex<Option<u64>>,

    /// Master volume multiplier (0.0–1.0). Applied to all sounds.
    ///
    /// Stored as `AtomicU32` (bit-cast `f32`) so it can be updated
    /// from any thread without locking the mixer.
    master_volume: AtomicU32,

    /// Volume for `computer.beep()` tones (0.0–1.0).
    ///
    /// Effective beep volume = `master_volume * beep_volume`.
    pub beep_volume: AtomicU32,

    /// Volume for disk access and ambient effects (0.0–1.0).
    ///
    /// Effective effect volume = `master_volume * effect_volume`.
    pub effect_volume: AtomicU32,

    /// Timestamp of the last disk access sound that was actually played.
    ///
    /// Used to enforce [`DISK_SOUND_COOLDOWN`]: if less than 80 ms
    /// have passed, new disk sound requests are silently dropped.
    /// This prevents the deafening overlap of hundreds of HDD clicks
    /// during boot or heavy I/O.
    last_disk_sound: Mutex<Instant>,
}

// cpal::Stream is Send on most platforms. We only access SoundSystem
// from the main thread anyway; the stream just needs to stay alive.
unsafe impl Send for SoundSystem {}
unsafe impl Sync for SoundSystem {}

impl SoundSystem {
    /// Create the sound system. Falls back to silent on any failure.
    ///
    /// # Arguments
    ///
    /// * `master_vol` - Initial master volume (0.0–1.0).
    /// * `effect_vol` - Initial effect volume (0.0–1.0).
    /// * `beep_vol` - Initial beep volume (0.0–1.0).
    pub fn new(master_vol: f32, effect_vol: f32, beep_vol: f32) -> Self {
        let mixer = Arc::new(Mutex::new(Mixer::new()));

        #[cfg(feature = "sound")]
        {
            match Self::open_device(mixer.clone()) {
                Some((stream, sr, ch)) => {
                    eprintln!("[sound] Audio: {}Hz {}ch", sr, ch);
                    let mut sys = Self {
                        mixer,
                        output_sr: sr,
                        output_channels: ch,
                        cache: HashMap::new(),
                        _stream: Some(stream),
                        running_voice: Mutex::new(None),
                        master_volume: atomic_f32(master_vol.clamp(0.0, 1.0)),
                        beep_volume: atomic_f32(beep_vol.clamp(0.0, 1.0)),
                        effect_volume: atomic_f32(effect_vol.clamp(0.0, 1.0)),
                        last_disk_sound: Mutex::new(
                            Instant::now() - Duration::from_secs(1)
                        ),
                    };
                    sys.load_sounds();
                    return sys;
                }
                None => {
                    eprintln!("[sound] Audio init failed, running silent");
                }
            }
        }

        #[cfg(not(feature = "sound"))]
        eprintln!("[sound] Compiled without sound feature");

        Self {
            mixer,
            output_sr: 44100,
            output_channels: 2,
            cache: HashMap::new(),
            #[cfg(feature = "sound")]
            _stream: None,
            running_voice: Mutex::new(None),
            master_volume: atomic_f32(master_vol.clamp(0.0, 1.0)),
            beep_volume: atomic_f32(beep_vol.clamp(0.0, 1.0)),
            effect_volume: atomic_f32(effect_vol.clamp(0.0, 1.0)),
            last_disk_sound: Mutex::new(
                Instant::now() - Duration::from_secs(1)
            ),
        }
    }

    pub fn is_available(&self) -> bool {
        #[cfg(feature = "sound")]
        { self._stream.is_some() }
        #[cfg(not(feature = "sound"))]
        { false }
    }

    // -- Volume getters / setters ---------------------------------------

    /// Get the current master volume (0.0–1.0).
    pub fn get_master_volume(&self) -> f32 { load_f32(&self.master_volume) }

    /// Get the current beep volume (0.0–1.0).
    pub fn get_beep_volume(&self) -> f32 { load_f32(&self.beep_volume) }

    /// Get the current effect volume (0.0–1.0).
    pub fn get_effect_volume(&self) -> f32 { load_f32(&self.effect_volume) }

    /// Set the master volume (clamped to 0.0–1.0).
    pub fn set_master_volume(&self, v: f32) {
        store_f32(&self.master_volume, v.clamp(0.0, 1.0));
    }

    /// Set the beep volume (clamped to 0.0–1.0).
    pub fn set_beep_volume(&self, v: f32) {
        store_f32(&self.beep_volume, v.clamp(0.0, 1.0));
    }

    /// Set the effect volume (clamped to 0.0–1.0).
    pub fn set_effect_volume(&self, v: f32) {
        store_f32(&self.effect_volume, v.clamp(0.0, 1.0));
    }

    /// Effective beep volume: `master * beep`.
    #[inline]
    fn effective_beep_vol(&self) -> f32 {
        load_f32(&self.master_volume) * load_f32(&self.beep_volume)
    }

    /// Effective effect volume: `master * effect`.
    #[inline]
    fn effective_effect_vol(&self) -> f32 {
        load_f32(&self.master_volume) * load_f32(&self.effect_volume)
    }

    /// Update the volume of the running loop voice (if active) to
    /// match the current effect volume setting. Call this after
    /// changing volume sliders so the ambient loop adjusts immediately.
    pub fn update_running_volume(&self) {
        let rv = self.running_voice.lock().unwrap();
        if let Some(id) = *rv {
            if let Ok(mut mx) = self.mixer.lock() {
                if let Some(voice) = mx.voices.iter_mut().find(|v| v.id == id) {
                    voice.volume = self.effective_effect_vol() * 0.5;
                }
            }
        }
    }

    // -- Device init (feature-gated) ------------------------------------

    #[cfg(feature = "sound")]
    fn open_device(mixer: Arc<Mutex<Mixer>>) -> Option<(cpal::Stream, u32, usize)> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;

        let supported = device.default_output_config().ok()?;
        let sr = supported.sample_rate();
        let ch = supported.channels() as usize;
        let config: cpal::StreamConfig = supported.into();

        let m = mixer.clone();
        let channels = ch;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut mx) = m.lock() {
                    mx.fill(data, channels);
                } else {
                    // Mutex poisoned, output silence
                    data.fill(0.0);
                }
            },
            |err| eprintln!("[sound] Stream error: {err}"),
            None,
        ).ok()?;

        stream.play().ok()?;
        Some((stream, sr, ch))
    }

    // -- OGG loading and caching ----------------------------------------

    /// Load all OGG files from `assets/sounds/`, decode to mono f32
    /// at the output sample rate, and cache them.
    fn load_sounds(&mut self) {
        if !self.is_available() { return; }

        let dir = std::path::Path::new("assets/sounds");
        if !dir.is_dir() {
            eprintln!("[sound] assets/sounds/ not found");
            return;
        }

        let files = [
            "computer_running.ogg",
            "hdd_access1.ogg", "hdd_access2.ogg", "hdd_access3.ogg",
            "hdd_access4.ogg", "hdd_access5.ogg", "hdd_access6.ogg",
            "hdd_access7.ogg",
            "floppy_insert.ogg", "floppy_eject.ogg",
        ];

        for name in &files {
            let path = dir.join(name);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    match self.decode_ogg(&bytes) {
                        Some(samples) => {
                            eprintln!("[sound]   {} -> {} samples", name, samples.len());
                            self.cache.insert(name.to_string(), Arc::new(samples));
                        }
                        None => {
                            eprintln!("[sound]   {} DECODE FAILED", name);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[sound]   {} not found: {}", name, e);
                }
            }
        }

        eprintln!("[sound] Cached {} sounds", self.cache.len());
    }

    /// Decode an OGG/Vorbis file to mono f32, resampled to output SR.
    #[cfg(feature = "sound")]
    fn decode_ogg(&self, data: &[u8]) -> Option<Vec<f32>> {
        use lewton::inside_ogg::OggStreamReader;

        let cursor = Cursor::new(data);
        let mut reader = OggStreamReader::new(cursor).ok()?;

        let source_sr = reader.ident_hdr.audio_sample_rate;
        let source_ch = reader.ident_hdr.audio_channels as usize;

        let mut mono_samples: Vec<f32> = Vec::new();

        loop {
            match reader.read_dec_packet_itl() {
                Ok(Some(packet)) => {
                    // packet is Vec<i16>, interleaved across channels
                    for chunk in packet.chunks(source_ch) {
                        // Mix to mono by averaging channels
                        let sum: f32 = chunk.iter()
                            .map(|&s| s as f32 / 32768.0)
                            .sum();
                        mono_samples.push(sum / source_ch as f32);
                    }
                }
                Ok(None) => break,  // end of stream
                Err(e) => {
                    eprintln!("[sound] Vorbis decode error: {:?}", e);
                    break;
                }
            }
        }

        if mono_samples.is_empty() {
            return None;
        }

        // Resample from source_sr to output_sr if needed
        if source_sr != self.output_sr {
            Some(resample_linear(&mono_samples, source_sr, self.output_sr))
        } else {
            Some(mono_samples)
        }
    }

    #[cfg(not(feature = "sound"))]
    fn decode_ogg(&self, _data: &[u8]) -> Option<Vec<f32>> { None }

    // == Public API =====================================================

    // -- Beep -----------------------------------------------------------

    /// `computer.beep(frequency, duration)`.
    ///
    /// Generates a square wave. Frequency in Hz (20-2000), duration
    /// in seconds (0.05-5.0). Matches OC's Java implementation.
    pub fn beep(&self, frequency: f64, duration_secs: f64) {
        if !self.is_available() { return; }
        let freq = frequency.clamp(20.0, 2000.0);
        let dur = duration_secs.clamp(0.05, 5.0);
        let samples = gen_square_wave(freq, dur, self.output_sr);
        let arc = Arc::new(samples);
        if let Ok(mut mx) = self.mixer.lock() {
            mx.play(arc, self.effective_beep_vol(), false);
        }
    }

    /// `computer.beep(pattern)` where `.` = short, `-` = long.
    ///
    /// Frequency: 1000 Hz. Short: 0.1s. Long: 0.3s. Pause: 0.1s.
    pub fn beep_pattern(&self, pattern: &str) {
        if !self.is_available() { return; }
        let sr = self.output_sr;
        let freq = 1000.0;
        let short_dur = 0.1;
        let long_dur = 0.3;
        let pause_samples = (sr as f64 * 0.1) as usize;

        let mut all: Vec<f32> = Vec::new();
        for ch in pattern.chars() {
            match ch {
                '.' => {
                    all.extend_from_slice(&gen_square_wave(freq, short_dur, sr));
                    all.extend(std::iter::repeat(0.0f32).take(pause_samples));
                }
                '-' => {
                    all.extend_from_slice(&gen_square_wave(freq, long_dur, sr));
                    all.extend(std::iter::repeat(0.0f32).take(pause_samples));
                }
                _ => {
                    // Space or unknown -> just pause
                    all.extend(std::iter::repeat(0.0f32).take(pause_samples));
                }
            }
        }

        if !all.is_empty() {
            let arc = Arc::new(all);
            if let Ok(mut mx) = self.mixer.lock() {
                mx.play(arc, self.effective_beep_vol(), false);
            }
        }
    }

    // -- Disk sounds (from OGG files) -----------------------------------

    /// Play a disk access sound from the cache, with debouncing.
    ///
    /// If less than [`DISK_SOUND_COOLDOWN`] (80 ms) has elapsed since
    /// the last disk sound, the request is silently dropped. This
    /// prevents the deafening overlap of hundreds of simultaneous HDD
    /// clicks during boot or heavy I/O (e.g. `ls` reading many files).
    ///
    /// The cooldown also solves the perceived "non-random" sound issue:
    /// without it, dozens of overlapping voices merge into an
    /// indistinguishable wall of noise. With the cooldown, each
    /// individual random sound is clearly audible.
    pub fn play_disk_sound(&self, sound: DiskSound) {
        if !self.is_available() { return; }

        // Debounce: skip if the last disk sound was too recent.
        let now = Instant::now();
        {
            let mut last = self.last_disk_sound.lock().unwrap();
            if now.duration_since(*last) < DISK_SOUND_COOLDOWN {
                return;
            }
            *last = now;
        }

        let name = sound.filename();
        if let Some(samples) = self.cache.get(name) {
            if let Ok(mut mx) = self.mixer.lock() {
                mx.play(Arc::clone(samples), self.effective_effect_vol(), false);
            }
        }
    }

    // -- Computer running loop ------------------------------------------

    /// Start the looping computer_running.ogg ambient sound.
    pub fn start_running_loop(&self) {
        if !self.is_available() { return; }
        let mut rv = self.running_voice.lock().unwrap();
        if rv.is_some() { return; }

        if let Some(samples) = self.cache.get("computer_running.ogg") {
            if let Ok(mut mx) = self.mixer.lock() {
                let id = mx.play(Arc::clone(samples), self.effective_effect_vol() * 0.5, true);
                *rv = Some(id);
            }
        }
    }

    /// Stop the computer_running loop.
    pub fn stop_running_loop(&self) {
        let mut rv = self.running_voice.lock().unwrap();
        if let Some(id) = rv.take() {
            if let Ok(mut mx) = self.mixer.lock() {
                mx.stop(id);
            }
        }
    }

    // -- One-shot from cache by name ------------------------------------

    /// Play any cached sound by filename (fire-and-forget).
    pub fn play_cached(&self, name: &str, volume: f32) {
        if !self.is_available() { return; }
        if let Some(samples) = self.cache.get(name) {
            let effective = load_f32(&self.master_volume) * volume;
            if let Ok(mut mx) = self.mixer.lock() {
                mx.play(Arc::clone(samples), effective, false);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Waveform generation
// -----------------------------------------------------------------------

/// Generate a square wave at the given frequency and duration.
///
/// This matches how OpenComputers generates beeps: a simple square
/// wave with a short fade-in/fade-out to prevent click artifacts.
///
/// * `freq` - Frequency in Hz.
/// * `duration_secs` - Duration in seconds.
/// * `sr` - Output sample rate.
fn gen_square_wave(freq: f64, duration_secs: f64, sr: u32) -> Vec<f32> {
    let num_samples = (sr as f64 * duration_secs) as usize;
    if num_samples == 0 { return Vec::new(); }

    // Half-period in samples (integer for clean transitions)
    let half_period = ((sr as f64 / freq) / 2.0).round() as usize;
    if half_period == 0 { return vec![0.0; num_samples]; }

    let full_period = half_period * 2;

    // Fade length: 2ms or 10% of duration, whichever is shorter
    let fade = ((sr as f64 * 0.002) as usize).min(num_samples / 10).max(1);

    let mut out = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // Square wave: +1 for first half of period, -1 for second
        let pos_in_period = i % full_period;
        let raw: f32 = if pos_in_period < half_period { 1.0 } else { -1.0 };

        // Envelope: short linear fade in/out to prevent clicks
        let env: f32 = if i < fade {
            i as f32 / fade as f32
        } else if i >= num_samples - fade {
            (num_samples - 1 - i) as f32 / fade as f32
        } else {
            1.0
        };

        out.push(raw * env);
    }

    out
}

// -----------------------------------------------------------------------
// Resampling (linear interpolation)
// -----------------------------------------------------------------------

/// Resample mono audio from `from_sr` Hz to `to_sr` Hz using linear
/// interpolation. Good enough for sound effects; not audiophile quality.
fn resample_linear(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if input.is_empty() || from_sr == 0 || to_sr == 0 {
        return Vec::new();
    }

    let ratio = from_sr as f64 / to_sr as f64;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        let s0 = input.get(idx).copied().unwrap_or(0.0);
        let s1 = input.get(idx + 1).copied().unwrap_or(s0);

        output.push(s0 + (s1 - s0) * frac);
    }

    output
}