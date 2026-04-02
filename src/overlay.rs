//! # Settings GUI (F8) and Debug Bar (F9)
//!
//! This module provides two overlay features:
//!
//! ## Settings GUI (F8)
//!
//! A Catppuccin Mocha-themed settings panel rendered in software.
//! Features:
//!
//! * Keyboard and mouse navigation.
//! * Auto-scaling: scale 1 at <1000px, scale 2 at 1000-1599px, scale 3 at 1600+px.
//! * Panel width = 50% of window (clamped to 50-80 character columns).
//! * Four tabs: Machine, Display, Storage, Fun.
//! * Widget types: Toggle, Choice (dropdown), Slider, Section header,
//!   Separator, Info text.
//! * "Pinky mode" (Emacs-style keybindings) with humorous arrow-key
//!   rejection messages.
//!
//! ```text
//! +----------------------------------------------+
//! | ** OpenComputers Emulator **                  |
//! |----------------------------------------------|
//! | [Machine]  Display   Storage   Fun           |
//! |----------------------------------------------|
//! | -- Execution --                              |
//! | >> Timeout (s)      [====|-----] 5.0         |
//! |    Signal queue      [< 256 >]               |
//! | -- Sandbox --                                |
//! |    Ignore power      [ ] --                  |
//! |    Allow bytecode    [ ] --                  |
//! |----------------------------------------------|
//! | Arrows: nav | Enter: toggle | Tab: next tab  |
//! +----------------------------------------------+
//! ```
//!
//! ## Debug Bar (F9)
//!
//! A single-row overlay at the top of the screen showing real-time
//! performance metrics:
//!
//! ```text
//! Software | 120 fps | 0.83ms | CPU: 1280x800 Fifo | buf 80x25 | sig 0 | lua 3 | up 12.5s
//! ```
//!
//! Rendered as:
//! * A second Vulkan draw call (when using Vulkan backends), or
//! * A software composite (when using the Software backend).
//!
//! ## Colour theme (Catppuccin Mocha)
//!
//! All UI colours are sourced from the Catppuccin Mocha palette:
//! * Background: `#1E1E2E` (Base)
//! * Panel: `#181825` (Mantle)
//! * Border: `#313244` (Surface0)
//! * Text: `#CDD6F4` (Text)
//! * Accent: `#89B4FA` (Blue)
//! * Values: `#89B4FA` (Blue), highlighted: `#B4BEFE` (Lavender)
//! * Toggles ON: `#A6E3A1` (Green)
//! * Toggles OFF: `#585B70` (Surface2)
//! * Flash messages: various (Yellow, Green, Purple, Cyan, Red, Gray)
//!
//! ## Pinky mode
//!
//! When enabled in the "Fun" tab, replaces arrow-key navigation with
//! Emacs-style keybindings:
//!
//! * C-n / C-p: navigate down / up
//! * C-f / C-b: adjust right / left
//! * C-a / C-e: jump to first / last item
//! * C-g: cancel (close without saving)
//! * C-x prefix: secret commands (C-x C-s "saves", C-x C-c "quits")
//! * M-1..M-4: switch tabs
//! * Enter: activate / toggle
//!
//! Arrow keys in pinky mode trigger a rotating set of humorous
//! rejection messages.

use crate::config::OcConfig;
use crate::display::font::{GlyphAtlas, ATLAS_SIZE, CELL_W, CELL_H};
use crate::render::RenderMode;
use std::time::{Duration, Instant};

// == Catppuccin Mocha theme ===================================================

mod theme {
    pub const BG: u32         = 0x1E1E2E;
    pub const PANEL: u32      = 0x181825;
    pub const BORDER: u32     = 0x313244;
    pub const HEADER: u32     = 0x313244;
    pub const TITLE: u32      = 0xCDD6F4;
    pub const LABEL: u32      = 0xBAC2DE;
    pub const VALUE: u32      = 0x89B4FA;
    pub const VALUE_HI: u32   = 0xB4BEFE;
    pub const HINT: u32       = 0x6C7086;
    pub const BRIGHT: u32     = 0xCDD6F4;
    pub const ACCENT: u32     = 0x89B4FA;
    pub const CURSOR: u32     = 0xF5C2E7;
    pub const ON: u32         = 0xA6E3A1;
    pub const OFF: u32        = 0x585B70;
    pub const TAB_ON: u32     = 0x89B4FA;
    pub const TAB_OFF: u32    = 0x6C7086;
    pub const TAB_HOVER: u32  = 0xF5C2E7;
    pub const HOVER_BG: u32   = 0x313244;
    pub const SEL_BG: u32     = 0x45475A;
    pub const SEP: u32        = 0x45475A;
    pub const SLIDER_FG: u32  = 0x89B4FA;
    pub const SLIDER_BG: u32  = 0x45475A;
    pub const SLIDER_TH: u32  = 0xCDD6F4;
    pub const SECTION: u32    = 0xF5C2E7;
    pub const PINKY_BDR: u32  = 0xCBA6F7;
    pub const FL_YELLOW: u32  = 0xF9E2AF;
    pub const FL_GREEN: u32   = 0xA6E3A1;
    pub const FL_PURPLE: u32  = 0xCBA6F7;
    pub const FL_CYAN: u32    = 0x94E2D5;
    pub const FL_RED: u32     = 0xF38BA8;
    pub const FL_GRAY: u32    = 0x6C7086;
    pub const BAR_FG: u32     = 0xA6E3A1;
    pub const BAR_BG: u32     = 0x11111B;
}

// == Widgets ==================================================================

#[derive(Debug, Clone)]
pub enum Widget {
    Toggle   { label: &'static str, key: &'static str, value: bool },
    Choice   { label: &'static str, key: &'static str,
               options: &'static [&'static str], index: usize },
    Slider   { label: &'static str, key: &'static str,
               value: f64, min: f64, max: f64, step: f64 },
    Section  { title: &'static str },
    Separator,
    Info        { text: &'static str, color: u32 },
    DynamicChoice
                { label: &'static str, key: &'static str,
                    options: Vec<(String, String)>, index: usize },
    Button      { label: &'static str, key: &'static str },
}

impl Widget {
    fn is_interactive(&self) -> bool {
        matches!(self, Self::Toggle{..} | Self::Choice{..} | Self::Slider{..}
                     | Self::DynamicChoice{..} | Self::Button{..})
    }
}

struct Tab { name: &'static str, items: Vec<Widget> }

struct Flash { text: String, color: u32, until: Instant }

pub enum GuiAction { None, CloseApply, CloseCancel }

const ARROW_MSGS: &[&str] = &[
    // -- Classic Emacs mockery --
    "Arrow keys? In MY Emacs? Try C-n/C-p.",
    "Arrows are for the carpal-tunnel-free.",
    "You pressed an arrow. RMS is disappointed.",
    "M-x use-arrow-keys RET -> Permission denied.",
    "Did you mean C-n? Of course you did.",
    "Arrows are a crutch. Embrace the Ctrl key.",
    "M-x butterfly is not available either.",
    "Your pinky called. It misses you.",
    "C-x C-c? Nice try. There is no exit.",
    "Arrows are O(n). C-n is O(1). Be asymptotic.",
    "RMS wrote GCC without arrow keys. What's your excuse?",
    "Every arrow press deletes a parenthesis in some Lisp file.",
    "Fun fact: arrow keys were invented by vi users.",
    "M-x therapy RET -> Have you tried C-n?",
    "Arrow keys are just HJKL with extra steps.",
    "The Church of Emacs condemns your arrow heresy.",
    "In Soviet Emacs, C-n presses YOU.",
    "Somewhere, an Elisp function cries when you use arrows.",
    "C-M-S-arrow? Now you're just showing off... badly.",
    "Pro tip: rebind arrows to C-n/p/f/b. Problem solved forever.",
    // -- Vi/Vim gets it too --
    "At least you're not pressing hjkl. ...Wait, are you a Vi spy?",
    "A Vi user walks into a bar. Types :q to leave. Can't. This is Emacs.",
    "hjkl? What is this, a Langton's ant simulator?",
    "Vi users peak in normal mode. Emacs users never leave it.",
    "Heard you use Vim. M-x condolences RET.",
    "Vi has two modes: beeping and breaking things.",
    "Vim is great! If you enjoy modal suffering.",
    ":wq? Sir, this is an Emacs. We use C-x C-s C-x C-c here. Sometimes.",
    "Vi: the editor that beeps at you judgmentally.",
    "Vim users press Esc 47 times per minute. We press Ctrl 48.",
    "They say Vi is the editor of the beast. 6 keystrokes to quit: :wq!<CR>.",
    "In Vim you Esc from your problems. In Emacs you C-g and face them.",
    "Vim's idea of extensibility: more modes. Our idea: more parentheses.",
    "A Vim macro walks into a bar. @@ @@ @@ segfault.",
    ":set nocompatible -> still not compatible with good taste.",
    "Vim users brag about efficiency. Then spend 40 min configuring .vimrc.",
    "Modal editing: because pressing 'i' before typing is peak UX.",
    "You can quit Vim with ZZ. You can never quit Emacs. Feature, not bug.",
    // -- Existential dread --
    "The arrow keys are a mass hallucination. Only Ctrl is real.",
    "If a tree falls in a forest and no one uses C-n, did it really nav?",
    "Emacs is not an editor. It's a lifestyle you can't arrow out of.",
    "C-h t -> Emacs tutorial. Chapter 1: Forget everything about arrows.",
    "M-x doctor says: Tell me about your relationship with arrow keys.",
    "You have mass-selected 0 things with that arrow. C-SPC does it better.",
    "Your arrow key has mass. It slows you down relativistically.",
    "Heat death of the universe ETA: sooner than you'll learn C-n.",
];

fn compute_scale(wh: u32) -> u32 {
    if wh >= 1600 { 3 } else if wh >= 1000 { 2 } else { 1 }
}

// NEW: scale multiplier parsing for budget/call-budget choices
fn parse_scale_str(s: &str) -> f64 {
    match s {
        "0.25x"     => 0.25,
        "0.5x"      => 0.5,
        "1x"        => 1.0,
        "2x"        => 2.0,
        "4x"        => 4.0,
        "8x"        => 8.0,
        "Unlimited" => f64::INFINITY,
        _           => 1.0,
    }
}

// NEW: inverse of parse_scale_str
fn scale_to_option(v: f64) -> &'static str {
    if v.is_infinite()    { "Unlimited" }
    else if v >= 7.0      { "8x" }
    else if v >= 3.0      { "4x" }
    else if v >= 1.5      { "2x" }
    else if v >= 0.75     { "1x" }
    else if v >= 0.375    { "0.5x" }
    else                  { "0.25x" }
}

// == SettingsGui ==============================================================

pub struct SettingsGui {
    pub visible: bool,
    tabs: Vec<Tab>,
    active_tab: usize,
    cursor: usize,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub pinky: bool,
    pending_cx: bool,
    arrow_idx: usize,
    flash: Option<Flash>,
    // Layout cache (set during render)
    panel_ox: u32,
    panel_oy: u32,
    tab_row_y: u32,
    content_y: u32,
    scale: u32,
    cw: u32,
    ch: u32,
    row_h: u32,
    pub reboot_requested: bool,
    pub pending_beep: Option<(f64, f64)>,
    // NEW: live-apply flag
    pub changed: bool,
}

impl SettingsGui {
    pub fn new() -> Self {
        Self {
            visible: false,
            tabs: vec![
                Tab { name: "Machine", items: vec![
                    Widget::Section { title: "Boot" },
                    Widget::DynamicChoice { label: "Boot device", key: "boot_device",
                        options: vec![], index: 0 },
                    Widget::Button { label: "Reboot now", key: "reboot" },
                    Widget::Separator,
                    Widget::Section { title: "Execution" },
                    Widget::Slider { label: "Timeout (s)", key: "timeout",
                        value: 5.0, min: 1.0, max: 30.0, step: 0.5 },
                    Widget::Choice { label: "Signal queue", key: "max_signals",
                        options: &["64","128","256","512","1024"], index: 2 },
                    Widget::Separator,
                    // NEW: CPU & Memory section
                    Widget::Section { title: "CPU & Memory" },
                    Widget::Choice { label: "CPU Tier", key: "cpu_tier",
                        options: &["T1", "T2", "T3"], index: 2 },
                    Widget::Choice { label: "RAM", key: "ram_tier",
                        options: &["T1 (192K)","T2 (384K)","T3 (768K)",
                                   "T4 (Unlimited)"], index: 2 },
                    Widget::Choice { label: "Budget Scale", key: "call_budget_scale",
                        options: &["0.25x","0.5x","1x","2x","4x","8x",
                                   "Unlimited"], index: 2 },
                    Widget::Separator,
                    Widget::Section { title: "Sandbox" },
                    Widget::Toggle { label: "Ignore power", key: "ignore_power", value: false },
                    Widget::Toggle { label: "Allow bytecode", key: "allow_bytecode", value: false },
                    Widget::Toggle { label: "Allow __gc", key: "allow_gc", value: false },
                    Widget::Toggle { label: "Allow persistence", key: "allow_persist", value: true },
                ]},
                Tab { name: "Display", items: vec![
                    Widget::Section { title: "Renderer" },
                    Widget::Choice { label: "Render mode", key: "render_mode",
                        options: &["Software (CPU)","Vulkan Indirect","Vulkan Direct"], index: 0 },
                    Widget::Toggle { label: "VSync", key: "vsync", value: false },
                    Widget::Choice { label: "FPS limit", key: "fps_limit",
                        options: &["Unlimited","60","72","75","90","120","144",
                                   "160","175","180","200","240"], index: 0 },
                    Widget::Separator,
                    // NEW: GPU Performance section
                    Widget::Section { title: "GPU Performance" },
                    Widget::Choice { label: "GPU Budget", key: "gpu_budget_scale",
                        options: &["0.25x","0.5x","1x","2x","4x","8x",
                                   "Unlimited"], index: 2 },
                    Widget::Choice { label: "Screen Tier", key: "screen_tier",
                        options: &["T1 (50x16 1-bit)","T2 (80x25 4-bit)",
                                   "T3 (160x50 8-bit)"], index: 2 },
                    Widget::Separator,
                    Widget::Section { title: "Sound" },
                    Widget::Slider { label: "Master volume", key: "vol_master",
                        value: 1.0, min: 0.0, max: 1.0, step: 0.1 },
                    Widget::Slider { label: "Effect volume", key: "vol_effect",
                        value: 0.4, min: 0.0, max: 1.0, step: 0.1 },
                    Widget::Slider { label: "Beep volume", key: "vol_beep",
                        value: 0.3, min: 0.0, max: 1.0, step: 0.1 },
                    Widget::Separator,
                    Widget::Info { text: "F5 also cycles renderers", color: theme::HINT },
                    Widget::Info { text: "FPS limit applies when VSync is OFF", color: theme::HINT },
                    Widget::Info { text: "All changes apply instantly", color: theme::HINT }, // NEW
                ]},
                Tab { name: "Storage", items: vec![
                    Widget::Section { title: "EEPROM" },
                    Widget::Choice { label: "Code size", key: "eeprom_size",
                        options: &["1024","2048","4096","8192","16384"], index: 2 },
                    Widget::Choice { label: "Data size", key: "eeprom_data",
                        options: &["64","128","256","512"], index: 2 },
                    Widget::Separator,
                    Widget::Section { title: "Filesystem" },
                    Widget::Choice { label: "Tmp size (KiB)", key: "tmp_size",
                        options: &["0","32","64","128","256"], index: 2 },
                    Widget::Choice { label: "Max handles", key: "max_handles",
                        options: &["4","8","12","16","24","32"], index: 2 },
                    Widget::Choice { label: "Read buffer", key: "max_read_buf",
                        options: &["512","1024","2048","4096","8192"], index: 2 },
                    Widget::Toggle { label: "Erase tmp on reboot", key: "erase_tmp", value: true },
                ]},
                Tab { name: "Fun", items: vec![
                    Widget::Toggle { label: "Pinky mode (Emacs)", key: "pinky", value: false },
                    Widget::Separator,
                    Widget::Info { text: "Replaces arrows with C-n/p/f/b", color: theme::HINT },
                    Widget::Info { text: "C-g cancel | M-1..4 tabs | RET ok", color: theme::HINT },
                    Widget::Info { text: "C-x prefix for secret commands", color: theme::HINT },
                ]},
            ],
            active_tab: 0, cursor: 0,
            mouse_x: 0.0, mouse_y: 0.0,
            pinky: false, pending_cx: false, arrow_idx: 0, flash: None,
            panel_ox: 0, panel_oy: 0, tab_row_y: 0, content_y: 0,
            scale: 1, cw: 8, ch: 16, row_h: 18, reboot_requested: false,
            pending_beep: None,
            changed: false, // NEW
        }
    }

    // -- Getters for main.rs --------------------------------------------------

    fn beep_nav(&mut self)      { self.pending_beep = Some((1000.0, 0.015)); }
    fn beep_activate(&mut self) { self.pending_beep = Some((1200.0, 0.025)); }
    fn beep_tab(&mut self)      { self.pending_beep = Some((800.0,  0.025)); }
    fn beep_error(&mut self)    { self.pending_beep = Some((300.0,  0.05));  }

    // NEW: drain the pending beep
    pub fn take_beep(&mut self) -> Option<(f64, f64)> {
        self.pending_beep.take()
    }

    // NEW: poll and clear live-apply flag
    pub fn take_changed(&mut self) -> bool {
        let c = self.changed;
        self.changed = false;
        c
    }

    pub fn get_vsync(&self) -> bool {
        self.find_toggle("vsync").unwrap_or(false)
    }

    pub fn get_fps_limit(&self) -> Option<u32> {
        self.find_choice_value("fps_limit")
            .and_then(|v| if v == "Unlimited" { None } else { v.parse().ok() })
    }

    // NEW: render mode getter
    pub fn get_render_mode(&self) -> Option<RenderMode> {
        match self.find_choice_index("render_mode") {
            Some(0) => Some(RenderMode::Software),
            Some(1) => Some(RenderMode::VulkanIndirect),
            Some(2) => Some(RenderMode::VulkanDirect),
            _ => None,
        }
    }

    // NEW: RAM in bytes for the selected tier
    pub fn get_ram_bytes(&self) -> i64 {
        match self.find_choice_index("ram_tier") {
            Some(0) => 192 * 1024,
            Some(1) => 384 * 1024,
            Some(2) => 768 * 1024,
            Some(3) => 64 * 1024 * 1024,
            _       => 768 * 1024,
        }
    }

    // NEW: CPU tier index (0, 1, or 2)
    pub fn get_cpu_tier(&self) -> usize {
        self.find_choice_index("cpu_tier").unwrap_or(2)
    }

    // NEW: call budget multiplier
    pub fn get_call_budget_scale(&self) -> f64 {
        self.find_choice_value("call_budget_scale")
            .map(parse_scale_str)
            .unwrap_or(1.0)
    }

    // NEW: GPU call cost multiplier
    pub fn get_gpu_budget_scale(&self) -> f64 {
        self.find_choice_value("gpu_budget_scale")
            .map(parse_scale_str)
            .unwrap_or(1.0)
    }

    // NEW: screen tier index (0, 1, or 2)
    pub fn get_screen_tier(&self) -> usize {
        self.find_choice_index("screen_tier").unwrap_or(2)
    }

    fn find_toggle(&self, key: &str) -> Option<bool> {
        for t in &self.tabs { for w in &t.items {
            if let Widget::Toggle { key: k, value, .. } = w {
                if *k == key { return Some(*value); }
            }
        }}
        None
    }

    fn find_choice_value(&self, key: &str) -> Option<&str> {
        for t in &self.tabs { for w in &t.items {
            if let Widget::Choice { key: k, index, options, .. } = w {
                if *k == key { return Some(options[*index]); }
            }
        }}
        None
    }

    fn find_slider_value(&self, key: &str) -> Option<f64> {
        for t in &self.tabs { for w in &t.items {
            if let Widget::Slider { key: k, value, .. } = w {
                if *k == key { return Some(*value); }
            }
        }}
        None
    }

    // NEW: find choice index by key
    fn find_choice_index(&self, key: &str) -> Option<usize> {
        for t in &self.tabs { for w in &t.items {
            if let Widget::Choice { key: k, index, .. } = w {
                if *k == key { return Some(*index); }
            }
        }}
        None
    }

    // -- Config sync ----------------------------------------------------------

    pub fn sync_from_config(&mut self, cfg: &OcConfig, mode: RenderMode,
                            vsync: bool, fps_limit: Option<u32>) {
        for tab in &mut self.tabs { for w in &mut tab.items { match w {
            Widget::Toggle { key, value, .. } => match *key {
                "ignore_power"  => *value = cfg.ignore_power,
                "allow_bytecode"=> *value = cfg.allow_bytecode,
                "allow_gc"      => *value = cfg.allow_gc,
                "allow_persist" => *value = cfg.allow_persistence,
                "erase_tmp"     => *value = cfg.erase_tmp_on_reboot,
                "vsync"         => *value = vsync,
                "pinky"         => *value = self.pinky,
                _ => {}
            },
            Widget::Choice { key, index, options, .. } => {
                let find = |val: &str| options.iter().position(|o| *o == val).unwrap_or(0);
                match *key {
                    "render_mode" => *index = match mode {
                        RenderMode::Software => 0,
                        RenderMode::VulkanIndirect => 1,
                        RenderMode::VulkanDirect => 2,
                    },
                    "fps_limit" => *index = match fps_limit {
                        None    => 0, // "Unlimited"
                        Some(v) => find(&v.to_string()),
                    },
                    "max_signals"  => *index = find(&cfg.max_signal_queue_size.to_string()),
                    "eeprom_size"  => *index = find(&cfg.eeprom_size.to_string()),
                    "eeprom_data"  => *index = find(&cfg.eeprom_data_size.to_string()),
                    "tmp_size"     => *index = find(&cfg.tmp_size_kib.to_string()),
                    "max_handles"  => *index = find(&cfg.max_handles.to_string()),
                    "max_read_buf" => *index = find(&cfg.max_read_buffer.to_string()),
                    // NEW
                    "cpu_tier"          => *index = cfg.cpu_tier.min(2),
                    "ram_tier"          => *index = cfg.ram_tier.min(3),
                    "screen_tier"       => *index = cfg.screen_tier.min(2),
                    "call_budget_scale" => *index = find(scale_to_option(cfg.call_budget_scale)),
                    "gpu_budget_scale"  => *index = find(scale_to_option(cfg.gpu_budget_scale)),
                    _ => {}
                }
            }
            Widget::Slider { key, value, .. } => match *key {
                "timeout"    => *value = cfg.timeout,
                "vol_master" => *value = cfg.master_volume as f64,
                "vol_effect" => *value = cfg.effect_volume as f64,
                "vol_beep"   => *value = cfg.beep_volume as f64,
                _ => {}
            },
            _ => {}
        }}}
        self.changed = false; // NEW
    }

    pub fn apply_to_config(&self, cfg: &mut OcConfig) -> Option<RenderMode> {
        let mut mode = None;
        for tab in &self.tabs { for w in &tab.items { match w {
            Widget::Toggle { key, value, .. } => match *key {
                "ignore_power"  => cfg.ignore_power = *value,
                "allow_bytecode"=> cfg.allow_bytecode = *value,
                "allow_gc"      => cfg.allow_gc = *value,
                "allow_persist" => cfg.allow_persistence = *value,
                "erase_tmp"     => cfg.erase_tmp_on_reboot = *value,
                _ => {}
            },
            Widget::Choice { key, index, options, .. } => {
                let val = options[*index];
                match *key {
                    "render_mode" => mode = Some(match *index {
                        1 => RenderMode::VulkanIndirect,
                        2 => RenderMode::VulkanDirect,
                        _ => RenderMode::Software,
                    }),
                    "max_signals"  => cfg.max_signal_queue_size = val.parse().unwrap_or(256),
                    "eeprom_size"  => cfg.eeprom_size = val.parse().unwrap_or(4096),
                    "eeprom_data"  => cfg.eeprom_data_size = val.parse().unwrap_or(256),
                    "tmp_size"     => cfg.tmp_size_kib = val.parse().unwrap_or(64),
                    "max_handles"  => cfg.max_handles = val.parse().unwrap_or(12),
                    "max_read_buf" => cfg.max_read_buffer = val.parse().unwrap_or(2048),
                    // NEW
                    "cpu_tier"          => cfg.cpu_tier = *index,
                    "ram_tier"          => cfg.ram_tier = *index,
                    "screen_tier"       => cfg.screen_tier = *index,
                    "call_budget_scale" => cfg.call_budget_scale = parse_scale_str(val),
                    "gpu_budget_scale"  => cfg.gpu_budget_scale = parse_scale_str(val),
                    _ => {}
                }
            }
            Widget::Slider { key, value, .. } => match *key {
                "timeout"    => cfg.timeout = *value,
                "vol_master" => cfg.master_volume = *value as f32,
                "vol_effect" => cfg.effect_volume = *value as f32,
                "vol_beep"   => cfg.beep_volume = *value as f32,
                _ => {}
            },
            _ => {}
        }}}
        mode
    }

    // -- Navigation -----------------------------------------------------------

    fn clamp_cursor(&mut self) {
        let items = &self.tabs[self.active_tab].items;
        let len = items.len();
        if len == 0 { self.cursor = 0; return; }
        if self.cursor >= len { self.cursor = len - 1; }
        // Skip forward past non-interactive
        while self.cursor < len && !items[self.cursor].is_interactive() {
            self.cursor += 1;
        }
        if self.cursor >= len {
            self.cursor = len.saturating_sub(1);
            while self.cursor > 0 && !items[self.cursor].is_interactive() {
                self.cursor -= 1;
            }
        }
    }

    fn nav_up(&mut self) {
        if self.cursor == 0 { return; }
        self.cursor -= 1;
        let items = &self.tabs[self.active_tab].items;
        while self.cursor > 0 && !items[self.cursor].is_interactive() {
            self.cursor -= 1;
        }
        if !items[self.cursor].is_interactive() { self.clamp_cursor(); }
        self.beep_nav();
    }

    fn nav_down(&mut self) {
        let items = &self.tabs[self.active_tab].items;
        if self.cursor + 1 >= items.len() { return; }
        self.cursor += 1;
        while self.cursor < items.len() - 1 && !items[self.cursor].is_interactive() {
            self.cursor += 1;
        }
        if !items[self.cursor].is_interactive() { self.clamp_cursor(); }
        self.beep_nav();
    }

    fn nav_left(&mut self) {
        let idx = self.cursor;
        let items = &mut self.tabs[self.active_tab].items;
        if let Some(w) = items.get_mut(idx) { match w {
            Widget::Slider { value, min, step, .. } => {
                *value = (*value - *step).max(*min);
                self.changed = true; // NEW
            }
            Widget::Choice { index, options, .. } => {
                *index = if *index == 0 { options.len() - 1 } else { *index - 1 };
                self.changed = true; // NEW
            }
            Widget::DynamicChoice { index, options, .. } => {
                *index = if options.is_empty() { 0 }
                    else if *index == 0 { options.len() - 1 } else { *index - 1 };
                self.changed = true; // NEW
            }
            _ => {}
        }}
        self.beep_nav();
    }

    fn nav_right(&mut self) {
        let idx = self.cursor;
        let items = &mut self.tabs[self.active_tab].items;
        if let Some(w) = items.get_mut(idx) { match w {
            Widget::Slider { value, max, step, .. } => {
                *value = (*value + *step).min(*max);
                self.changed = true; // NEW
            }
            Widget::Choice { index, options, .. } => {
                *index = (*index + 1) % options.len();
                self.changed = true; // NEW
            }
            Widget::DynamicChoice { index, options, .. } => {
                if !options.is_empty() { *index = (*index + 1) % options.len(); }
                self.changed = true; // NEW
            }
            _ => {}
        }}
        self.beep_nav();
    }

    fn activate(&mut self) {
        let idx = self.cursor;
        let tab = self.active_tab;
        let mut was_pinky_toggle = false;
        let mut was_reboot = false;
        {
            let items = &mut self.tabs[tab].items;
            if let Some(w) = items.get_mut(idx) { match w {
                Widget::Toggle { key, value, .. } => {
                    *value = !*value;
                    was_pinky_toggle = *key == "pinky";
                    self.changed = true; // NEW
                }
                Widget::Choice { index, options, .. } => {
                    *index = (*index + 1) % options.len();
                    self.changed = true; // NEW
                }
                Widget::DynamicChoice { index, options, .. } => {
                    if !options.is_empty() { *index = (*index + 1) % options.len(); }
                    self.changed = true; // NEW
                }
                Widget::Button { key, .. } => {
                    if *key == "reboot" { was_reboot = true; }
                }
                _ => {}
            }}
        }
        if was_pinky_toggle {
            let v = match &self.tabs[tab].items[idx] {
                Widget::Toggle { value, .. } => *value, _ => false,
            };
            self.pinky = v;
            self.set_flash(
                if v { "Pinky mode ON! M-x doctor for therapy." }
                else { "Pinky mode OFF. Freedom." },
                if v { theme::FL_PURPLE } else { theme::FL_GREEN },
            );
        }
        if was_reboot {
            self.reboot_requested = true;
        }
        self.beep_activate();
    }

    fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % self.tabs.len();
        self.cursor = 0;
        self.clamp_cursor();
        self.beep_tab();
    }

    fn set_flash(&mut self, text: &str, color: u32) {
        self.flash = Some(Flash { text: text.into(), color,
            until: Instant::now() + Duration::from_secs(2) });
    }

    // -- Keyboard -------------------------------------------------------------

    pub fn handle_key(&mut self, kc: winit::keyboard::KeyCode, ctrl: bool, alt: bool) -> GuiAction {
        use winit::keyboard::KeyCode as K;
        if let Some(f) = &self.flash {
            if Instant::now() >= f.until { self.flash = None; }
        }
        if self.pinky { return self.handle_pinky(kc, ctrl, alt); }
        match kc {
            K::ArrowUp    => self.nav_up(),
            K::ArrowDown  => self.nav_down(),
            K::ArrowLeft  => self.nav_left(),
            K::ArrowRight => self.nav_right(),
            K::Enter | K::Space => {
                self.activate();
                if self.reboot_requested { return GuiAction::CloseApply; }
            }
            K::Tab        => self.next_tab(),
            K::Escape     => return GuiAction::CloseCancel,
            K::F8         => return GuiAction::CloseApply,
            _ => {}
        }
        GuiAction::None
    }

    fn handle_pinky(&mut self, kc: winit::keyboard::KeyCode, ctrl: bool, alt: bool) -> GuiAction {
        use winit::keyboard::KeyCode as K;
        if self.pending_cx {
            self.pending_cx = false;
            if ctrl { match kc {
                K::KeyS => self.set_flash("Wrote settings to /dev/null.", theme::FL_GREEN),
                K::KeyC => self.set_flash("There is no escape. Only Emacs.", theme::FL_PURPLE),
                _ => self.set_flash("C-x C-? is undefined.", theme::FL_RED),
            }} else { self.set_flash("C-x prefix cancelled.", theme::FL_GRAY); }
            return GuiAction::None;
        }
        if ctrl { match kc {
            K::KeyN => self.nav_down(),
            K::KeyP => self.nav_up(),
            K::KeyF => self.nav_right(),
            K::KeyB => self.nav_left(),
            K::KeyA => { self.cursor = 0; self.clamp_cursor(); }
            K::KeyE => {
                self.cursor = self.tabs[self.active_tab].items.len().saturating_sub(1);
                self.clamp_cursor();
            }
            K::KeyG => return GuiAction::CloseCancel,
            K::KeyX => self.pending_cx = true,
            K::KeyH => self.set_flash("You are in a maze of twisty settings.", theme::FL_CYAN),
            _ => {}
        }} else if alt { match kc {
            K::Digit1 => { self.active_tab = 0; self.cursor = 0; self.clamp_cursor(); }
            K::Digit2 => { self.active_tab = 1.min(self.tabs.len()-1); self.cursor = 0; self.clamp_cursor(); }
            K::Digit3 => { self.active_tab = 2.min(self.tabs.len()-1); self.cursor = 0; self.clamp_cursor(); }
            K::Digit4 => { self.active_tab = 3.min(self.tabs.len()-1); self.cursor = 0; self.clamp_cursor(); }
            _ => {}
        }} else { match kc {
            K::Enter => {
                self.activate();
                if self.reboot_requested { return GuiAction::CloseApply; }
            }
            K::F8    => return GuiAction::CloseApply,
            K::ArrowUp | K::ArrowDown | K::ArrowLeft | K::ArrowRight => {
                let msg = ARROW_MSGS[self.arrow_idx % ARROW_MSGS.len()];
                self.arrow_idx += 1;
                self.set_flash(msg, theme::FL_YELLOW);
                self.beep_error();
            }
            K::Tab => self.set_flash("Tab is for the weak. Use M-1..M-4.", theme::FL_YELLOW),
            _ => {}
        }}
        GuiAction::None

    }

    // -- Mouse ----------------------------------------------------------------

    pub fn handle_mouse_move(&mut self, x: f64, y: f64) {
        self.mouse_x = x as f32; self.mouse_y = y as f32;
    }

    pub fn handle_click(&mut self, x: f64, y: f64) -> bool {
        if !self.visible { return false; }
        let (mx, my) = (x as u32, y as u32);
        let (ox, oy, s) = (self.panel_ox, self.panel_oy, self.scale);

        // Tab bar
        if my >= self.tab_row_y && my < self.tab_row_y + self.row_h {
            let mut col_px = ox + self.cw * 2;
            for (i, tab) in self.tabs.iter().enumerate() {
                let tw = (tab.name.len() as u32 + 2) * self.cw;
                if mx >= col_px && mx < col_px + tw {
                    self.active_tab = i;
                    self.cursor = 0;
                    self.clamp_cursor();
                    return true;
                }
                col_px += tw + self.cw;
            }
            return true;
        }

        // Content
        if my >= self.content_y {
            let row = ((my - self.content_y) / self.row_h) as usize;
            let items = &self.tabs[self.active_tab].items;
            if row < items.len() && items[row].is_interactive() {
                self.cursor = row;
                self.activate();
            }
        }
        true
    }

    pub fn handle_scroll(&mut self, dy: f64) {
        if dy > 0.0 { self.nav_up(); } else if dy < 0.0 { self.nav_down(); }
    }

    // -- Rendering ------------------------------------------------------------

    pub fn render(&mut self, atlas: &GlyphAtlas, pixels: &mut [u32], ww: u32, wh: u32) {
        if !self.visible { return; }

        let s = compute_scale(wh);
        self.scale = s;
        self.cw = 8 * s;
        self.ch = 16 * s;
        self.row_h = self.ch + 2 * s;

        let panel_chars = ((ww / 2) / self.cw).max(50).min(80) as u32;
        let items = &self.tabs[self.active_tab].items;
        let content_rows = items.len() as u32;
        let total_rows = content_rows + 6;
        let pw = panel_chars * self.cw;
        let ph = total_rows * self.row_h + 8 * s;

        let ox = (ww.saturating_sub(pw)) / 2;
        let oy = (wh.saturating_sub(ph)) / 2;
        self.panel_ox = ox;
        self.panel_oy = oy;

        let bdr = if self.pinky { theme::PINKY_BDR } else { theme::BORDER };
        let bs = 2 * s;

        // Border + background
        rect(pixels, ww, wh, ox.wrapping_sub(bs), oy.wrapping_sub(bs), pw+bs*2, ph+bs*2, bdr);
        rect(pixels, ww, wh, ox, oy, pw, ph, theme::PANEL);

        let mut y = oy + 4 * s;
        let label_x = ox + self.cw * 3;
        let val_col = panel_chars / 2 + 2;
        let val_x = ox + val_col * self.cw;

        // Title
        let title = if self.pinky { "** PINKY MODE ** Settings **" }
            else { "** OpenComputers Emulator **" };
        let tx = ox + (pw.saturating_sub(title.len() as u32 * self.cw)) / 2;
        stext(atlas, pixels, ww, wh, tx, y, title,
            if self.pinky { theme::FL_PURPLE } else { theme::TITLE }, theme::PANEL, s);
        y += self.row_h;

        // Top separator
        let sep_w = pw - self.cw * 2;
        rect(pixels, ww, wh, ox + self.cw, y + self.ch / 2, sep_w, s.max(1), theme::SEP);
        y += self.row_h / 2 + 2 * s;

        // Tab bar
        self.tab_row_y = y;
        let hmx = self.mouse_x as u32;
        let hmy = self.mouse_y as u32;
        let mut cx = ox + self.cw * 2;
        for (i, tab) in self.tabs.iter().enumerate() {
            let active = i == self.active_tab;
            let label = if active { format!("[{}]", tab.name) } else { format!(" {} ", tab.name) };
            let tw = label.len() as u32 * self.cw;
            let hovered = hmy >= y && hmy < y + self.row_h && hmx >= cx && hmx < cx + tw;
            let fg = if active { theme::TAB_ON }
                else if hovered { theme::TAB_HOVER }
                else { theme::TAB_OFF };
            stext(atlas, pixels, ww, wh, cx, y, &label, fg, theme::PANEL, s);
            cx += tw + self.cw;
        }
        y += self.row_h;

        // Separator
        rect(pixels, ww, wh, ox + self.cw, y + self.ch / 2, sep_w, s.max(1), theme::SEP);
        y += self.row_h / 2 + s;

        // Content
        self.content_y = y;
        for (i, item) in items.iter().enumerate() {
            let sel = i == self.cursor && item.is_interactive();
            let hovered = item.is_interactive()
                && hmy >= y && hmy < y + self.row_h
                && hmx >= ox && hmx < ox + pw;

            if hovered || sel {
                let bg = if sel { theme::SEL_BG } else { theme::HOVER_BG };
                rect(pixels, ww, wh, ox + self.cw, y, pw - self.cw * 2, self.row_h, bg);
            }
            let bg = if sel { theme::SEL_BG }
                else if hovered { theme::HOVER_BG }
                else { theme::PANEL };

            match item {
                Widget::Section { title } => {
                    let sx = ox + self.cw * 2;
                    let bar = format!("-- {title} ");
                    stext(atlas, pixels, ww, wh, sx, y, &bar, theme::SECTION, theme::PANEL, s);
                }
                Widget::Separator => {
                    rect(pixels, ww, wh, ox + self.cw * 2, y + self.ch / 2,
                        pw - self.cw * 4, s.max(1), theme::SEP);
                }
                Widget::Info { text: t, color } => {
                    stext(atlas, pixels, ww, wh, label_x, y, t, *color, theme::PANEL, s);
                }
                Widget::Toggle { label, value, .. } => {
                    let arrow = if sel { ">>" } else { "  " };
                    stext(atlas, pixels, ww, wh, ox + self.cw, y, arrow, theme::CURSOR, bg, s);
                    let fg = if sel { theme::BRIGHT } else { theme::LABEL };
                    stext(atlas, pixels, ww, wh, label_x, y, label, fg, bg, s);
                    let (mark, mfg) = if *value { ("[x] ON", theme::ON) } else { ("[ ] --", theme::OFF) };
                    let vfg = if hovered { theme::VALUE_HI } else { mfg };
                    stext(atlas, pixels, ww, wh, val_x, y, mark, vfg, bg, s);
                }
                Widget::Choice { label, options, index, .. } => {
                    let arrow = if sel { ">>" } else { "  " };
                    stext(atlas, pixels, ww, wh, ox + self.cw, y, arrow, theme::CURSOR, bg, s);
                    let fg = if sel { theme::BRIGHT } else { theme::LABEL };
                    stext(atlas, pixels, ww, wh, label_x, y, label, fg, bg, s);
                    let val = format!("[< {} >]", options[*index]);
                    let vfg = if sel || hovered { theme::VALUE_HI } else { theme::VALUE };
                    stext(atlas, pixels, ww, wh, val_x, y, &val, vfg, bg, s);
                }
                Widget::DynamicChoice { label, options, index, .. } => {
                    let arrow = if sel { ">>" } else { "  " };
                    stext(atlas, pixels, ww, wh, ox + self.cw, y, arrow, theme::CURSOR, bg, s);
                    let fg = if sel { theme::BRIGHT } else { theme::LABEL };
                    stext(atlas, pixels, ww, wh, label_x, y, label, fg, bg, s);
                    // Truncate value to fit within panel
                    let avail_chars = ((ox + pw).saturating_sub(val_x) / self.cw)
                        .saturating_sub(1) as usize;
                    let display = if options.is_empty() {
                        "(none)".to_string()
                    } else {
                        let inner = &options[*index].0;
                        let chrome = 5; // "[< " + " >]"
                        let max_inner = avail_chars.saturating_sub(chrome);
                        if inner.len() > max_inner && max_inner > 2 {
                            format!("[< {}.. >]", &inner[..max_inner - 2])
                        } else {
                            format!("[< {} >]", inner)
                        }
                    };
                    let vfg = if sel || hovered { theme::VALUE_HI } else { theme::VALUE };
                    stext(atlas, pixels, ww, wh, val_x, y, &display, vfg, bg, s);
                }
                Widget::Button { label, .. } => {
                    let arrow = if sel { ">>" } else { "  " };
                    stext(atlas, pixels, ww, wh, ox + self.cw, y, arrow, theme::CURSOR, bg, s);
                    let text = format!("[ {} ]", label);
                    let fg = if sel { theme::ACCENT } else if hovered { theme::VALUE_HI }
                        else { theme::LABEL };
                    stext(atlas, pixels, ww, wh, label_x, y, &text, fg, bg, s);
                }
                Widget::Slider { label, value, min, max, .. } => {
                    let arrow = if sel { ">>" } else { "  " };
                    stext(atlas, pixels, ww, wh, ox + self.cw, y, arrow, theme::CURSOR, bg, s);
                    let fg = if sel { theme::BRIGHT } else { theme::LABEL };
                    stext(atlas, pixels, ww, wh, label_x, y, label, fg, bg, s);
                    // Visual slider track
                    let track_chars = 14u32;
                    let pct = ((*value - min) / (max - min)).clamp(0.0, 1.0);
                    let fill = (pct * track_chars as f64).round() as usize;
                    let empty = (track_chars as usize).saturating_sub(fill);
                    let track = format!("[{}|{}] {value:.1}",
                        "=".repeat(fill), "-".repeat(empty));
                    // Color the filled part blue, empty part gray
                    let tx = val_x;
                    // Simple: draw whole string, then overdraw filled part
                    stext(atlas, pixels, ww, wh, tx, y, &track, theme::SLIDER_BG, bg, s);
                    let filled_str: String = std::iter::once('[')
                        .chain(std::iter::repeat('=').take(fill))
                        .collect();
                    stext(atlas, pixels, ww, wh, tx, y, &filled_str, theme::SLIDER_FG, bg, s);
                    // Thumb
                    let thumb_x = tx + (fill as u32 + 1) * self.cw;
                    stext(atlas, pixels, ww, wh, thumb_x, y, "|", theme::SLIDER_TH, bg, s);
                    // Value text after track
                    let vx = tx + (track_chars + 3) * self.cw;
                    stext(atlas, pixels, ww, wh, vx, y, &format!("{value:.1}"), theme::VALUE, bg, s);
                }
            }
            y += self.row_h;
        }

        // Bottom separator
        y += s;
        rect(pixels, ww, wh, ox + self.cw, y + self.ch / 2, sep_w, s.max(1), theme::SEP);
        y += self.row_h / 2 + s;

        // Help
        let help = if self.pinky {
            "C-n/p nav | C-f/b adj | RET ok | C-g quit | M-1..4 tab"
        } else {
            "Arrows: nav | Enter: toggle | Tab: next tab | F8: apply"
        };
        stext(atlas, pixels, ww, wh, ox + self.cw, y, help, theme::HINT, theme::PANEL, s);
        y += self.row_h;

        // Flash
        if let Some(f) = &self.flash {
            if Instant::now() < f.until {
                stext(atlas, pixels, ww, wh, ox + self.cw, y, &f.text, f.color, theme::PANEL, s);
            }
        }
    }
    /// Populate the boot device dynamic choice from the filesystem list.
    ///
    /// # Arguments
    ///
    /// * `devices` - Vec of `(display_name, address)` pairs.
    /// * `current_addr` - The currently active boot address. The
    ///   selector will be positioned on this entry.
    pub fn set_boot_devices(&mut self, devices: Vec<(String, String)>, current_addr: &str) {
        for tab in &mut self.tabs {
            for w in &mut tab.items {
                if let Widget::DynamicChoice { key, options, index, .. } = w {
                    if *key == "boot_device" {
                        let sel = devices.iter()
                            .position(|(_, addr)| addr == current_addr)
                            .unwrap_or(0);
                        *options = devices;
                        *index = sel;
                        return; // devices moved, must exit
                    }
                }
            }
        }
    }

    /// Get the address of the currently selected boot device.
    ///
    /// Returns `None` if no boot devices are available.
    pub fn get_boot_device(&self) -> Option<&str> {
        for tab in &self.tabs {
            for w in &tab.items {
                if let Widget::DynamicChoice { key, options, index, .. } = w {
                    if *key == "boot_device" && !options.is_empty() {
                        return Some(&options[*index].1);
                    }
                }
            }
        }
        None
    }
}

// == Top-level state ==========================================================

pub struct OverlayState {
    pub settings: SettingsGui,
    pub debug_bar_visible: bool,
}

impl OverlayState {
    pub fn new() -> Self {
        Self { settings: SettingsGui::new(), debug_bar_visible: false }
    }
}

// == Debug metrics ============================================================

pub struct DebugMetrics {
    pub gpu_name: String,
    pub present_mode: String,
    pub fps: u32,
    pub frame_time_us: u64,
    pub swapchain_extent: (u32, u32),
    pub render_mode: RenderMode,
    pub uptime_s: f64,
    pub signal_queue_len: usize,
    pub buffer_resolution: (u32, u32),
    pub lua_steps: u32,
}

impl Default for DebugMetrics {
    fn default() -> Self {
        Self {
            gpu_name: String::new(), present_mode: String::new(),
            fps: 0, frame_time_us: 0, swapchain_extent: (0, 0),
            render_mode: RenderMode::Software, uptime_s: 0.0,
            signal_queue_len: 0, buffer_resolution: (80, 25), lua_steps: 0,
        }
    }
}

pub fn build_debug_cells(m: &DebugMetrics, max_cols: u32) -> Vec<[u32; 3]> {
    let s = format!(
        " {} | {} fps | {:.2}ms | {}: {}x{} {} | buf {}x{} | sig {} | lua {} | up {:.1}s ",
        m.render_mode.label(), m.fps, m.frame_time_us as f64 / 1000.0,
        if m.gpu_name.is_empty() { "CPU" } else { &m.gpu_name },
        m.swapchain_extent.0, m.swapchain_extent.1, m.present_mode,
        m.buffer_resolution.0, m.buffer_resolution.1,
        m.signal_queue_len, m.lua_steps, m.uptime_s);
    let n = max_cols as usize;
    let mut c = Vec::with_capacity(n);
    for ch in s.chars().take(n) { c.push([ch as u32, theme::BAR_FG, theme::BAR_BG]); }
    while c.len() < n { c.push([' ' as u32, theme::BAR_FG, theme::BAR_BG]); }
    c
}

pub fn render_debug_bar_sw(m: &DebugMetrics, a: &GlyphAtlas, px: &mut [u32], ww: u32, wh: u32) {
    let cols = (ww / 8).min(512);
    let cells = build_debug_cells(m, cols);
    for py in 0..CELL_H.min(wh) {
        let row = (py * ww) as usize;
        for x in 0..(ww as usize) { if row + x < px.len() { px[row + x] = theme::BAR_BG; } }
    }
    for (i, c) in cells.iter().enumerate() {
        let x = (i as u32) * 8;
        if x + 8 > ww { break; }
        glyph(a, px, ww, wh, x, 0, c[0], c[1], c[2]);
    }
}

// == Drawing with scale =======================================================

fn glyph(atlas: &GlyphAtlas, px: &mut [u32], ww: u32, wh: u32,
          x: u32, y: u32, cp: u32, fg: u32, bg: u32) {
    sglyph(atlas, px, ww, wh, x, y, cp, fg, bg, 1);
}

/// Scaled glyph: each font pixel becomes `s x s` screen pixels.
fn sglyph(atlas: &GlyphAtlas, px: &mut [u32], ww: u32, wh: u32,
           x: u32, y: u32, cp: u32, fg: u32, bg: u32, s: u32) {
    let gw = if (cp as usize) < atlas.widths.len() && atlas.widths[cp as usize] > 8 { 16u32 } else { 8 };
    let ac = cp & 0xFF;
    let ar = (cp >> 8) & 0xFF;
    for gy in 0..CELL_H {
        for gx in 0..gw {
            let ax = (ac * CELL_W + gx) as usize;
            let ay = (ar * CELL_H + gy) as usize;
            let hit = ax < ATLAS_SIZE as usize && ay < ATLAS_SIZE as usize
                && atlas.pixels[ay * ATLAS_SIZE as usize + ax] > 128;
            let color = if hit { fg } else { bg };
            for sy in 0..s {
                for sx in 0..s {
                    let px_x = x + gx * s + sx;
                    let px_y = y + gy * s + sy;
                    if px_x < ww && px_y < wh {
                        let idx = (px_y * ww + px_x) as usize;
                        if idx < px.len() { px[idx] = color; }
                    }
                }
            }
        }
    }
}

/// Draw a string at pixel position with scale. Returns X after last char.
fn stext(atlas: &GlyphAtlas, px: &mut [u32], ww: u32, wh: u32,
          mut x: u32, y: u32, text: &str, fg: u32, bg: u32, s: u32) -> u32 {
    let cw = 8 * s;
    for ch in text.chars() {
        if x + cw > ww { break; }
        sglyph(atlas, px, ww, wh, x, y, ch as u32, fg, bg, s);
        x += cw;
    }
    x
}

fn rect(px: &mut [u32], ww: u32, wh: u32, x: u32, y: u32, w: u32, h: u32, c: u32) {
    for ry in y..y.saturating_add(h).min(wh) {
        let off = (ry * ww) as usize;
        for rx in x..x.saturating_add(w).min(ww) {
            let idx = off + rx as usize;
            if idx < px.len() { px[idx] = c; }
        }
    }
}