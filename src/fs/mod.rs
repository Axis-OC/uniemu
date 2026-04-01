//! # Virtual filesystem
//!
//! Provides an in-memory filesystem used for:
//! - The ROM (read-only, loaded from assets)
//! - `/tmp` (volatile, cleared on reboot)
//! - Writable disks (persisted to the host filesystem)
//!
//! The API mirrors OC's `api.fs.FileSystem` interface.

use std::collections::HashMap;

/// File open mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    Read,
    Write,
    Append,
}

/// A handle to an open file.
#[derive(Debug)]
pub struct FileHandle {
    /// Index into the parent filesystem's file storage.
    file_path: String,
    /// Current read/write cursor position.
    position: u64,
    /// Mode this handle was opened in.
    mode: OpenMode,
}

/// An in-memory virtual filesystem.
///
/// Directories are implicit / they exist if any file has them as a prefix.
/// File contents are stored as `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct VirtualFs {
    /// File contents, keyed by normalized path (no leading `/`).
    files: HashMap<String, Vec<u8>>,
    /// Whether this filesystem is read-only.
    read_only: bool,
    /// Maximum total capacity in bytes (0 = unlimited).
    capacity: u64,
    /// Human-readable label.
    label: Option<String>,
}

impl VirtualFs {
    /// Create a new empty filesystem.
    pub fn new(capacity: u64, read_only: bool) -> Self {
        Self {
            files: HashMap::new(),
            read_only,
            capacity,
            label: None,
        }
    }

    /// Create a read-only filesystem pre-loaded with the given files.
    pub fn from_files(files: HashMap<String, Vec<u8>>) -> Self {
        Self {
            files,
            read_only: true,
            capacity: 0,
            label: None,
        }
    }

    // ── Label ───────────────────────────────────────────────────────────

    pub fn label(&self) -> Option<&str> { self.label.as_deref() }

    pub fn set_label(&mut self, label: Option<String>) {
        if !self.read_only { self.label = label; }
    }

    // ── Capacity ────────────────────────────────────────────────────────

    pub fn is_read_only(&self) -> bool { self.read_only }

    pub fn space_total(&self) -> u64 { self.capacity }

    pub fn space_used(&self) -> u64 {
        self.files.values().map(|v| v.len() as u64).sum()
    }

    // ── Path operations ─────────────────────────────────────────────────

    /// Normalize a path: strip leading `/`, collapse `//`, resolve `.`/`..`.
    pub fn normalize(path: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => { parts.pop(); }
                p => parts.push(p),
            }
        }
        parts.join("/")
    }

    pub fn exists(&self, path: &str) -> bool {
        let p = Self::normalize(path);
        if p.is_empty() { return true; } // root always exists
        // Check exact file match.
        if self.files.contains_key(&p) { return true; }
        // Check if it's a directory prefix.
        let prefix = format!("{}/", p);
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    pub fn is_directory(&self, path: &str) -> bool {
        let p = Self::normalize(path);
        if p.is_empty() { return true; }
        if self.files.contains_key(&p) { return false; }
        let prefix = format!("{}/", p);
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    pub fn size(&self, path: &str) -> u64 {
        let p = Self::normalize(path);
        self.files.get(&p).map(|v| v.len() as u64).unwrap_or(0)
    }

    /// List entries in a directory. Returns `None` if not a directory.
    pub fn list(&self, path: &str) -> Option<Vec<String>> {
        let p = Self::normalize(path);
        let prefix = if p.is_empty() { String::new() } else { format!("{}/", p) };

        if !p.is_empty() && !self.is_directory(path) {
            return None;
        }

        let mut entries = std::collections::BTreeSet::new();
        for key in self.files.keys() {
            if let Some(rest) = key.strip_prefix(&prefix) {
                if let Some(idx) = rest.find('/') {
                    entries.insert(format!("{}/", &rest[..idx]));
                } else if !rest.is_empty() {
                    entries.insert(rest.to_string());
                }
            }
        }
        Some(entries.into_iter().collect())
    }

    // ── Mutation ────────────────────────────────────────────────────────

    pub fn make_directory(&mut self, path: &str) -> bool {
        // Directories are implicit. We create a marker if needed.
        // Actually, in OC virtual FS, directories exist implicitly.
        // Just return true if not read-only and path doesn't exist as a file.
        if self.read_only { return false; }
        let p = Self::normalize(path);
        !self.files.contains_key(&p)
    }

    pub fn delete(&mut self, path: &str) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);
        if self.files.remove(&p).is_some() {
            return true;
        }
        // Delete directory (all files with this prefix).
        let prefix = format!("{}/", p);
        let keys: Vec<String> = self.files.keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        if keys.is_empty() { return false; }
        for k in keys { self.files.remove(&k); }
        true
    }

    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        if self.read_only { return false; }
        let f = Self::normalize(from);
        let t = Self::normalize(to);
        if let Some(data) = self.files.remove(&f) {
            self.files.insert(t, data);
            true
        } else {
            false
        }
    }

    // ── Read / Write ────────────────────────────────────────────────────

    /// Read entire file contents.
    pub fn read_file(&self, path: &str) -> Option<&[u8]> {
        let p = Self::normalize(path);
        self.files.get(&p).map(|v| v.as_slice())
    }

    /// Write entire file contents (overwrites).
    pub fn write_file(&mut self, path: &str, data: Vec<u8>) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);
        if self.capacity > 0 {
            let current_used = self.space_used();
            let old_size = self.files.get(&p).map(|v| v.len() as u64).unwrap_or(0);
            let new_total = current_used - old_size + data.len() as u64;
            if new_total > self.capacity { return false; }
        }
        self.files.insert(p, data);
        true
    }

    /// Append data to a file (creates if not exists).
    pub fn append_file(&mut self, path: &str, data: &[u8]) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);

        // Check capacity BEFORE borrowing `self.files` mutably.
        if self.capacity > 0 {
            let new_total = self.space_used() + data.len() as u64;
            if new_total > self.capacity { return false; }
        }

        // Now safe to mutably borrow `files`.
        self.files
            .entry(p)
            .or_insert_with(Vec::new)
            .extend_from_slice(data);
        true
    }
}