//! # Virtual filesystem
//!
//! Provides an in-memory filesystem implementation used for:
//!
//! * **ROM** (read-only): Loaded from `assets/openos/` at startup.
//!   Contains the OpenOS distribution files.
//! * **tmpfs** (`/tmp`): Volatile, cleared on reboot. Used by programs
//!   for temporary storage.
//! * **Writable disks**: Persisted to the host filesystem via
//!   [`save_to_directory`](VirtualFs::save_to_directory) and
//!   [`load_from_directory`](VirtualFs::load_from_directory).
//!
//! ## Design
//!
//! The virtual filesystem stores file contents as `HashMap<String, Vec<u8>>`
//! where keys are normalised paths (no leading `/`). Directories are
//! **implicit**: a directory exists if any file has it as a path prefix.
//! There are no empty directories.
//!
//! This approach is simple and matches how OC's `api.fs.FileSystem`
//! interface works: there is no notion of inodes, permissions, or
//! timestamps (except `lastModified`, which we stub as 0).
//!
//! ## Explicit directories
//!
//! While directories are primarily implicit (derived from file path
//! prefixes), `makeDirectory` creates **explicit** empty directories
//! stored in a separate `HashSet`. These explicit entries ensure that:
//!
//! * `exists("newdir")` returns `true` immediately after `makeDirectory`.
//! * `isDirectory("newdir")` returns `true` even with no files inside.
//! * `list("/")` includes `"newdir/"` in its results.
//! * `remove("newdir")` can delete an empty directory.
//!
//! When a file is created under an explicit directory, the explicit
//! entry becomes redundant (the directory is now also implicit), but
//! it is kept for simplicity. Explicit entries are removed by `delete`.
//!
//! ## Host-directory persistence
//!
//! Writable filesystems can be persisted to a host directory using
//! [`save_to_directory`](VirtualFs::save_to_directory) and restored
//! using [`load_from_directory`](VirtualFs::load_from_directory).
//! The host directory mirrors the VFS structure exactly: each VFS file
//! becomes a file on the host at the same relative path.
//!
//! A special file `.vfs_meta` inside the save directory is used by
//! [`FilesystemComponent`](crate::components::filesystem::FilesystemComponent)
//! to persist component metadata (address, label). This file is
//! automatically excluded when loading a VFS from a host directory.
//!
//! ## Path normalisation
//!
//! All paths go through [`VirtualFs::normalize`] before use:
//!
//! * Leading `/` is stripped: `/foo/bar` -> `foo/bar`
//! * Double slashes are collapsed: `foo//bar` -> `foo/bar`
//! * `.` components are removed: `foo/./bar` -> `foo/bar`
//! * `..` components are resolved: `foo/bar/../baz` -> `foo/baz`
//!
//! ## Capacity enforcement
//!
//! Each filesystem has an optional capacity (in bytes). When `capacity > 0`,
//! write operations check that the total size of all files does not
//! exceed the capacity. If a write would exceed it, the operation fails.
//!
//! ## Open modes
//!
//! The [`OpenMode`] enum defines three modes matching the C/Lua
//! conventions:
//!
//! * `Read` (`"r"`) - File must exist. Read-only access.
//! * `Write` (`"w"`) - File is created/truncated. Write-only access.
//! * `Append` (`"a"`) - File is created if absent, not truncated.
//!   Writes go to the end.
//!
//! ## Thread safety
//!
//! `VirtualFs` is not `Sync`. It is accessed only from the main
//! emulation thread (behind the `FilesystemComponent` wrapper).

use std::collections::{HashMap, HashSet};

/// File open mode.
///
/// Matches the mode strings used by `io.open()` in Lua and
/// `filesystem.open()` in OC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// Read-only access. File must exist.
    ///
    /// Lua mode strings: `"r"`, `"rb"`.
    Read,

    /// Write-only access. File is created or truncated.
    ///
    /// Lua mode strings: `"w"`, `"wb"`.
    Write,

    /// Write-only access. File is created if absent, not truncated.
    /// Writes go to the end of the file.
    ///
    /// Lua mode strings: `"a"`, `"ab"`.
    Append,
}

/// A handle to an open file.
///
/// This struct is defined here but is NOT used by `VirtualFs` directly;
/// it is used by [`FilesystemComponent`](crate::components::filesystem::FilesystemComponent).
/// It is included in this module for proximity to [`OpenMode`].
#[derive(Debug)]
pub struct FileHandle {
    /// Normalised path to the open file.
    file_path: String,

    /// Current read/write cursor position within the file.
    position: u64,

    /// Mode this handle was opened in.
    mode: OpenMode,
}

/// An in-memory virtual filesystem.
///
/// Stores file contents in a `HashMap<String, Vec<u8>>` keyed by
/// normalised path. Directories are implicit (derived from file path
/// prefixes) but can also be explicitly created via `make_directory`.
///
/// # Capacity
///
/// If `capacity > 0`, the total size of all files is limited.
/// If `capacity == 0`, the filesystem is unlimited (used for ROM and
/// tmpfs-like filesystems).
///
/// # Read-only mode
///
/// If `read_only` is `true`, all mutation methods (write, delete,
/// rename, mkdir, set_label) return `false` or are no-ops.
#[derive(Debug, Clone)]
pub struct VirtualFs {
    /// File contents, keyed by normalised path.
    ///
    /// Keys never have a leading `/`. Example: `"init.lua"`,
    /// `"lib/term.lua"`, `"bin/ls.lua"`.
    files: HashMap<String, Vec<u8>>,

    /// Explicitly created directories.
    ///
    /// Directories are primarily implicit (they exist if any file has
    /// them as a path prefix). However, `makeDirectory` can create
    /// empty directories that have no files yet. These are stored here
    /// so that `exists`, `isDirectory`, and `list` report them correctly.
    ///
    /// Entries are normalised paths without trailing `/`.
    /// Example: `"foo"`, `"foo/bar"`.
    dirs: HashSet<String>,

    /// Whether this filesystem rejects all writes.
    read_only: bool,

    /// Maximum total capacity in bytes. 0 means unlimited.
    capacity: u64,

    /// Human-readable label (e.g. `"openos"`).
    label: Option<String>,
}

impl VirtualFs {
    /// Create a new empty filesystem.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum total bytes. 0 for unlimited.
    /// * `read_only` - Whether to reject all writes.
    pub fn new(capacity: u64, read_only: bool) -> Self {
        Self {
            files: HashMap::new(),
            dirs: HashSet::new(),
            read_only,
            capacity,
            label: None,
        }
    }

    /// Create a read-only filesystem pre-loaded with the given files.
    ///
    /// Used for loading the OpenOS ROM from `assets/openos/`.
    ///
    /// # Arguments
    ///
    /// * `files` - Map from normalised path to file contents.
    pub fn from_files(files: HashMap<String, Vec<u8>>) -> Self {
        Self {
            files,
            dirs: HashSet::new(),
            read_only: true,
            capacity: 0,
            label: None,
        }
    }

    // -------------------------------------------------------------------
    // Label
    // -------------------------------------------------------------------

    /// Get the filesystem label (if any).
    pub fn label(&self) -> Option<&str> { self.label.as_deref() }

    /// Set the filesystem label.
    ///
    /// No-op on read-only filesystems.
    pub fn set_label(&mut self, label: Option<String>) {
        if !self.read_only { self.label = label; }
    }

    // -------------------------------------------------------------------
    // Capacity
    // -------------------------------------------------------------------

    /// Whether this filesystem is read-only.
    pub fn is_read_only(&self) -> bool { self.read_only }

    /// Total capacity in bytes. 0 means unlimited.
    pub fn space_total(&self) -> u64 { self.capacity }

    /// Currently used space in bytes (sum of all file sizes).
    pub fn space_used(&self) -> u64 {
        self.files.values().map(|v| v.len() as u64).sum()
    }

    // -------------------------------------------------------------------
    // Path operations
    // -------------------------------------------------------------------

    /// Normalize a path: strip leading `/`, collapse `//`, resolve
    /// `.` and `..` components.
    ///
    /// # Examples
    ///
    /// ```text
    /// normalize("/foo/bar")      -> "foo/bar"
    /// normalize("foo//bar")      -> "foo/bar"
    /// normalize("foo/./bar")     -> "foo/bar"
    /// normalize("foo/bar/../baz") -> "foo/baz"
    /// normalize("/")             -> ""
    /// normalize("")              -> ""
    /// ```
    ///
    /// # Returns
    ///
    /// A normalised path with no leading `/` and no `.`/`..` components.
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

    /// Check if a path exists (as a file or directory).
    ///
    /// The root path (`""` after normalisation) always exists.
    /// A path exists if it is a file, an explicit directory, or an
    /// implicit directory (prefix of at least one file path).
    pub fn exists(&self, path: &str) -> bool {
        let p = Self::normalize(path);
        if p.is_empty() { return true; }
        if self.files.contains_key(&p) { return true; }
        if self.dirs.contains(&p) { return true; }
        let prefix = format!("{}/", p);
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    /// Check if a path is a directory (not a file).
    ///
    /// A path is a directory if:
    /// * It is the root (`""` after normalisation), OR
    /// * It is NOT a file AND (is an explicit directory OR is a prefix
    ///   of at least one file path).
    pub fn is_directory(&self, path: &str) -> bool {
        let p = Self::normalize(path);
        if p.is_empty() { return true; }
        if self.files.contains_key(&p) { return false; }
        if self.dirs.contains(&p) { return true; }
        let prefix = format!("{}/", p);
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    /// Get the size of a file in bytes.
    ///
    /// Returns 0 for directories and non-existent paths.
    pub fn size(&self, path: &str) -> u64 {
        let p = Self::normalize(path);
        self.files.get(&p).map(|v| v.len() as u64).unwrap_or(0)
    }

    /// List entries in a directory.
    ///
    /// # Returns
    ///
    /// * `Some(entries)` - Sorted list of entry names. Subdirectories
    ///   have a trailing `/` (e.g. `"subdir/"`), files do not.
    /// * `None` - Path does not exist or is not a directory.
    pub fn list(&self, path: &str) -> Option<Vec<String>> {
        let p = Self::normalize(path);
        let prefix = if p.is_empty() { String::new() } else { format!("{}/", p) };

        if !p.is_empty() && !self.is_directory(path) {
            return None;
        }

        let mut entries = std::collections::BTreeSet::new();

        // Entries from files
        for key in self.files.keys() {
            if let Some(rest) = key.strip_prefix(&prefix) {
                if let Some(idx) = rest.find('/') {
                    entries.insert(format!("{}/", &rest[..idx]));
                } else if !rest.is_empty() {
                    entries.insert(rest.to_string());
                }
            }
        }

        // Entries from explicit directories
        for dir in &self.dirs {
            if let Some(rest) = dir.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    entries.insert(format!("{}/", rest));
                }
            } else if prefix.is_empty() && !dir.contains('/') && !dir.is_empty() {
                entries.insert(format!("{}/", dir));
            }
        }

        Some(entries.into_iter().collect())
    }

    // -------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------

    /// Create a directory.
    ///
    /// Corresponds to `filesystem.makeDirectory(path)` in Lua.
    ///
    /// Creates an explicit directory entry so that `exists()`,
    /// `isDirectory()`, and `list()` report it even when empty.
    /// Parent directories along the path are also created implicitly.
    ///
    /// # Returns
    ///
    /// `true` if the directory was created (or already exists as a
    /// directory). `false` if the filesystem is read-only or a file
    /// exists at that path.
    pub fn make_directory(&mut self, path: &str) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);
        if p.is_empty() { return true; }
        if self.files.contains_key(&p) { return false; }

        self.dirs.insert(p.clone());
        // Also create parent directories
        let mut current = p.as_str();
        while let Some(idx) = current.rfind('/') {
            current = &current[..idx];
            if !current.is_empty() {
                self.dirs.insert(current.to_string());
            }
        }
        true
    }

    /// Delete a file or directory (recursively).
    ///
    /// If the path is a file, deletes that file.
    /// If the path is a directory prefix, deletes all files under it
    /// and any explicit directory entries under it.
    ///
    /// # Returns
    ///
    /// `true` if anything was deleted.
    pub fn delete(&mut self, path: &str) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);
        let mut deleted = false;

        if self.files.remove(&p).is_some() { deleted = true; }
        if self.dirs.remove(&p) { deleted = true; }

        let prefix = format!("{}/", p);
        let file_keys: Vec<String> = self.files.keys()
            .filter(|k| k.starts_with(&prefix)).cloned().collect();
        if !file_keys.is_empty() { deleted = true; }
        for k in file_keys { self.files.remove(&k); }

        let dir_keys: Vec<String> = self.dirs.iter()
            .filter(|k| k.starts_with(&prefix)).cloned().collect();
        if !dir_keys.is_empty() { deleted = true; }
        for k in dir_keys { self.dirs.remove(&k); }

        deleted
    }

    /// Rename (move) a file.
    ///
    /// # Returns
    ///
    /// `true` if the file was found and renamed.
    /// `false` if read-only or the source does not exist.
    ///
    /// # Note
    ///
    /// Does not handle renaming directories (only single files).
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

    // -------------------------------------------------------------------
    // Read / Write
    // -------------------------------------------------------------------

    /// Read entire file contents.
    ///
    /// # Returns
    ///
    /// * `Some(bytes)` - The file contents as a byte slice.
    /// * `None` - File does not exist.
    pub fn read_file(&self, path: &str) -> Option<&[u8]> {
        let p = Self::normalize(path);
        self.files.get(&p).map(|v| v.as_slice())
    }

    /// Write entire file contents (overwrites any existing data).
    ///
    /// # Returns
    ///
    /// `true` if the write succeeded.
    /// `false` if read-only or if the write would exceed capacity.
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

    /// Append data to a file (creates the file if it doesn't exist).
    ///
    /// # Returns
    ///
    /// `true` if the append succeeded.
    /// `false` if read-only or if the append would exceed capacity.
    pub fn append_file(&mut self, path: &str, data: &[u8]) -> bool {
        if self.read_only { return false; }
        let p = Self::normalize(path);

        if self.capacity > 0 {
            let new_total = self.space_used() + data.len() as u64;
            if new_total > self.capacity { return false; }
        }

        self.files
            .entry(p)
            .or_insert_with(Vec::new)
            .extend_from_slice(data);
        true
    }

    // -------------------------------------------------------------------
    // Host directory persistence
    // -------------------------------------------------------------------

    /// Save all files and explicit directories to a host directory.
    ///
    /// The directory structure on the host mirrors the VFS exactly:
    /// each file in the VFS becomes a file on the host at the same
    /// relative path. Explicit empty directories are created as empty
    /// directories on the host.
    ///
    /// **Warning:** any existing contents in `base` are removed first
    /// to ensure a clean snapshot.
    ///
    /// # Arguments
    ///
    /// * `base` - The host directory to save into. Created if absent.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` on any host filesystem failure.
    pub fn save_to_directory(&self, base: &std::path::Path) -> std::io::Result<()> {
        // Clear and recreate to ensure a clean snapshot
        if base.exists() {
            std::fs::remove_dir_all(base)?;
        }
        std::fs::create_dir_all(base)?;

        // Write all files
        for (path, data) in &self.files {
            let full = base.join(path);
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full, data)?;
        }

        // Create explicit empty directories
        for dir in &self.dirs {
            std::fs::create_dir_all(base.join(dir))?;
        }

        Ok(())
    }

    /// Load files from a host directory into this VFS, replacing all
    /// existing contents.
    ///
    /// Recursively walks the host directory and loads every file and
    /// subdirectory. Files named `.vfs_meta` are skipped (they hold
    /// component metadata used by
    /// [`FilesystemComponent`](crate::components::filesystem::FilesystemComponent),
    /// not VFS content).
    ///
    /// # Arguments
    ///
    /// * `base` - The host directory to load from. If it does not
    ///   exist, this is a silent no-op.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` on any host filesystem failure.
    pub fn load_from_directory(&mut self, base: &std::path::Path) -> std::io::Result<()> {
        if !base.is_dir() { return Ok(()); }
        self.files.clear();
        self.dirs.clear();
        self.walk_host_dir(base, base)
    }

    /// Recursively walk a host directory, loading files and tracking
    /// subdirectories.
    fn walk_host_dir(
        &mut self,
        base: &std::path::Path,
        current: &std::path::Path,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(current)? {
            let entry = entry?;
            let path = entry.path();
            let rel = path.strip_prefix(base)
                .map_err(|e| std::io::Error::new(
                    std::io::ErrorKind::Other, e.to_string(),
                ))?
                .to_string_lossy()
                .replace('\\', "/");

            // Skip component metadata file
            if rel == ".vfs_meta" || rel.ends_with("/.vfs_meta") {
                continue;
            }

            if path.is_dir() {
                self.dirs.insert(rel);
                self.walk_host_dir(base, &path)?;
            } else {
                let data = std::fs::read(&path)?;
                self.files.insert(rel, data);
            }
        }
        Ok(())
    }
}