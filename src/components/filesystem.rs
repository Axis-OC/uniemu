//! # Filesystem Component
//!
//! This module wraps a [`VirtualFs`] (in-memory filesystem) and exposes
//! it as an OpenComputers `filesystem` component, complete with integer
//! file descriptors for open/read/write/seek/close operations.
//!
//! ## Relationship to OpenComputers
//!
//! This mirrors `li.cil.oc.server.component.FileSystem` from the OC
//! Scala source. In the original mod:
//!
//! * Each hard drive, floppy disk, or the ROM (OpenOS loot disk) is
//!   represented as a `filesystem` component.
//! * The component exposes methods like `open`, `read`, `write`,
//!   `close`, `seek`, `list`, `exists`, `isDirectory`, `size`,
//!   `makeDirectory`, `remove`, `rename`, `getLabel`, `setLabel`,
//!   `isReadOnly`, `spaceTotal`, `spaceUsed`, and `lastModified`.
//! * File handles are integer IDs (not userdata), limited to
//!   `maxHandles` per filesystem per machine.
//! * Read operations return at most `maxReadBuffer` bytes per call.
//!
//! ## Handle management
//!
//! Each call to [`open`](FilesystemComponent::open) allocates a new
//! integer handle ID (monotonically increasing from 1). The handle
//! tracks:
//!
//! * The normalised file path
//! * The current read/write cursor position
//! * The open mode (read, write, or append)
//!
//! Handles are stored in a `HashMap<i32, OpenHandle>`. When the machine
//! stops or reboots, [`close_all`](FilesystemComponent::close_all) is
//! called to release all handles.
//!
//! ## Data flow for a read operation
//!
//! ```text
//! Lua: component.invoke(fs_addr, "read", handle, count)
//!   |
//!   v
//! host.rs -> dispatch_filesystem() -> fs.read(handle, count)
//!   |
//!   v
//! FilesystemComponent::read()
//!   |
//!   | 1. Look up handle in self.handles
//!   | 2. Verify mode is Read
//!   | 3. Read file bytes from VirtualFs
//!   | 4. Slice [position..position+count]
//!   | 5. Advance position
//!   | 6. Return Ok(Some(chunk)) or Ok(None) on EOF
//!   v
//! host.rs pushes bytes onto Lua stack
//! ```
//!
//! ## Path normalisation
//!
//! All paths are normalised via [`VirtualFs::normalize`] before use:
//! leading slashes are stripped, `//` is collapsed, `.` and `..` are
//! resolved. This means `/foo/../bar` becomes `bar`.
//!
//! ## Persistence
//!
//! Writable filesystems can be persisted to a host directory by setting
//! [`save_path`](FilesystemComponent::save_path) and calling
//! [`save_to_disk`](FilesystemComponent::save_to_disk). The component
//! address and label are stored in a `.vfs_meta` file alongside the
//! VFS data, so that the address survives restarts (important because
//! the BIOS stores the boot filesystem address in EEPROM data).

use crate::components::Address;
use crate::fs::{VirtualFs, OpenMode};
use std::collections::HashMap;

/// Internal state for a single open file handle.
///
/// This is not exposed to Lua; Lua only sees the integer handle ID.
/// The host translates between IDs and this struct via the `handles`
/// map in [`FilesystemComponent`].
struct OpenHandle {
    /// Normalised path to the open file (no leading `/`).
    ///
    /// Stored as an owned `String` because the file might be renamed
    /// or deleted while the handle is open (in which case subsequent
    /// operations will fail gracefully).
    path: String,

    /// Current byte offset within the file.
    ///
    /// For read mode: advances with each `read()` call.
    /// For write mode: always equals the file length (append-only).
    /// For append mode: starts at the end of the file and advances.
    ///
    /// Can be moved arbitrarily via `seek()` (read mode only).
    position: u64,

    /// The mode this handle was opened in.
    ///
    /// Determines which operations are permitted:
    /// * `Read` -> `read()` and `seek()` allowed
    /// * `Write` -> `write()` allowed (file was truncated on open)
    /// * `Append` -> `write()` allowed (file was not truncated)
    mode: OpenMode,
}

/// Filesystem component -- mirrors `FileSystem.scala`.
///
/// Owns a [`VirtualFs`] instance and a table of open file handles.
/// Each emulated hard drive, floppy, or ROM is one `FilesystemComponent`.
///
/// # Handle ID allocation
///
/// Handle IDs are allocated from a simple counter (`next_handle`),
/// starting at 1 and incrementing for each `open()` call. IDs are
/// never reused within a single session. After a reboot (which calls
/// `close_all()`), the counter is *not* reset (matching OC behaviour
/// where handle IDs are per-filesystem, not per-session).
///
/// # Label
///
/// The optional `label` field is the user-visible name of the
/// filesystem (e.g. `"openos"` for the OpenOS loot disk). It can be
/// up to 16 characters, set via `setLabel`, and is `None` for
/// unlabelled drives.
pub struct FilesystemComponent {
    /// Unique UUID address of this component.
    ///
    /// Used by Lua code in `component.invoke(address, ...)` calls
    /// and in `component.list("filesystem")` results.
    pub address: Address,

    /// Optional human-readable label.
    ///
    /// Displayed in the Lua `filesystem.getLabel()` return value and
    /// in the tooltip of the item in-game. Truncated to 16 characters
    /// on write. `None` if no label has been set.
    pub label: Option<String>,

    /// Optional path to a host directory for persistence.
    ///
    /// When set, [`save_to_disk`](FilesystemComponent::save_to_disk)
    /// writes all VFS contents to this directory, and
    /// [`load_from_disk`](FilesystemComponent::load_from_disk) restores
    /// them (including the component address and label from a
    /// `.vfs_meta` metadata file).
    ///
    /// Example: `"save/hdd"`.
    ///
    /// `None` for filesystems that do not need persistence (ROM, tmpfs).
    pub save_path: Option<String>,

    /// The underlying in-memory filesystem.
    ///
    /// All file content, directory structure, capacity, and read-only
    /// status live here. The component layer adds handle management
    /// on top.
    fs: VirtualFs,

    /// Map from integer handle ID to open handle state.
    ///
    /// Handles are inserted by `open()` and removed by `close()` or
    /// `close_all()`.
    handles: HashMap<i32, OpenHandle>,

    /// Next handle ID to allocate.
    ///
    /// Starts at 1 and increments monotonically. Never reused.
    next_handle: i32,
}

impl FilesystemComponent {
    /// Create a new filesystem component wrapping the given [`VirtualFs`].
    ///
    /// # Arguments
    ///
    /// * `fs` - The virtual filesystem to wrap. Can be read-only (ROM),
    ///   capacity-limited (hard drive), or unlimited (tmpfs).
    ///
    /// # Returns
    ///
    /// A `FilesystemComponent` with a fresh UUID address, no label,
    /// no save path, no open handles, and handle counter starting at 1.
    pub fn new(fs: VirtualFs) -> Self {
        Self {
            address: crate::components::new_address(),
            label: None,
            save_path: None,
            fs,
            handles: HashMap::new(),
            next_handle: 1,
        }
    }

    /// Returns the OC component type name: `"filesystem"`.
    ///
    /// Used in `component.list()` filtering and `component.type()` results.
    pub const fn component_name() -> &'static str { "filesystem" }

    /// Immutable reference to the underlying virtual filesystem.
    ///
    /// Useful for reading filesystem metadata (capacity, label, etc.)
    /// without going through the handle-based I/O interface.
    pub fn fs(&self) -> &VirtualFs { &self.fs }

    /// Mutable reference to the underlying virtual filesystem.
    ///
    /// Needed for operations that modify the filesystem outside of
    /// the handle-based interface (e.g. pre-loading files during
    /// emulator setup).
    pub fn fs_mut(&mut self) -> &mut VirtualFs { &mut self.fs }

    // -------------------------------------------------------------------
    // Persistence
    // -------------------------------------------------------------------

    /// Save the filesystem contents and component metadata to the host
    /// directory specified by [`save_path`](FilesystemComponent::save_path).
    ///
    /// The VFS contents are written by
    /// [`VirtualFs::save_to_directory`]. A `.vfs_meta` file is created
    /// alongside the VFS data containing the component address and
    /// label, so that they survive application restarts.
    ///
    /// No-op if `save_path` is `None` or if the filesystem is read-only.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` on any host filesystem failure.
    pub fn save_to_disk(&self) -> std::io::Result<()> {
        let path = match &self.save_path {
            Some(p) => p,
            None => return Ok(()),
        };
        if self.fs.is_read_only() { return Ok(()); }

        let base = std::path::Path::new(path);
        self.fs.save_to_directory(base)?;

        // Write component metadata
        let meta_path = base.join(".vfs_meta");
        let mut meta = format!("address={}\n", self.address);
        if let Some(label) = &self.label {
            meta.push_str(&format!("label={}\n", label));
        }
        std::fs::write(meta_path, meta)?;

        eprintln!("[fs] Saved {} to {}", self.address, path);
        Ok(())
    }

    /// Load filesystem contents and component metadata from the host
    /// directory specified by [`save_path`](FilesystemComponent::save_path).
    ///
    /// Restores the component address and label from the `.vfs_meta`
    /// file if it exists. This is important because the BIOS stores
    /// the boot filesystem address in EEPROM data -- if the address
    /// changed on every restart, the BIOS would not find the disk.
    ///
    /// The VFS contents are loaded by
    /// [`VirtualFs::load_from_directory`].
    ///
    /// No-op if `save_path` is `None` or if the directory does not exist.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` on any host filesystem failure.
    pub fn load_from_disk(&mut self) -> std::io::Result<()> {
        let path = match &self.save_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        let base = std::path::Path::new(&path);
        if !base.is_dir() { return Ok(()); }

        // Load component metadata
        let meta_path = base.join(".vfs_meta");
        if meta_path.exists() {
            let meta = std::fs::read_to_string(&meta_path)?;
            for line in meta.lines() {
                if let Some(addr) = line.strip_prefix("address=") {
                    self.address = addr.trim().to_string();
                } else if let Some(label) = line.strip_prefix("label=") {
                    self.label = Some(label.trim().to_string());
                }
            }
        }

        // Load VFS contents
        self.fs.load_from_directory(base)?;

        eprintln!("[fs] Loaded {} from {} ({} files)",
            self.address, path,
            self.fs().space_used());
        Ok(())
    }

    // -------------------------------------------------------------------
    // Query methods (read-only, no handle required)
    // -------------------------------------------------------------------

    /// Check if the filesystem is read-only.
    ///
    /// Corresponds to `filesystem.isReadOnly()` in Lua.
    /// Read-only filesystems reject all write, delete, rename, and
    /// mkdir operations.
    pub fn is_read_only(&self)  -> bool { self.fs.is_read_only() }

    /// Total capacity of the filesystem in bytes.
    ///
    /// Corresponds to `filesystem.spaceTotal()` in Lua.
    /// Returns 0 for unlimited-capacity filesystems (which Lua
    /// interprets as `math.huge`).
    pub fn space_total(&self)   -> u64  { self.fs.space_total() }

    /// Currently used space in bytes (sum of all file sizes).
    ///
    /// Corresponds to `filesystem.spaceUsed()` in Lua.
    /// Does not account for per-file overhead (`file_cost` in config);
    /// the `VirtualFs` layer handles that.
    pub fn space_used(&self)    -> u64  { self.fs.space_used() }

    /// Check if a path exists (file or directory).
    ///
    /// Corresponds to `filesystem.exists(path)` in Lua.
    /// The path is normalised before checking.
    pub fn exists(&self, p: &str)       -> bool { self.fs.exists(p) }

    /// Check if a path is a directory.
    ///
    /// Corresponds to `filesystem.isDirectory(path)` in Lua.
    /// Returns `false` for files and non-existent paths.
    pub fn is_directory(&self, p: &str) -> bool { self.fs.is_directory(p) }

    /// Get the size of a file in bytes.
    ///
    /// Corresponds to `filesystem.size(path)` in Lua.
    /// Returns 0 for directories and non-existent paths.
    pub fn size(&self, p: &str)         -> u64  { self.fs.size(p) }

    /// List entries in a directory.
    ///
    /// Corresponds to `filesystem.list(path)` in Lua.
    ///
    /// # Returns
    ///
    /// * `Some(entries)` - A sorted list of entry names. Directories
    ///   have a trailing `/` (e.g. `"subdir/"`), files do not.
    /// * `None` - The path does not exist or is not a directory.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local entries = component.invoke(fs_addr, "list", "/")
    /// -- entries = {"bin/", "etc/", "init.lua", "lib/", ...}
    /// ```
    pub fn list(&self, path: &str) -> Option<Vec<String>> {
        self.fs.list(path)
    }

    // -------------------------------------------------------------------
    // Mutation methods (no handle required)
    // -------------------------------------------------------------------

    /// Create a directory.
    ///
    /// Corresponds to `filesystem.makeDirectory(path)` in Lua.
    ///
    /// # Returns
    ///
    /// `true` if the directory was created (or already exists as a
    /// directory). `false` if the filesystem is read-only or a file
    /// exists at that path.
    pub fn make_directory(&mut self, path: &str) -> bool {
        self.fs.make_directory(path)
    }

    /// Delete a file or directory (recursively).
    ///
    /// Corresponds to `filesystem.remove(path)` in Lua.
    ///
    /// # Returns
    ///
    /// `true` if anything was deleted, `false` otherwise (read-only,
    /// or path does not exist).
    ///
    /// # Warning
    ///
    /// If the deleted file has an open handle, subsequent reads/writes
    /// on that handle will fail with `"file not found"`.
    pub fn remove(&mut self, path: &str) -> bool {
        self.fs.delete(path)
    }

    /// Rename (move) a file.
    ///
    /// Corresponds to `filesystem.rename(from, to)` in Lua.
    ///
    /// # Returns
    ///
    /// `true` if the rename succeeded, `false` if the source does not
    /// exist or the filesystem is read-only.
    ///
    /// # Note
    ///
    /// Open handles pointing to the old path will become stale.
    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        self.fs.rename(from, to)
    }

    // -------------------------------------------------------------------
    // Handle-based I/O
    // -------------------------------------------------------------------

    /// Open a file and return an integer handle ID.
    ///
    /// Corresponds to `filesystem.open(path, mode)` in Lua.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to open (normalised internally).
    /// * `mode` - One of:
    ///   * `OpenMode::Read` - File must exist and not be a directory.
    ///   * `OpenMode::Write` - File is created/truncated. Filesystem
    ///     must not be read-only.
    ///   * `OpenMode::Append` - File is created if absent, not
    ///     truncated. Cursor starts at end.
    ///
    /// # Returns
    ///
    /// * `Ok(handle_id)` - A positive integer that can be passed to
    ///   `read`, `write`, `seek`, and `close`.
    /// * `Err(msg)` - An error string:
    ///   * `"file not found"` - Read mode, file missing or is directory.
    ///   * `"filesystem is read-only"` - Write/append on a read-only FS.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local handle, err = component.invoke(fs_addr, "open", "/init.lua", "r")
    /// if not handle then error(err) end
    /// ```
    pub fn open(&mut self, path: &str, mode: OpenMode) -> Result<i32, &'static str> {
        match mode {
            OpenMode::Read => {
                if !self.fs.exists(path) || self.fs.is_directory(path) {
                    return Err("file not found");
                }
            }
            OpenMode::Write => {
                if self.fs.is_read_only() { return Err("filesystem is read-only"); }
                // Truncate existing file or create a new empty one.
                self.fs.write_file(path, Vec::new());
            }
            OpenMode::Append => {
                if self.fs.is_read_only() { return Err("filesystem is read-only"); }
                if !self.fs.exists(path) {
                    self.fs.write_file(path, Vec::new());
                }
            }
        }

        let handle_id = self.next_handle;
        self.next_handle += 1;

        let pos = match mode {
            OpenMode::Append => self.fs.size(path),
            _ => 0,
        };

        self.handles.insert(handle_id, OpenHandle {
            path: VirtualFs::normalize(path),
            position: pos,
            mode,
        });
        Ok(handle_id)
    }

    /// Read up to `count` bytes from an open handle.
    ///
    /// Corresponds to `filesystem.read(handle, count)` in Lua.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle ID returned by [`open`](FilesystemComponent::open).
    /// * `count` - Maximum number of bytes to read. Clamped to file
    ///   length minus current position.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(bytes))` - The read data (may be shorter than `count`).
    /// * `Ok(None)` - End of file reached (position >= file length).
    /// * `Err("bad file descriptor")` - Handle not found or not opened
    ///   in read mode.
    /// * `Err("file not found")` - The underlying file was deleted while
    ///   the handle was open.
    ///
    /// # Side effects
    ///
    /// Advances the handle's position by the number of bytes read.
    pub fn read(&mut self, handle: i32, count: usize) -> Result<Option<Vec<u8>>, &'static str> {
        let h = self.handles.get_mut(&handle).ok_or("bad file descriptor")?;
        if h.mode != OpenMode::Read { return Err("bad file descriptor"); }

        let data = self.fs.read_file(&h.path).ok_or("file not found")?;
        let pos = h.position as usize;
        if pos >= data.len() { return Ok(None); }

        let end = (pos + count).min(data.len());
        let chunk = data[pos..end].to_vec();
        h.position = end as u64;
        Ok(Some(chunk))
    }

    /// Write bytes to an open handle.
    ///
    /// Corresponds to `filesystem.write(handle, data)` in Lua.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle ID opened in write or append mode.
    /// * `data` - Bytes to write. Appended to the file at the current
    ///   position.
    ///
    /// # Errors
    ///
    /// * `"bad file descriptor"` - Handle not found or opened in read mode.
    ///
    /// # Side effects
    ///
    /// * Appends `data` to the underlying file via `VirtualFs::append_file`.
    /// * Advances the handle's position by `data.len()`.
    ///
    /// # Note
    ///
    /// In this implementation, write-mode handles always append to the
    /// file (the file was truncated on open). Random-access writes
    /// (seeking then writing) are not supported, matching OC's
    /// behaviour where seek is read-only.
    pub fn write(&mut self, handle: i32, data: &[u8]) -> Result<(), &'static str> {
        let h = self.handles.get_mut(&handle).ok_or("bad file descriptor")?;
        if h.mode == OpenMode::Read { return Err("bad file descriptor"); }

        self.fs.append_file(&h.path, data);
        h.position += data.len() as u64;
        Ok(())
    }

    /// Seek within an open handle.
    ///
    /// Corresponds to `filesystem.seek(handle, whence, offset)` in Lua.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle ID opened in read mode.
    /// * `whence` - One of:
    ///   * `"set"` - Absolute position from the start of the file.
    ///   * `"cur"` - Relative to the current position.
    ///   * `"end"` - Relative to the end of the file.
    /// * `offset` - Byte offset (may be negative for `"cur"` and `"end"`).
    ///
    /// # Returns
    ///
    /// * `Ok(new_position)` - The new absolute byte position.
    /// * `Err("bad file descriptor")` - Handle not found or not in read mode.
    /// * `Err("invalid mode")` - Unknown `whence` string.
    /// * `Err("invalid offset")` - Resulting position would be negative.
    pub fn seek(&mut self, handle: i32, whence: &str, offset: i64) -> Result<u64, &'static str> {
        let h = self.handles.get_mut(&handle).ok_or("bad file descriptor")?;
        if h.mode != OpenMode::Read { return Err("bad file descriptor"); }

        let file_len = self.fs.size(&h.path) as i64;
        let new_pos = match whence {
            "set" => offset,
            "cur" => h.position as i64 + offset,
            "end" => file_len + offset,
            _     => return Err("invalid mode"),
        };
        if new_pos < 0 { return Err("invalid offset"); }
        h.position = new_pos as u64;
        Ok(h.position)
    }

    /// Close an open handle.
    ///
    /// Corresponds to `filesystem.close(handle)` in Lua.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle ID to close.
    ///
    /// # Errors
    ///
    /// Returns `Err("bad file descriptor")` if the handle does not exist
    /// (already closed or never opened).
    ///
    /// # Side effects
    ///
    /// Removes the handle from the internal map. The integer ID will
    /// never be reused.
    pub fn close(&mut self, handle: i32) -> Result<(), &'static str> {
        self.handles.remove(&handle).map(|_| ()).ok_or("bad file descriptor")
    }

    /// Close all open handles.
    ///
    /// Called during computer stop or reboot to ensure no file handles
    /// leak across sessions. This is equivalent to calling `close()` on
    /// every open handle, but more efficient (just clears the map).
    ///
    /// # When called
    ///
    /// * `Machine::stop()` or `Machine::crash()`
    /// * Before re-initialising the Lua state on reboot
    pub fn close_all(&mut self) {
        self.handles.clear();
    }
}