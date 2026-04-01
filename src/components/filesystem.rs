//! # Filesystem Component
//!
//! Wraps a [`VirtualFs`] and exposes it as an OC `filesystem` component.
//! Mirrors `FileSystem.scala`.

use crate::components::Address;
use crate::fs::VirtualFs;
use std::collections::HashMap;

/// Open file handle state.
struct OpenHandle {
    path: String,
    position: u64,
    mode: crate::fs::OpenMode,
}

/// Filesystem component.
pub struct FilesystemComponent {
    pub address: Address,
    pub label: Option<String>,
    fs: VirtualFs,
    handles: HashMap<i32, OpenHandle>,
    next_handle: i32,
}

impl FilesystemComponent {
    pub fn new(fs: VirtualFs) -> Self {
        Self {
            address: crate::components::new_address(),
            label: None,
            fs,
            handles: HashMap::new(),
            next_handle: 1,
        }
    }

    pub const fn component_name() -> &'static str { "filesystem" }

    pub fn fs(&self) -> &VirtualFs { &self.fs }
    pub fn fs_mut(&mut self) -> &mut VirtualFs { &mut self.fs }

    // ── Component API methods ───────────────────────────────────────────

    pub fn is_read_only(&self) -> bool { self.fs.is_read_only() }
    pub fn space_total(&self) -> u64 { self.fs.space_total() }
    pub fn space_used(&self) -> u64 { self.fs.space_used() }
    pub fn exists(&self, path: &str) -> bool { self.fs.exists(path) }
    pub fn is_directory(&self, path: &str) -> bool { self.fs.is_directory(path) }
    pub fn size(&self, path: &str) -> u64 { self.fs.size(path) }

    pub fn list(&self, path: &str) -> Option<Vec<String>> {
        self.fs.list(path)
    }

    pub fn make_directory(&mut self, path: &str) -> bool {
        self.fs.make_directory(path)
    }

    pub fn remove(&mut self, path: &str) -> bool {
        self.fs.delete(path)
    }

    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        self.fs.rename(from, to)
    }

    /// Open a file. Returns handle ID.
    pub fn open(&mut self, path: &str, mode: crate::fs::OpenMode) -> Result<i32, &'static str> {
        use crate::fs::OpenMode::*;
        match mode {
            Read => {
                if !self.fs.exists(path) || self.fs.is_directory(path) {
                    return Err("file not found");
                }
            }
            Write | Append => {
                if self.fs.is_read_only() { return Err("filesystem is read-only"); }
                // Create file if it doesn't exist.
                if !self.fs.exists(path) {
                    self.fs.write_file(path, Vec::new());
                }
            }
        }

        let handle_id = self.next_handle;
        self.next_handle += 1;

        let pos = match mode {
            Append => self.fs.size(path),
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
    pub fn read(&mut self, handle: i32, count: usize) -> Result<Option<Vec<u8>>, &'static str> {
        let h = self.handles.get_mut(&handle).ok_or("bad file descriptor")?;
        if h.mode != crate::fs::OpenMode::Read { return Err("bad file descriptor"); }

        let data = self.fs.read_file(&h.path).ok_or("file not found")?;
        let pos = h.position as usize;
        if pos >= data.len() { return Ok(None); } // EOF

        let end = (pos + count).min(data.len());
        let chunk = data[pos..end].to_vec();
        h.position = end as u64;
        Ok(Some(chunk))
    }

    /// Write bytes to an open handle.
    pub fn write(&mut self, handle: i32, data: &[u8]) -> Result<(), &'static str> {
        let h = self.handles.get_mut(&handle).ok_or("bad file descriptor")?;
        if h.mode == crate::fs::OpenMode::Read { return Err("bad file descriptor"); }

        // For simplicity, append always goes to end.
        self.fs.append_file(&h.path, data);
        h.position += data.len() as u64;
        Ok(())
    }

    /// Close a handle.
    pub fn close(&mut self, handle: i32) {
        self.handles.remove(&handle);
    }

    /// Close all handles (called on computer stop).
    pub fn close_all(&mut self) {
        self.handles.clear();
    }
}