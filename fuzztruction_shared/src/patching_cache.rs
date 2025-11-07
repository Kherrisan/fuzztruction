use std::{
    assert_matches::assert_matches,
    collections::{BTreeMap, HashSet},
    env,
    ffi::CString,
    fmt::Debug,
    mem, slice,
};

use anyhow::{Context, Result};
use llvm_stackmap::LocationType;
use log::*;

use anyhow::anyhow;
use num_enum::IntoPrimitive;
use shared_memory::ShmemError;
use std::alloc;
use strum_macros::{AsRefStr, Display, EnumString};
use thiserror::Error;

use crate::{
    constants::ENV_SHM_NAME,
    patching_cache_content::{BitmapIter, PatchingCacheContent},
    patching_cache_entry::{
        PatchingCacheEntry, PatchingCacheEntryDirty, PatchingOperation, PatchingOperator,
        flags_to_str,
    },
    patchpoint::PatchPoint,
    tracing::Trace,
    types::PatchPointID,
    util,
};

pub const PATCHING_CACHE_DEFAULT_ENTRY_SIZE: usize = 400000;
pub const PATCHING_CACHE_DEFAULT_OP_SIZE: usize = 100000;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, IntoPrimitive, EnumString, AsRefStr, Display,
)]
#[repr(u8)]
pub enum PatchingCacheEntryFlags {
    /// Count the number of executions of this patch point and report it
    /// to the coordinator on termination.
    Tracing = 0,
    TracingVal = 1,
    Patching = 2,
    Jumping = 3,
}

pub const PATCHING_CACHE_ENTRY_FLAGS: [PatchingCacheEntryFlags; 4] = [
    PatchingCacheEntryFlags::Tracing,
    PatchingCacheEntryFlags::TracingVal,
    PatchingCacheEntryFlags::Patching,
    PatchingCacheEntryFlags::Jumping,
];

#[derive(Error, Debug)]
pub enum PatchingCacheError {
    #[error("Shared memory error: {0}")]
    ShareMemoryError(#[from] ShmemError),
    #[error("The cache can not hold anymore elements.")]
    CacheOverflow,
    #[error("Unable to find shm with the given name: {0}")]
    ShmNotFound(String),
    #[error("Shared memory error: {0}")]
    ShmError(String),
    #[error("Unkown error occurred: {0}")]
    Other(String),
}

pub mod backing_memory {
    use std::{env, ffi::CString};

    use crate::patching_cache_content::PatchingCacheContent;

    #[derive(Debug)]
    pub struct ShmMemory {
        pub name: String,
        pub shm_fd: i32,
        pub shm_path: CString,
        pub size: usize,
    }

    #[derive(Debug)]
    pub struct HeapMemory {
        pub memory: Box<PatchingCacheContent>,
        pub size: usize,
    }

    #[derive(Debug)]
    pub enum Memory {
        ShmMemory(ShmMemory),
        HeapMemory(HeapMemory),
    }

    impl Memory {
        pub fn shm_memory(&self) -> Option<&ShmMemory> {
            if let Memory::ShmMemory(shm) = self {
                Some(shm)
            } else {
                None
            }
        }

        #[allow(unused)]
        pub fn heap_memory(&self) -> Option<&HeapMemory> {
            if let Memory::HeapMemory(heap) = self {
                Some(heap)
            } else {
                None
            }
        }
    }

    impl Drop for Memory {
        fn drop(&mut self) {
            if let Some(s) = self.shm_memory() {
                if env::var("PINGU_GDB").is_ok() {
                    log::info!("PINGU_GDB is set, skipping unlinking of patching cache");
                } else {
                    unsafe {
                        libc::shm_unlink(s.shm_path.as_ptr());
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct PatchingCacheContentRawPtr(*mut PatchingCacheContent);

unsafe impl Send for PatchingCacheContentRawPtr {}

pub struct PatchingCache {
    backing_memory: backing_memory::Memory,
    content_size: usize,
    content: PatchingCacheContentRawPtr,
    b_tree: Option<BTreeMap<PatchPointID, usize>>,
    dirty: bool,
}

impl Debug for PatchingCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dirty_cnt = 0;
        let mut clear_cnt = 0;
        let mut enable_cnt = 0;
        for entry in self.entries() {
            if entry.is_any(PatchingCacheEntryDirty::Dirty) {
                dirty_cnt += 1;
            }
            if entry.is_any(PatchingCacheEntryDirty::Clear) {
                clear_cnt += 1;
            }
            if entry.is_any(PatchingCacheEntryDirty::Enable) {
                enable_cnt += 1;
            }
        }
        write!(f, "PatchingCache {{\n")?;
        write!(f, "    total_size: {},\n", self.total_size())?;
        write!(f, "    used_len: {},\n", self.len())?;
        write!(f, "    dirty_cnt: {},\n", dirty_cnt)?;
        write!(f, "    enable_cnt: {},\n", enable_cnt)?;
        write!(f, "    clear_cnt: {},\n", clear_cnt)?;
        write!(f, "}}")
    }
}

impl Clone for PatchingCache {
    fn clone(&self) -> Self {
        let mut new_cache = PatchingCache::new(
            self.content().entry_table_size(),
            self.content().op_table_size(),
        )
        .unwrap();

        assert!(new_cache.content_size == self.content_size);

        unsafe {
            let dst = new_cache.content.0 as *mut u8;
            let src = self.content.0 as *const u8;
            std::ptr::copy_nonoverlapping(src, dst, self.content_size);
        }

        new_cache.b_tree = self.b_tree.clone();
        new_cache.dirty = self.dirty;
        new_cache
    }
}

impl Default for PatchingCache {
    fn default() -> Self {
        Self::new(
            PATCHING_CACHE_DEFAULT_ENTRY_SIZE,
            PATCHING_CACHE_DEFAULT_OP_SIZE,
        )
        .expect(
            format!(
                "Failed to create default patching cache, with entry size {} and op size {}",
                PATCHING_CACHE_DEFAULT_ENTRY_SIZE, PATCHING_CACHE_DEFAULT_OP_SIZE
            )
            .as_str(),
        )
    }
}

/// Getter and setter implementation.
impl PatchingCache {
    /// Get the name of the backing shared memory if it is allocated
    /// in a shm.
    pub fn shm_name(&self) -> Option<String> {
        self.backing_memory.shm_memory().map(|e| e.name.clone())
    }

    /// Get the size in bytes.
    pub fn total_size(&self) -> usize {
        self.content_size
    }

    pub fn content(&self) -> &PatchingCacheContent {
        unsafe { &*self.content.0 }
    }

    pub fn content_ptr(&self) -> *const PatchingCacheContent {
        self.content.0
    }

    pub fn content_mut(&mut self) -> &mut PatchingCacheContent {
        unsafe { &mut *(self.content.0 as *mut PatchingCacheContent) }
    }

    pub fn content_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.content.0 as *const u8, self.content_size) }
    }

    pub fn content_slice_mut(&self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.content.0 as *mut u8, self.content_size) }
    }

    /// Get a vector of references to all entries contained in the cache.
    /// NOTE: The returned referencers are only valid as long as no entries are added
    /// or removed.
    pub fn entries(&self) -> Vec<&PatchingCacheEntry> {
        self.content().entries()
    }

    /// Get a vector of mutable references to all entries contained in the cache.
    /// NOTE: The returned referencers are only valid as long as no entries are added
    /// or removed.
    pub fn entries_mut(&mut self) -> Vec<&mut PatchingCacheEntry> {
        self.content_mut().entries_mut()
    }

    pub fn entries_mut_static(&mut self) -> Vec<&'static mut PatchingCacheEntry> {
        self.content_mut()
            .entries_mut()
            .into_iter()
            .map(|e| unsafe { mem::transmute(e) })
            .collect()
    }

    pub fn entries_mut_ptr(&mut self) -> Vec<*mut PatchingCacheEntry> {
        self.entries_mut()
            .into_iter()
            .map(|e| e.as_mut_ptr())
            .collect()
    }
}

pub struct PatchingCacheIterMut<'a> {
    cache: &'a mut PatchingCache,
    bitmap_iter: BitmapIter,
}

impl<'a> Iterator for PatchingCacheIterMut<'a> {
    type Item = &'a mut PatchingCacheEntry;

    fn next(&mut self) -> Option<Self::Item> {
        self.bitmap_iter.next().map(|idx| unsafe {
            // SAFETY: We're extending the lifetime to match the iterator's lifetime parameter
            // The caller must ensure the cache outlives the iterator
            std::mem::transmute(self.cache.content_mut().entry_mut(idx))
        })
    }
}

impl<'a> PatchingCacheIterMut<'a> {
    fn new(cache: &'a mut PatchingCache) -> Self {
        let iter = cache.content_mut().iter();
        Self {
            cache,
            bitmap_iter: iter,
        }
    }
}
pub struct PatchingCacheIter<'a> {
    cache: &'a PatchingCache,
    bitmap_iter: BitmapIter,
}

impl<'a> Iterator for PatchingCacheIter<'a> {
    type Item = &'a PatchingCacheEntry;

    fn next(&mut self) -> Option<Self::Item> {
        self.bitmap_iter
            .next()
            .map(|idx| self.cache.content().entry_ref(idx))
    }
}

impl<'a> PatchingCacheIter<'a> {
    fn new(cache: &'a PatchingCache) -> Self {
        Self {
            cache,
            bitmap_iter: cache.content().iter(),
        }
    }
}

/// Implementations realted to creation and lifecycle management of a MutationCache.
impl PatchingCache {
    fn shm_open(name: &str, create: bool) -> Result<PatchingCache, PatchingCacheError> {
        let mut size = PatchingCacheContent::estimate_memory_occupied(
            PATCHING_CACHE_DEFAULT_ENTRY_SIZE,
            PATCHING_CACHE_DEFAULT_OP_SIZE,
        );

        assert!(!create || size >= mem::size_of::<PatchingCacheContent>());
        let name_c = CString::new(name).unwrap();
        let fd;

        trace!(
            "shm_open(name={name}, name_c={name_c:#?}, create={create}, size={size:x} ({size_kb}))",
            name = name,
            name_c = name_c,
            create = create,
            size = size,
            size_kb = size / 1024
        );

        unsafe {
            let mut flags = libc::O_RDWR;
            if create {
                flags |= libc::O_CREAT | libc::O_EXCL;
            }

            fd = libc::shm_open(name_c.as_ptr(), flags, 0o777);
            trace!("shm fd={}", fd);
            if fd < 0 {
                let err = format!("Failed to open shm {}", name);
                error!("{}", err);
                return Err(PatchingCacheError::ShmError(err));
            }

            if create {
                trace!("Truncating fd {fd} to {bytes} bytes", fd = fd, bytes = size);
                let ret = libc::ftruncate(fd, size as i64);
                if ret != 0 {
                    return Err(PatchingCacheError::ShmError(
                        "Failed to ftruncate shm".to_owned(),
                    ));
                }
            } else {
                let mut stat_buffer: libc::stat = std::mem::zeroed();
                let ret = libc::fstat(fd, &mut stat_buffer);
                if ret != 0 {
                    return Err(PatchingCacheError::ShmError(
                        "Failed to get shm size".to_owned(),
                    ));
                }
                size = stat_buffer.st_size as usize;
            }

            // Calculate the maximum alignment requirement
            let max_align = std::cmp::max(
                mem::align_of::<crate::patching_cache_entry::PatchingCacheEntry>(),
                mem::align_of::<crate::patching_cache_entry::PatchingOperation>(),
            );
            let max_align = std::cmp::max(max_align, mem::align_of::<PatchingCacheContent>());

            // First attempt: try to map at any address
            let mapping = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if mapping == libc::MAP_FAILED {
                return Err(PatchingCacheError::ShmError(
                    "Failed to map shm into addresspace".to_owned(),
                ));
            }

            let final_mapping = if mapping as usize % max_align == 0 {
                // Already aligned, use it directly
                trace!("Shared memory already aligned at {:p}", mapping);
                mapping
            } else {
                // Not aligned, need to remap to an aligned address
                trace!(
                    "Shared memory not aligned (addr={:p}, align={}), remapping...",
                    mapping, max_align
                );

                // Unmap the current mapping
                libc::munmap(mapping, size);

                // Find an aligned address using a larger anonymous mapping
                let aligned_size = size + max_align;
                let large_mapping = libc::mmap(
                    std::ptr::null_mut(),
                    aligned_size,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );

                if large_mapping == libc::MAP_FAILED {
                    return Err(PatchingCacheError::ShmError(
                        "Failed to find aligned address space".to_owned(),
                    ));
                }

                // Calculate the aligned address within the large mapping
                let aligned_addr = (large_mapping as usize + max_align - 1) & !(max_align - 1);

                // Unmap the large mapping
                libc::munmap(large_mapping, aligned_size);

                // Map the shared memory to the aligned address
                let aligned_mapping = libc::mmap(
                    aligned_addr as *mut libc::c_void,
                    size,
                    libc::PROT_WRITE | libc::PROT_READ,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    0,
                );

                if aligned_mapping == libc::MAP_FAILED {
                    return Err(PatchingCacheError::ShmError(
                        "Failed to map shm to aligned address".to_owned(),
                    ));
                }

                if aligned_mapping as usize != aligned_addr {
                    return Err(PatchingCacheError::ShmError(
                        "Failed to get expected aligned address".to_owned(),
                    ));
                }

                trace!(
                    "Successfully remapped to aligned address {:p}",
                    aligned_mapping
                );
                aligned_mapping
            };

            // Verify alignment
            let l = alloc::Layout::new::<PatchingCacheContent>();
            assert!(final_mapping as usize % l.align() == 0);
            assert!(final_mapping as usize % max_align == 0);

            let ptr = final_mapping as *mut PatchingCacheContent;
            let mutation_cache_content: &mut PatchingCacheContent = ptr.as_mut().unwrap();
            mutation_cache_content.init(
                PATCHING_CACHE_DEFAULT_ENTRY_SIZE,
                PATCHING_CACHE_DEFAULT_OP_SIZE,
                create,
            );

            return Ok(PatchingCache {
                backing_memory: backing_memory::Memory::ShmMemory(backing_memory::ShmMemory {
                    shm_fd: fd,
                    shm_path: name_c,
                    name: name.to_owned(),
                    size,
                }),
                content_size: size,
                content: PatchingCacheContentRawPtr(ptr),
                b_tree: Some(BTreeMap::new()),
                dirty: false,
            });
        };
    }

    pub fn new_shm(name: impl AsRef<str>) -> Result<PatchingCache> {
        let mc = PatchingCache::shm_open(name.as_ref(), true)?;
        Ok(mc)
    }

    pub fn open_shm(name: impl AsRef<str>) -> Result<PatchingCache> {
        let mc = PatchingCache::shm_open(name.as_ref(), false)?;
        Ok(mc)
    }

    pub fn open_shm_from_env() -> Result<PatchingCache> {
        let send_name = env::var(ENV_SHM_NAME);
        match send_name {
            Err(_) => Err(PatchingCacheError::ShmNotFound(format!(
                "Failed to find environment variable {}",
                ENV_SHM_NAME
            )))?,
            Ok(name) => PatchingCache::open_shm(&name),
        }
    }

    pub fn unlink(&mut self) {
        if let Some(name) = self.shm_name() {
            let name = CString::new(name).unwrap();
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
        }
    }

    pub fn new(entry_size: usize, op_size: usize) -> Result<PatchingCache> {
        let size = PatchingCacheContent::estimate_memory_occupied(entry_size, op_size);

        assert!(size >= std::mem::size_of::<PatchingCacheContent>());
        let mut memory = util::alloc_box_aligned_zeroed::<PatchingCacheContent>(size);
        unsafe {
            std::ptr::write_bytes(memory.as_mut() as *mut PatchingCacheContent, 0, 1);
        }
        let mutation_cache_content =
            unsafe { &mut *(memory.as_mut() as *mut PatchingCacheContent) };
        mutation_cache_content.init(entry_size, op_size, true);

        return Ok(PatchingCache {
            backing_memory: backing_memory::Memory::HeapMemory(backing_memory::HeapMemory {
                memory,
                size,
            }),
            content_size: size,
            content: PatchingCacheContentRawPtr(mutation_cache_content),
            b_tree: Some(BTreeMap::new()),
            dirty: false,
        });
    }

    /// Make the cache content non shared if it is backed by an shm.
    /// It is not allowed to call this function on caches using any other
    /// backing storage than shm.
    pub fn make_private(&self) -> Result<()> {
        use backing_memory::*;
        assert_matches!(self.backing_memory, Memory::ShmMemory(..));

        let mem_shm_fd;
        if let Memory::ShmMemory(shm) = &self.backing_memory {
            mem_shm_fd = shm.shm_fd;
        } else {
            unreachable!();
        }

        unsafe {
            let ret = libc::mmap(
                self.content.0 as *mut libc::c_void,
                self.content_size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_FIXED | libc::MAP_PRIVATE,
                mem_shm_fd,
                0,
            );
            if ret != self.content.0 as *mut libc::c_void {
                return Err(PatchingCacheError::Other(format!(
                    "Remapping shm failed (ret={:#?})",
                    ret
                )))?;
            }
        }

        // Make sure that the client does not mess with the shm fd.
        let ret = unsafe { libc::close(mem_shm_fd) };
        if ret != 0 {
            return Err(PatchingCacheError::Other(
                "Failed to close shm fd".to_owned(),
            ))?;
        }

        Ok(())
    }

    pub fn try_clone(&self) -> Result<PatchingCache> {
        // let ret = PatchingCache::new()?;
        // let bytes = self.content_slice();
        // ret.content_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        // unsafe {
        //     // Update the size since we might loaded the content from a differently
        //     // sized cache.
        //     (&mut *(ret.content.0 as *mut PatchingCacheContent)).update(ret.content_size);
        // }

        unimplemented!()
    }
}

/// Methods that are working on the actual cache content.
impl PatchingCache {
    pub fn print(&self) {
        self.entries().iter().for_each(|e| {
            println!("{:#?}", e);
        });
    }

    pub fn print_dirty_flags(&self) {
        self.entries().iter().for_each(|e| {
            println!("{}: {}", e.id().0, flags_to_str(e.dirty_flags()));
        });
    }

    pub fn get(&self, pp: PatchPointID) -> Option<&PatchingCacheEntry> {
        self.b_tree
            .as_ref()
            .unwrap()
            .get(&pp)
            .map(|idx| self.content().entry_ref(*idx))
    }

    pub fn push(&mut self, entry: PatchingCacheEntry, ops: Vec<PatchingOperation>) -> Result<()> {
        let entry = entry;
        self.content_mut().push(entry.clone()).and_then(|idx| {
            if ops.len() > 0 {
                self.content_mut().push_op_batch(idx, &ops)?;
            }
            self.b_tree.as_mut().unwrap().insert(entry.id(), idx);

            Ok(())
        })
    }

    /// Remove the [MutationCacheEntry] with the given `id` .
    /// !!! This will invalidate all references to the cached entries !!!
    pub fn remove(&mut self, id: PatchPointID) -> Result<()> {
        self.content_mut().remove(id).and_then(|_| {
            self.b_tree.as_mut().unwrap().remove(&id);
            Ok(())
        })
    }

    /// Clears the content of the cache.
    pub fn clear(&mut self) {
        PATCHING_CACHE_ENTRY_FLAGS.iter().for_each(|flag| {
            self.entries_mut().iter_mut().for_each(|e| {
                e.set_clear(*flag);
            });
        });
    }

    pub fn clear_by<F>(&mut self, flag: PatchingCacheEntryFlags, f: F)
    where
        F: Fn(&PatchingCacheEntry) -> bool,
    {
        self.entries_mut().iter_mut().for_each(|e| {
            if f(e) {
                e.set_clear(flag);
            }
        });
    }

    pub fn clear_flag(&mut self, flag: PatchingCacheEntryFlags) {
        self.content_mut().entries_mut().iter_mut().for_each(|e| {
            e.set_clear(flag);
        });
    }

    /// Get the number of `MutationCacheEntry` elements in this set.
    pub fn len(&self) -> usize {
        self.content().entry_count()
    }

    pub fn from_patchpoints(patchpoints: &[PatchPoint]) -> Result<PatchingCache> {
        let entry_size = patchpoints.len();
        let mut ret = Self::new(entry_size, entry_size)?;

        for pp in patchpoints {
            let e = PatchingCacheEntry::from(pp);
            ret.push(e, vec![])?;
        }

        Ok(ret)
    }

    pub fn need_sync(&self) -> bool {
        self.dirty
    }

    pub fn reset_dirty(&mut self) {
        self.dirty = false;
    }

    // pub fn replace(&mut self, other: &PatchingCache) -> Result<()> {
    //     if self.content_size < other.content_size {
    //         Err(PatchingCacheError::CacheOverflow)
    //             .context("Can not replace cache content with content from a larger cache.")?
    //     }
    //     // assert_eq!(self.content_size, other.content_size);

    //     // self.b_tree = other.b_tree.clone();

    //     // Copy the element (both the entries and the operations) from the other cache
    //     // into our content space.

    //     for entry in self.entries_mut() {
    //         entry.set_dirty(PatchingCacheEntryDirty::Clear);
    //     }

    //     for entry_idx in other.content().iter() {
    //         let new_entry = other.content().entry_ref(entry_idx);

    //         if let Some(&current_idx) = self.b_tree.as_ref().unwrap().get(&new_entry.id()) {
    //             let current_entry = self.content().entry_mut(current_idx);

    //             if current_entry.flags() != new_entry.flags() {
    //                 // The entry is already in the cache, but the flags are different.
    //                 // We need to update the flags, and recompiling the code
    //                 current_entry.set_dirty(PatchingCacheEntryDirty::Dirty);
    //                 current_entry.set_flags(new_entry.flags());
    //             } else {
    //                 // The entry is already in the cache, and the flags are unchanged.
    //                 // We leave the entry as it is.
    //                 current_entry.set_dirty(PatchingCacheEntryDirty::Nop);
    //             }

    //             let ops = other.content().ops(entry_idx);
    //             let current_ops = self.content().ops(current_idx);
    //             if current_ops != ops {
    //                 // The patching operations are different.
    //                 // Clear the current operations, and push the new ones.
    //                 // Note that the code does not need to be re-compiled.
    //                 self.content_mut().clear_entry_ops(current_idx)?;
    //                 self.content_mut().push_op_batch(current_idx, &ops)?;
    //             }
    //         } else {
    //             // The entry is not in the cache, so we need to add it.
    //             let ops = other.content().ops(entry_idx);
    //             let mut new_entry = new_entry.clone();
    //             new_entry.set_dirty(PatchingCacheEntryDirty::Dirty);
    //             self.push(new_entry, ops)?;
    //         }
    //     }

    //     Ok(())
    // }

    /***
     * Restore the content of the cache from another cache.
     *
     * This is used to restore the cache from a **backup** cache.
     * Make this cache the exact same as the backup cache.
     */
    pub fn restore(&mut self, backup: &PatchingCache) -> Result<()> {
        assert_eq!(self.content_size, backup.content_size);

        self.b_tree = backup.b_tree.clone();

        self.content_mut().clear();
        // Copy the other content into our content buffer. We checked that
        // the other content is <= our content, thus this is safe.
        let dst = self.content_slice_mut();
        let src = backup.content_slice();
        dst[..src.len()].copy_from_slice(src);

        // // The copied content might stream from a cache that was smaller than us,
        // // hence we need to inform the MutationCacheContent that its backing memory
        // // size might have changed.
        // self.content_mut().update(content_size);

        Ok(())
    }

    /// Only retain elements for which `f` returns true.
    /// !!! This will invalidate all references to the contained entries !!!
    pub fn retain<F>(&mut self, f: F) -> usize
    where
        F: FnMut(&PatchingCacheEntry) -> bool,
    {
        self.content_mut().retain(f)
    }

    pub fn iter(&self) -> impl Iterator<Item = &PatchingCacheEntry> {
        PatchingCacheIter::new(self)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut PatchingCacheEntry> {
        PatchingCacheIterMut::new(self)
    }

    pub fn replace(&mut self, other: &PatchingCache) -> Result<()> {
        // 使用优化的单次遍历版本
        self.replace_single_pass(other)
    }

    /// 单次遍历优化版本：使用双指针算法一次性处理所有flags
    fn replace_single_pass(&mut self, other: &PatchingCache) -> Result<()> {
        log::trace!("Replacing all flags in single pass");

        // 收集需要添加的新条目，避免遍历时修改BTreeMap
        let mut entries_to_add: Vec<(PatchingCacheEntry, Vec<PatchingOperation>)> = Vec::new();

        let mut self_iter = self.b_tree.as_ref().unwrap().iter().peekable();
        let mut other_iter = other.b_tree.as_ref().unwrap().iter().peekable();

        loop {
            match (self_iter.peek(), other_iter.peek()) {
                (Some((self_pp_id, self_idx)), Some((other_pp_id, other_idx))) => {
                    match self_pp_id.cmp(other_pp_id) {
                        std::cmp::Ordering::Equal => {
                            // 交集：同时存在于两个缓存中的条目
                            let self_entry = self.content().entry_mut(**self_idx);
                            let other_entry = other.content().entry_ref(**other_idx);
                            Self::process_intersected_entry_all_flags(self_entry, other_entry);

                            self_iter.next();
                            other_iter.next();
                        }
                        std::cmp::Ordering::Less => {
                            // 只在self中存在的条目
                            let self_entry = self.content().entry_mut(**self_idx);
                            Self::process_self_only_entry_all_flags(self_entry);

                            self_iter.next();
                        }
                        std::cmp::Ordering::Greater => {
                            // 只在other中存在的条目 - 收集而不是立即添加
                            let mut other_entry = other.content().entry_ref(**other_idx).clone();
                            if Self::should_add_other_only_entry_all_flags(&mut other_entry) {
                                let ops = other.content().ops(**other_idx);
                                entries_to_add.push((other_entry, ops));
                            }

                            other_iter.next();
                        }
                    }
                }
                (Some((_, self_idx)), None) => {
                    // 处理剩余的self条目
                    let self_entry = self.content().entry_mut(**self_idx);
                    Self::process_self_only_entry_all_flags(self_entry);
                    self_iter.next();
                }
                (None, Some((_, other_idx))) => {
                    // 处理剩余的other条目
                    let mut other_entry = other.content().entry_ref(**other_idx).clone();
                    if Self::should_add_other_only_entry_all_flags(&mut other_entry) {
                        let ops = other.content().ops(**other_idx);
                        entries_to_add.push((other_entry, ops));
                    }
                    other_iter.next();
                }
                (None, None) => {
                    // 两个迭代器都结束
                    break;
                }
            }
        }

        // 批量添加收集的条目
        for (entry, ops) in entries_to_add {
            self.push(entry, ops)?;
        }

        Ok(())
    }

    /// 处理交集条目：同时存在于self和other中的条目
    fn process_intersected_entry_flag(
        self_entry: &mut PatchingCacheEntry,
        other_entry: &PatchingCacheEntry,
        flag: PatchingCacheEntryFlags,
    ) {
        match (other_entry.flag(flag), self_entry.flag(flag)) {
            (PatchingCacheEntryDirty::Clear, PatchingCacheEntryDirty::Nop) => {}
            (PatchingCacheEntryDirty::Clear, _) => {
                // Current entry is Enable, we need to clear it
                self_entry.set_clear(flag);
            }
            (PatchingCacheEntryDirty::Dirty, PatchingCacheEntryDirty::Enable) => {}
            (PatchingCacheEntryDirty::Dirty, _) => {
                // Current entry is Nop, maybe cleared before
                // We need to set it dirty, so it will be re-compiled next time.
                self_entry.set_dirty(flag);
            }
            (PatchingCacheEntryDirty::Nop, PatchingCacheEntryDirty::Enable) => {
                // Current entry is Enable, we need to clear it next time.
                self_entry.set_clear(flag);
            }
            (PatchingCacheEntryDirty::Nop, _) => {
                // (Nop, Nop), do nothing
            }
            (PatchingCacheEntryDirty::Enable, PatchingCacheEntryDirty::Enable) => {}
            (PatchingCacheEntryDirty::Enable, _) => {
                // Current entry is Nop, we need to set it dirty, so it will be re-compiled next time.
                self_entry.set_dirty(flag);
            }
        }
    }

    /// 处理只在self中存在的条目
    fn process_self_only_entry_flag(
        self_entry: &mut PatchingCacheEntry,
        flag: PatchingCacheEntryFlags,
    ) {
        match self_entry.flag(flag) {
            PatchingCacheEntryDirty::Enable => {
                self_entry.set_clear(flag);
            }
            _ => {
                // Nop, means the current entry has been cleared before, being set to nop.
            }
        }
    }

    /// 处理只在other中存在的条目，返回是否应该添加到self中
    fn should_add_other_only_entry_flag(
        other_entry: &mut PatchingCacheEntry,
        flag: PatchingCacheEntryFlags,
    ) -> bool {
        match other_entry.flag(flag) {
            PatchingCacheEntryDirty::Dirty | PatchingCacheEntryDirty::Enable => {
                // The entry is dirty or enable, we need to set it dirty, so it will be re-compiled next time.
                other_entry.set_dirty(flag);
                true
            }
            _ => {
                // Nop or Clear, do nothing
                false
            }
        }
    }

    /// 处理所有flags的交集条目
    fn process_intersected_entry_all_flags(
        self_entry: &mut PatchingCacheEntry,
        other_entry: &PatchingCacheEntry,
    ) {
        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        for &flag in &FLAGS {
            Self::process_intersected_entry_flag(self_entry, other_entry, flag);
        }
    }

    /// 处理所有flags的仅self条目
    fn process_self_only_entry_all_flags(self_entry: &mut PatchingCacheEntry) {
        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        for &flag in &FLAGS {
            Self::process_self_only_entry_flag(self_entry, flag);
        }
    }

    /// 处理所有flags的仅other条目，返回是否应该添加
    fn should_add_other_only_entry_all_flags(other_entry: &mut PatchingCacheEntry) -> bool {
        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        let mut should_add = false;
        for &flag in &FLAGS {
            if Self::should_add_other_only_entry_flag(other_entry, flag) {
                should_add = true;
            }
        }
        should_add
    }

    #[deprecated(since = "2025-09-07", note = "Use replace_single_pass instead")]
    fn replace_with_dirty_flag(
        &mut self,
        other: &PatchingCache,
        flag: PatchingCacheEntryFlags,
    ) -> Result<()> {
        log::trace!("Replacing with flag: {:?}", flag);
        let mut other_pp_ids = other.content().iter().collect::<HashSet<_>>();

        for idx in self.content().iter() {
            let entry = self.content().entry_mut(idx);
            let pp_id = entry.id();

            if let Some(&other_idx) = other.b_tree.as_ref().unwrap().get(&pp_id) {
                other_pp_ids.remove(&other_idx);

                // Current cache could also be found from the other cache.
                let other_entry = other.content().entry_ref(other_idx);
                Self::process_intersected_entry_flag(entry, other_entry, flag);
            } else {
                // Current entry could not be found from the other cache.
                Self::process_self_only_entry_flag(entry, flag);
            }
        }

        // For other entry that is not existed in the current cache
        for idx in other_pp_ids {
            let mut other_entry = other.content().entry_ref(idx).clone();
            if Self::should_add_other_only_entry_flag(&mut other_entry, flag) {
                self.push(other_entry.clone(), other.content().ops(idx))?;
            }
        }

        Ok(())
    }

    /**
     * Union two patching caches with different flags into one.
     *
     * For example, if we need to do tracing and patching at the same time, we can union the two
     * patching cache into the same one.
     * Note that when the two caches have the entries with the same flags and the same id,
     * the entries in the other (new) cache will be kept.
     */
    pub fn union(&mut self, other: &PatchingCache) -> Result<()> {
        // 使用优化的单次遍历版本
        self.union_single_pass(other)
    }

    /// Union操作的双指针优化版本：使用双指针算法一次性处理所有flags
    fn union_single_pass(&mut self, other: &PatchingCache) -> Result<()> {
        log::trace!("Unioning all flags with dual-pointer algorithm in single pass");

        // 收集需要添加的新条目，避免遍历时修改BTreeMap
        let mut entries_to_add: Vec<(PatchingCacheEntry, Vec<PatchingOperation>)> = Vec::new();
        // 收集需要处理patching操作的条目信息
        let mut patching_operations_to_handle: Vec<(usize, usize, PatchingCacheEntryFlags)> =
            Vec::new();
        let mut any_dirty = false;

        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        let mut self_iter = self.b_tree.as_ref().unwrap().iter().peekable();
        let mut other_iter = other.b_tree.as_ref().unwrap().iter().peekable();

        loop {
            match (self_iter.peek(), other_iter.peek()) {
                (Some((self_pp_id, self_idx)), Some((other_pp_id, other_idx))) => {
                    match self_pp_id.cmp(other_pp_id) {
                        std::cmp::Ordering::Equal => {
                            // 交集：同时存在于两个缓存中的条目
                            let other_entry = other.content().entry_ref(**other_idx);

                            // 检查other中是否有dirty flags
                            let has_dirty_flags =
                                FLAGS.iter().any(|&flag| other_entry.is_dirty(flag));
                            if has_dirty_flags {
                                let self_entry = self.content().entry_mut(**self_idx);
                                let set_dirty = Self::process_union_intersected_entry_all_flags(
                                    self_entry,
                                    other_entry,
                                );

                                // 检查冲突：patching和jumping不能同时设置
                                if self_entry.is_dirty(PatchingCacheEntryFlags::Patching)
                                    && self_entry.is_dirty(PatchingCacheEntryFlags::Jumping)
                                {
                                    return Err(anyhow!(
                                        "Patching and jumping are both set for entry (PatchPointID)#{}",
                                        self_pp_id.0
                                    ));
                                }

                                // 收集需要处理patching操作的信息
                                for &flag in &FLAGS {
                                    if other_entry.is_dirty(flag) {
                                        patching_operations_to_handle.push((
                                            **self_idx,
                                            **other_idx,
                                            flag,
                                        ));
                                    }
                                }

                                if set_dirty {
                                    any_dirty = true;
                                }
                            }

                            self_iter.next();
                            other_iter.next();
                        }
                        std::cmp::Ordering::Less => {
                            // 只在self中存在的条目 - Union操作不处理这种情况，直接跳过
                            self_iter.next();
                        }
                        std::cmp::Ordering::Greater => {
                            // 只在other中存在的条目
                            let other_entry = other.content().entry_ref(**other_idx);

                            // 检查是否有dirty flags需要添加
                            let has_dirty_flags =
                                FLAGS.iter().any(|&flag| other_entry.is_dirty(flag));
                            if has_dirty_flags {
                                let mut new_entry = other_entry.clone();
                                if Self::should_add_union_entry_all_flags(&mut new_entry) {
                                    let ops = other.content().ops(**other_idx);
                                    entries_to_add.push((new_entry, ops));
                                    any_dirty = true;
                                }
                            }

                            other_iter.next();
                        }
                    }
                }
                (Some(_), None) => {
                    // 剩余的self条目 - Union操作不处理，直接跳过
                    self_iter.next();
                }
                (None, Some((_, other_idx))) => {
                    // 剩余的other条目
                    let other_entry = other.content().entry_ref(**other_idx);

                    // 检查是否有dirty flags需要添加
                    let has_dirty_flags = FLAGS.iter().any(|&flag| other_entry.is_dirty(flag));
                    if has_dirty_flags {
                        let mut new_entry = other_entry.clone();
                        if Self::should_add_union_entry_all_flags(&mut new_entry) {
                            let ops = other.content().ops(**other_idx);
                            entries_to_add.push((new_entry, ops));
                            any_dirty = true;
                        }
                    }

                    other_iter.next();
                }
                (None, None) => {
                    // 两个迭代器都结束
                    break;
                }
            }
        }

        // 批量处理patching操作
        for (self_idx, other_idx, flag) in patching_operations_to_handle {
            self.handle_union_patching_operations(self_idx, other_idx, other, flag)?;
        }

        // 批量添加收集的条目
        for (entry, ops) in entries_to_add {
            self.push(entry, ops)?;
        }

        if any_dirty {
            self.dirty = true;
        }

        Ok(())
    }

    /// 传统四次遍历版本（保留用于对比测试）
    #[allow(dead_code)]
    fn union_four_pass(&mut self, other: &PatchingCache) -> Result<()> {
        self.union_with_dirty_flag(other, PatchingCacheEntryFlags::Tracing)?;
        self.union_with_dirty_flag(other, PatchingCacheEntryFlags::TracingVal)?;
        self.union_with_dirty_flag(other, PatchingCacheEntryFlags::Patching)?;
        self.union_with_dirty_flag(other, PatchingCacheEntryFlags::Jumping)?;

        Ok(())
    }

    /// 处理union操作中的交集条目：在self中存在的other脏条目
    fn process_union_intersected_entry_flag(
        self_entry: &mut PatchingCacheEntry,
        other_entry: &PatchingCacheEntry,
        flag: PatchingCacheEntryFlags,
    ) -> bool {
        // 只处理other中dirty的条目
        if !other_entry.is_dirty(flag) {
            return false;
        }

        let mut set_dirty = false;
        match self_entry.flag(flag) {
            PatchingCacheEntryDirty::Clear => {
                *self_entry.flag_mut(flag) = PatchingCacheEntryDirty::Enable;
            }
            PatchingCacheEntryDirty::Dirty => {}
            PatchingCacheEntryDirty::Nop => {
                self_entry.set_dirty(flag);
                set_dirty = true;
            }
            PatchingCacheEntryDirty::Enable => {}
        }

        set_dirty
    }

    /// 检查union操作中的新条目：只在other中存在的脏条目
    fn should_add_union_entry_flag(
        other_entry: &mut PatchingCacheEntry,
        flag: PatchingCacheEntryFlags,
    ) -> bool {
        if other_entry.is_dirty(flag) {
            other_entry.set_dirty(flag);
            true
        } else {
            false
        }
    }

    /// 处理所有flags的union交集条目
    fn process_union_intersected_entry_all_flags(
        self_entry: &mut PatchingCacheEntry,
        other_entry: &PatchingCacheEntry,
    ) -> bool {
        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        let mut any_set_dirty = false;
        for &flag in &FLAGS {
            if Self::process_union_intersected_entry_flag(self_entry, other_entry, flag) {
                any_set_dirty = true;
            }
        }

        any_set_dirty
    }

    /// 检查所有flags的union新条目
    fn should_add_union_entry_all_flags(other_entry: &mut PatchingCacheEntry) -> bool {
        const FLAGS: [PatchingCacheEntryFlags; 4] = [
            PatchingCacheEntryFlags::Tracing,
            PatchingCacheEntryFlags::TracingVal,
            PatchingCacheEntryFlags::Patching,
            PatchingCacheEntryFlags::Jumping,
        ];

        let mut should_add = false;
        for &flag in &FLAGS {
            if Self::should_add_union_entry_flag(other_entry, flag) {
                should_add = true;
            }
        }
        should_add
    }

    /// 处理patching操作的特殊逻辑
    fn handle_union_patching_operations(
        &mut self,
        self_idx: usize,
        other_idx: usize,
        other: &PatchingCache,
        flag: PatchingCacheEntryFlags,
    ) -> Result<()> {
        match flag {
            PatchingCacheEntryFlags::Patching | PatchingCacheEntryFlags::Jumping => {
                let other_ops = other.content().ops(other_idx);
                let ops = self.content().ops(self_idx);

                if ops != other_ops {
                    // patching operations are executed dynamically,
                    // inside the `patching_xxx` stub,
                    // so the code does not need to be re-compiled.
                    self.content_mut().clear_entry_ops(self_idx)?;
                    self.content_mut().push_op_batch(self_idx, &other_ops)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn union_with_dirty_flag(
        &mut self,
        other: &PatchingCache,
        flag: PatchingCacheEntryFlags,
    ) -> Result<()> {
        log::trace!("Unioning with flag: {:?}", flag);
        for other_idx in other.content().iter() {
            let other_entry = other.content().entry_ref(other_idx);

            if !other_entry.is_dirty(flag) {
                continue;
            }

            let pp_id = other_entry.id();

            let set_dirty = if let Some(&idx) = self.b_tree.as_ref().unwrap().get(&pp_id) {
                let entry = self.content().entry_mut(idx);
                let set_dirty =
                    Self::process_union_intersected_entry_flag(entry, other_entry, flag);

                // 检查冲突：patching和jumping不能同时设置
                if entry.is_dirty(PatchingCacheEntryFlags::Patching)
                    && entry.is_dirty(PatchingCacheEntryFlags::Jumping)
                {
                    return Err(anyhow!(
                        "Patching and jumping are both set for entry (PatchPointID)#{}",
                        pp_id.0
                    ));
                }

                // 处理patching操作
                self.handle_union_patching_operations(idx, other_idx, other, flag)?;

                set_dirty
            } else {
                let ops = other.content().ops(other_idx);
                let mut other_entry = other_entry.clone();
                other_entry.set_dirty(flag);
                self.push(other_entry, ops)?;

                true
            };

            if set_dirty {
                self.dirty = true;
            }
        }

        Ok(())
    }

    /// Set the flags of all mutation cache entries to zero.
    pub fn reset_flags(&mut self) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.reset_dirty_flags();
        });
        self
    }

    /// Enable tracing for all mutation entries in this set.
    pub fn enable_tracing(&mut self) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.set_dirty(PatchingCacheEntryFlags::Tracing);
        });
        self
    }

    /// Disable tracing for all mutation entries in this set.
    pub fn disable_tracing(&mut self) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.set_clear(PatchingCacheEntryFlags::Tracing);
        });
        self
    }

    /// Remove all mutation entries that have the passed LocationType.
    pub fn remove_by_location_type(&mut self, loc_type: LocationType) -> &mut Self {
        self.retain(|e| e.loc_type() != loc_type);
        self
    }

    /// Purge all expect `max` elements from the cache.
    pub fn limit(&mut self, max: usize) -> &mut Self {
        let mut limit = max;
        self.retain(|_e| {
            if limit > 0 {
                limit -= 1;
                true
            } else {
                false
            }
        });
        self
    }

    /// Remove all mutation entries that are of type LocationType::Constant and therefore
    /// are not associated to any live value we might mutate.
    pub fn remove_const_type(&mut self) -> &mut Self {
        log::trace!("Removing const type entries from patching cache");
        self.remove_by_location_type(LocationType::Constant);
        self
    }

    pub fn release_nop(&mut self) -> usize {
        self.entries_mut().iter_mut().for_each(|e| {
            e.set_by(PatchingCacheEntryDirty::Clear, PatchingCacheEntryDirty::Nop);
        });
        let removing_entries_ids = self
            .entries()
            .iter()
            .filter(|e| {
                if e.is_all(PatchingCacheEntryDirty::Nop) {
                    true
                } else if e.is_any(PatchingCacheEntryDirty::Enable) {
                    false
                } else {
                    // only nop and clear
                    // in fact, clear would be set to nop just before, so it will not happen
                    true
                }
            })
            .map(|e| e.id().clone())
            .collect::<Vec<_>>();

        removing_entries_ids.iter().for_each(|id| {
            self.remove(*id).unwrap();
        });

        removing_entries_ids.len()
        // self.retain(|e| {
        //     if e.is_all(PatchingCacheEntryDirty::Nop) {
        //         false
        //     } else if e.is_any(PatchingCacheEntryDirty::Enable) {
        //         true
        //     } else {
        //         // only nop and clear
        //         // in fact, clear would be set to nop just before, so it will not happen
        //         false
        //     }
        // })
    }
}

impl PatchingCache {
    pub fn remove_uncovered(&mut self, trace: &Trace) -> &mut Self {
        let covered_ids = trace.items().iter().map(|i| i.id).collect::<HashSet<_>>();
        self.retain(|e| covered_ids.contains(&e.id().0));
        self
    }

    fn resize_covered_entries(&mut self, trace: &Trace) -> &mut Self {
        self
    }

    fn resize_covered_entries_wo_msk(&mut self, trace: &Trace) -> &mut Self {
        self
    }
}

mod test {
    use std::{ffi::CString, mem::transmute};

    use rand::Rng;

    use crate::{
        patching_cache::{
            PATCHING_CACHE_DEFAULT_ENTRY_SIZE, PATCHING_CACHE_DEFAULT_OP_SIZE, PatchingCache,
        },
        patching_cache_content::PatchingCacheContent,
        patching_cache_entry::PatchingCacheEntry,
    };

    fn generate_random_string(length: usize) -> String {
        return "123".to_string();
    }

    fn dummy_entry(id: u64) -> PatchingCacheEntry {
        PatchingCacheEntry::new(
            id.into(),
            0,
            llvm_stackmap::LocationType::Constant,
            8,
            crate::dwarf::DwarfReg::Rax,
            0,
            0,
        )
    }

    #[test]
    fn test_backup_restore() {
        for size in [
            100, 101, 102, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009,
            10010,
        ] {
            let cache = PatchingCache::new(size, size).unwrap();
            let backup = cache.clone();

            assert_eq!(cache.total_size(), backup.total_size());
        }
    }

    // #[test]
    // fn patching_cache_replace() {
    //     let mut cache = PatchingCache::default();
    //     for i in 0..100 {
    //         cache.push(dummy_entry(i as u64), vec![]).unwrap();
    //     }

    //     let mut other = PatchingCache::new(100, 100).unwrap();
    //     other.replace(&cache).unwrap();

    //     assert_eq!(cache.len(), other.len());
    // }

    #[test]
    fn test_shared_memory_alignment() {
        // Test that shared memory alignment works correctly
        let name = format!("test_alignment_{}", generate_random_string(8));

        // Create a shared memory cache
        let mut cache1 = PatchingCache::new_shm(&name).unwrap();
        let content1 = cache1.content();

        // Verify that the data structures are properly aligned
        let entry_align = std::mem::align_of::<crate::patching_cache_entry::PatchingCacheEntry>();
        let op_align = std::mem::align_of::<crate::patching_cache_entry::PatchingOperation>();

        unsafe {
            let entry_ptr = content1
                .data_ptr()
                .offset(content1.entry_data_offset() as isize);
            let op_ptr = content1
                .data_ptr()
                .offset(content1.op_data_offset() as isize);

            assert_eq!(
                entry_ptr as usize % entry_align,
                0,
                "Entry data not aligned: addr={:p}, align={}",
                entry_ptr,
                entry_align
            );
            assert_eq!(
                op_ptr as usize % op_align,
                0,
                "Op data not aligned: addr={:p}, align={}",
                op_ptr,
                op_align
            );
        }

        // Now open the same shared memory from another "process" (simulated)
        let cache2 = PatchingCache::open_shm(&name).unwrap();
        let content2 = cache2.content();

        // Verify that the offsets are the same between different opens
        assert_eq!(
            content1.entry_data_offset(),
            content2.entry_data_offset(),
            "Entry data offsets differ between cache instances"
        );
        assert_eq!(
            content1.op_data_offset(),
            content2.op_data_offset(),
            "Op data offsets differ between cache instances"
        );

        println!("✓ Shared memory alignment test passed");
        println!("  Entry data offset: {}", content1.entry_data_offset());
        println!("  Op data offset: {}", content1.op_data_offset());

        cache1.push(dummy_entry(101), vec![]).unwrap();

        assert_eq!(cache1.len(), 1);
        assert_eq!(cache1.len(), cache2.len());

        let (idx, entry) = cache2.content().find_entry(101u32.into()).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(entry.id(), 101u32.into());

        println!("✓ Shared memory data reading and writing test passed");
        println!("  Entry in cache1: {:?}", cache1.entries());
        println!("  Entry in cache2: {:?}", cache2.entries());

        // Clean up
        drop(cache2);
        let mut cache1 = cache1;
        cache1.unlink();
    }

    #[test]
    fn mem_mapping_transmute() {
        let name = format!("testing_{}", generate_random_string(4));
        let size = 1000000;
        unsafe {
            let flags = libc::O_RDWR | libc::O_CREAT | libc::O_EXCL;
            let c_name = CString::new(name.clone()).unwrap();
            let fd = libc::shm_open(c_name.as_ptr(), flags, 0o777);
            if fd < 0 {
                let err = format!("Failed to open shm {}", name);
                log::error!("{}", err);
                return;
            }
            log::trace!("Truncating fd {fd} to {size} bytes");
            let ret = libc::ftruncate(fd, size as i64);
            if ret != 0 {
                log::error!("Failed to ftruncate shm");
                return;
            }

            let mapping = libc::mmap(
                0 as *mut libc::c_void,
                // ptr::null() as *mut c_void,
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if mapping == libc::MAP_FAILED {
                log::error!("Failed to map shm into addresspace");
                return;
            }

            // Check alignment. Implment logging and sanitize messages
            let l = std::alloc::Layout::new::<PatchingCacheContent>();
            assert!(mapping as usize % l.align() == 0);

            // let ptr = mapping as *mut MutationCacheContent;
            let ptr = transmute::<*mut libc::c_void, *mut PatchingCacheContent>(mapping);
            let mutation_cache_content: &mut PatchingCacheContent = ptr.as_mut().unwrap();

            mutation_cache_content.init(
                PATCHING_CACHE_DEFAULT_ENTRY_SIZE,
                PATCHING_CACHE_DEFAULT_OP_SIZE,
                true,
            );
        }
    }

    #[test]
    fn test_estimate_memory_occupied() {
        let entry_size = 10000;
        let op_size = 10000;
        let expected_size = PatchingCacheContent::estimate_memory_occupied(entry_size, op_size);
        let actual_size = PatchingCacheContent::estimate_memory_occupied(entry_size, op_size);
        assert_eq!(expected_size, actual_size);
    }
}
