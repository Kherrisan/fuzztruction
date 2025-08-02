use std::{
    assert_matches::assert_matches,
    collections::{BTreeMap, HashMap, HashSet},
    env,
    ffi::CString,
    mem, slice,
};

use anyhow::{Context, Result};
use llvm_stackmap::LocationType;
use log::*;

use anyhow::anyhow;
use num_enum::IntoPrimitive;
use shared_memory::ShmemError;
use std::alloc;
use thiserror::Error;

use crate::{
    constants::ENV_SHM_NAME,
    patching_cache_content::{BitmapIter, PatchingCacheContent},
    patching_cache_entry::{PatchingCacheEntry, PatchingOperation, PatchingOperator},
    patchpoint::PatchPoint,
    tracing::Trace,
    types::PatchPointID,
    util,
};

pub const PATCHING_CACHE_DEFAULT_ENTRY_SIZE: usize = 400000;
pub const PATCHING_CACHE_DEFAULT_OP_SIZE: usize = 100000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, IntoPrimitive)]
#[repr(u8)]
pub enum PatchingCacheEntryFlags {
    /// Count the number of executions of this patch point and report it
    /// to the coordinator on termination.
    Tracing = 1,
    TracingWithVal = 2,
    Patching = 4,
    Jumping = 8,
}

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
    use std::ffi::CString;

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
                unsafe {
                    libc::shm_unlink(s.shm_path.as_ptr());
                }
            }
        }
    }
}

#[derive(Debug)]
struct PatchingCacheContentRawPtr(*mut PatchingCacheContent);

unsafe impl Send for PatchingCacheContentRawPtr {}

#[derive(Debug)]
pub struct PatchingCache {
    backing_memory: backing_memory::Memory,
    content_size: usize,
    content: PatchingCacheContentRawPtr,
    b_tree: BTreeMap<PatchPointID, usize>,
}

impl Clone for PatchingCache {
    fn clone(&self) -> Self {
        let new_cache = PatchingCache::new(
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

    pub fn entries_by_flag(&self, flag: PatchingCacheEntryFlags) -> Vec<&PatchingCacheEntry> {
        self.content()
            .entries()
            .into_iter()
            .filter(|e| e.is_flag_set(flag))
            .collect()
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
                    mapping,
                    max_align
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
                b_tree: BTreeMap::new(),
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
            b_tree: BTreeMap::new(),
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
    pub fn push(&mut self, entry: PatchingCacheEntry) -> Result<()> {
        self.content_mut().push(entry.clone()).and_then(|idx| {
            // self.b_tree.insert(entry.id(), idx);
            Ok(())
        })
    }

    /// Remove the [MutationCacheEntry] with the given `id` .
    /// !!! This will invalidate all references to the cached entries !!!
    pub fn remove(&mut self, id: PatchPointID) -> Result<()> {
        self.content_mut().remove(id).and_then(|_| {
            // self.b_tree.remove(&id);
            Ok(())
        })
    }

    /// Clears the content of the cache.
    pub fn clear(&mut self) {
        self.content_mut().clear();
        self.b_tree.clear();
    }

    pub fn clear_by_flag(&mut self, flag: PatchingCacheEntryFlags) {
        self.content_mut().entries_mut().iter_mut().for_each(|e| {
            e.unset_flag(flag);
        });
    }

    /// Get the number of `MutationCacheEntry` elements in this set.
    pub fn len(&self) -> usize {
        self.content().entry_count()
    }

    // pub fn load_package(&mut self, package: &PatchingCacheContentPackage) -> Result<()> {
    //     // self.content_slice_mut()[..bytes.len()].copy_from_slice(bytes);
    //     // unsafe {
    //     //     // Update the size since we might loaded the content from a differently
    //     //     // sized cache.
    //     //     (&mut *(self.content.0 as *mut MutationCacheContent)).update(self.content_size);
    //     // }

    //     log::trace!("Loading package into patching cache");
    //     self.content_mut().clean_load_patching_table(package)
    // }

    pub fn from_iter<'a>(
        iter: impl Iterator<Item = &'a PatchingCacheEntry>,
    ) -> Result<PatchingCache> {
        let mut ret = Self::default();

        for elem in iter {
            ret.push(elem.clone())?;
        }

        Ok(ret)
    }

    pub fn from_patchpoints(patchpoints: &[PatchPoint]) -> Result<PatchingCache> {
        let entry_size = patchpoints.len();
        let mut ret = Self::new(entry_size, entry_size)?;

        for pp in patchpoints {
            let e = PatchingCacheEntry::from(pp);
            ret.push(e)?;
        }

        Ok(ret)
    }

    pub fn replace(&mut self, other: &PatchingCache) -> Result<()> {
        if self.content_size < other.content_size {
            Err(PatchingCacheError::CacheOverflow)
                .context("Can not replace cache content with content from a larger cache.")?
        }

        // Copy the element (both the entries and the operations) from the other cache
        // into our content space.
        self.content_mut().clear();
        for entry_idx in other.content().iter() {
            let entry = other.content().entry_ref(entry_idx);
            let new_entry = entry.clone();
            let idx= self.content_mut().push(new_entry)?;
            let ops = other.content().ops(entry_idx);
            if ops.len() > 0 {
                self.content_mut().push_op_batch(idx, &ops)?;
            }
        }

        // let dst = self.content_slice_mut();
        // let src = other.content_slice();
        // dst[..src.len()].copy_from_slice(src);

        // // Copy the other content into our content buffer. We checked that
        // // the other content is <= our content, thus this is safe.
        // let dst = self.content_slice_mut();
        // let src = other.content_slice();
        // dst[..src.len()].copy_from_slice(src);

        // // The copied content might stream from a cache that was smaller than us,
        // // hence we need to inform the MutationCacheContent that its backing memory
        // // size might have changed.
        // self.content_mut().update(content_size);

        Ok(())
    }

    /// Removes all elements from the cache that are not listed in the passed
    /// whitelist.
    fn retain_whitelisted(&mut self, whitelist: &[PatchPointID]) {
        self.content_mut().retain(|e| whitelist.contains(&e.id()));
    }

    /// Only retain elements for which `f` returns true.
    /// !!! This will invalidate all references to the contained entries !!!
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&PatchingCacheEntry) -> bool,
    {
        self.content_mut().retain(f);
    }

    /// Only retain MutationCacheEntry's that actually affect the execution.
    /// !!! This will invalidate all references to the contained entries !!!
    pub fn purge_nop_entries(&mut self) {
        self.content_mut().retain(|e| !e.is_nop());
    }

    pub fn iter(&self) -> impl Iterator<Item = &PatchingCacheEntry> {
        PatchingCacheIter::new(self)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut PatchingCacheEntry> {
        PatchingCacheIterMut::new(self)
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
        let mut bm_idx_map = HashMap::new();
        self.content().iter().for_each(|idx| {
            let e = self.content().entry_ref(idx);
            bm_idx_map.insert(e.id(), idx);
        });

        self.union_with_flag(other, PatchingCacheEntryFlags::Tracing, &bm_idx_map)?;
        self.union_with_flag(other, PatchingCacheEntryFlags::TracingWithVal, &bm_idx_map)?;
        self.union_with_flag(other, PatchingCacheEntryFlags::Patching, &bm_idx_map)?;
        self.union_with_flag(other, PatchingCacheEntryFlags::Jumping, &bm_idx_map)?;

        Ok(())
    }

    fn union_with_flag(
        &mut self,
        other: &PatchingCache,
        flag: PatchingCacheEntryFlags,
        bm_idx_map: &HashMap<PatchPointID, usize>,
    ) -> Result<()> {
        log::trace!("Unioning with flag: {:?}", flag);
        for other_idx in other.content().iter() {
            let other_entry = other.content().entry_ref(other_idx);
            if !other_entry.is_flag_set(flag) {
                continue;
            }
            let pp_id = other_entry.id();
            if let Some(idx) = bm_idx_map.get(&pp_id) {
                self.content_mut().entry_mut(*idx).set_flag(flag);

                let entry = self.content().entry_ref(*idx);
                if entry.is_flag_set(PatchingCacheEntryFlags::Patching)
                    && entry.is_flag_set(PatchingCacheEntryFlags::Jumping)
                {
                    return Err(anyhow!(
                        "Patching and jumping are both set for entry (PatchPointID)#{}",
                        pp_id.0
                    ));
                }

                match flag {
                    PatchingCacheEntryFlags::Patching | PatchingCacheEntryFlags::Jumping => {
                        let other_ops = other.content().ops(other_idx);
                        self.content_mut().clear_ops(*idx).unwrap();
                        let head_op_idx =
                            self.content_mut().push_op_batch(*idx, &other_ops).unwrap();
                        self.content_mut().entry_mut(*idx).metadata.op_idx = head_op_idx;
                    }
                    _ => {}
                }
            } else {
                let idx = self.content_mut().push(other_entry.clone()).expect("Failed to push entry");
                self.content_mut().entry_mut(idx).set_flag(flag);
                match flag {
                    PatchingCacheEntryFlags::Patching | PatchingCacheEntryFlags::Jumping => {
                        let ops = other.content().ops(other_idx);
                        self.content_mut().push_op_batch(idx, &ops)?;
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Set the flags of all mutation cache entries to zero.
    pub fn reset_flags(&mut self) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.reset_flags();
        });
        self
    }

    /// Set the passed MutationCacheEntryFlags on all mutation cache entries.
    pub fn set_flag(&mut self, flag: PatchingCacheEntryFlags) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.set_flag(flag);
        });
        self
    }

    /// Clear the MutationCacheEntryFlags from all mutation cache entries.
    pub fn unset_flag(&mut self, flag: PatchingCacheEntryFlags) -> &mut Self {
        self.iter_mut().for_each(|e| {
            e.unset_flag(flag);
        });
        self
    }

    /// Enable tracing for all mutation entries in this set.
    pub fn enable_tracing(&mut self) -> &mut Self {
        self.set_flag(PatchingCacheEntryFlags::Tracing)
    }

    /// Disable tracing for all mutation entries in this set.
    pub fn disable_tracing(&mut self) -> &mut Self {
        self.unset_flag(PatchingCacheEntryFlags::Tracing)
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
    use std::{ffi::CString, mem::transmute, ptr};

    use libc::c_void;
    use rand::{distributions::Alphanumeric, thread_rng, Rng};

    use crate::{
        patching_cache::{PATCHING_CACHE_DEFAULT_ENTRY_SIZE, PATCHING_CACHE_DEFAULT_OP_SIZE},
        patching_cache_content::PatchingCacheContent,
        patching_cache_entry::PatchingCacheEntry,
    };

    use super::PatchingCache;

    fn generate_random_string(length: usize) -> String {
        let rng = thread_rng();
        let random_string: String = rng
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect();
        random_string
    }

    fn dummy_entry(id: u64) -> PatchingCacheEntry {
        PatchingCacheEntry::new(
            id.into(),
            0,
            0,
            llvm_stackmap::LocationType::Constant,
            8,
            crate::dwarf::DwarfReg::Rax,
            0,
            0,
        )
    }

    #[test]
    fn patching_cache_replace() {
        let mut cache = PatchingCache::default();
        for i in 0..100 {
            cache.push(dummy_entry(i as u64)).unwrap();
        }

        let mut other = PatchingCache::new(100, 100).unwrap();
        other.replace(&cache).unwrap();

        assert_eq!(cache.len(), other.len());
    }

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

        cache1.push(dummy_entry(101)).unwrap();

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

    fn test_estimate_memory_occupied() {
        let entry_size = 10000;
        let op_size = 10000;
        let expected_size = PatchingCacheContent::estimate_memory_occupied(entry_size, op_size);
        let actual_size = PatchingCacheContent::estimate_memory_occupied(entry_size, op_size);
        assert_eq!(expected_size, actual_size);
    }
}
