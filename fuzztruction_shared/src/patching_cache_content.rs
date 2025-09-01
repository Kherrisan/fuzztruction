use std::{cell::RefCell, fmt::Debug, mem, rc::Rc};

use crate::{
    patching_cache::{PatchingCache, PatchingCacheEntryFlags},
    patching_cache_entry::{PatchingCacheEntry, PatchingOperation},
    types::PatchPointID,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

// #[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq, Default)]
// pub struct PatchingCacheContentPackage {
//     entry_size: usize,

//     op_size: usize,

//     pub entries: Vec<PatchingCacheEntry>,

//     ops: Vec<(usize, PatchingOperation)>,
// }

// impl From<&PatchingCache> for PatchingCacheContentPackage {
//     fn from(pc: &PatchingCache) -> Self {
//         pc.content().consolidate_package()
//     }
// }

// impl From<PatchingCache> for PatchingCacheContentPackage {
//     fn from(pc: PatchingCache) -> Self {
//         pc.content().consolidate_package()
//     }
// }

// impl Into<PatchingCache> for &PatchingCacheContentPackage {
//     fn into(self) -> PatchingCache {
//         let mut pc = PatchingCache::new(self.entry_size, self.op_size).unwrap();
//         pc.load_package(&self).unwrap();
//         pc
//     }
// }

// impl std::fmt::Debug for PatchingCacheContentPackage {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "PatchingCacheContentPackage {{")?;
//         write!(
//             f,
//             " entry_size: {}, op_size: {}",
//             self.entry_size, self.op_size
//         )?;
//         write!(f, " }}")
//     }
// }
pub struct BitmapIter {
    bitmap_ptr: *const Bitmap,
    current_index: usize,
}

impl Iterator for BitmapIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let bitmap = unsafe { &*self.bitmap_ptr };
        while self.current_index < bitmap.capacity() {
            let index = self.current_index;
            self.current_index += 1;

            if bitmap.is_set(index).unwrap_or(false) {
                return Some(index);
            }
        }
        None
    }
}

#[repr(C)]
#[derive(Debug)]
struct Bitmap {
    size: usize,
    next_free_idx: usize,
    bitmap_offset: isize,
    bitmap_len: usize,
}

impl Bitmap {
    fn iter(&self) -> BitmapIter {
        BitmapIter {
            bitmap_ptr: self as *const Bitmap,
            current_index: 0,
        }
    }

    fn init(&mut self, ptr: *mut u8, len: usize, create: bool) {
        let self_addr = self as *const Bitmap as *const u8;
        self.bitmap_offset = unsafe { ptr.offset_from(self_addr) };
        self.bitmap_len = len;

        if create {
            self.size = 0;
            self.next_free_idx = 0;
            self.clear();
        }
    }

    fn bitmap_ptr(&self) -> *mut u8 {
        let self_addr = self as *const Bitmap as *const u8;
        unsafe { self_addr.offset(self.bitmap_offset) as *mut u8 }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.bitmap_ptr(), self.bitmap_len) }
    }

    fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.bitmap_ptr(), self.bitmap_len) }
    }

    fn is_set(&self, idx: usize) -> Result<bool> {
        if idx >= self.capacity() {
            return Err(anyhow::anyhow!("Index out of bounds: {}", idx));
        }

        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        let slice = self.as_slice();
        let bit = slice[byte_idx] & (1 << bit_idx);
        Ok(bit != 0)
    }

    fn set(&mut self, idx: usize) -> Result<()> {
        if idx >= self.capacity() {
            return Err(anyhow::anyhow!("Index out of bounds: {}", idx));
        }

        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        let slice = self.as_slice_mut();
        slice[byte_idx] |= 1 << bit_idx;

        self.next_free_idx = (idx + 1) % self.capacity();
        self.size += 1;
        Ok(())
    }

    fn clear_at(&mut self, idx: usize) -> Result<()> {
        if idx >= self.capacity() {
            return Err(anyhow::anyhow!("Index out of bounds: {}", idx));
        }

        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        let slice = self.as_slice_mut();
        slice[byte_idx] &= !(1 << bit_idx);

        self.next_free_idx = idx % self.capacity();
        self.size -= 1;
        Ok(())
    }

    fn clear(&mut self) {
        let slice = self.as_slice_mut();
        slice.fill(0);
        self.next_free_idx = 0;
        self.size = 0;
    }

    fn find_n_clear(&mut self, n: usize) -> Result<Vec<usize>> {
        if n == 0 {
            return Ok(vec![]);
        }
        let mut n = n;
        let mut ret = vec![];
        for i in 0..self.capacity() {
            if !self.is_set(i)? {
                ret.push(i);
                n -= 1;
                if n == 0 {
                    break;
                }
            }
        }
        if n > 0 {
            return Err(anyhow::anyhow!("Not enough free slots found"));
        }
        Ok(ret)
    }

    fn find_first_clear(&mut self) -> Result<usize> {
        let marker = self.next_free_idx;
        while self.is_set(self.next_free_idx)? {
            self.next_free_idx = (self.next_free_idx + 1) % self.capacity();
            if self.next_free_idx == marker {
                return Err(anyhow::anyhow!("No free slot found"));
            }
        }
        let free_idx = self.next_free_idx;
        self.next_free_idx = (self.next_free_idx + 1) % self.capacity();
        Ok(free_idx)
    }

    fn capacity(&self) -> usize {
        self.bitmap_len * 8
    }

    fn size(&self) -> usize {
        self.size
    }

    pub fn print(&self, start: usize, length: usize) {
        let slice = self.as_slice();
        println!("Bitmap, start: {}, length: {}", start, length);
        for i in start..start + length {
            if i % 16 == 0 {
                print!("0x{:x}: ", (i - start) / 8 + self.bitmap_ptr() as usize);
            }
            print!("{:02x} ", slice[i]);
            if i % 16 == 15 {
                println!();
            }
        }
        println!();
    }
}

#[repr(C)]
pub struct PatchingCacheContent {
    /// Size of the memory region backing this instance (i.e., the memory &self points to).
    total_size: usize,

    entry_data_offset: usize,

    op_data_offset: usize,

    entry_table_size: usize,

    op_table_size: usize,

    entry_bitmap: Bitmap,

    op_bitmap: Bitmap,

    data: [u8; 0],
}

impl std::fmt::Debug for PatchingCacheContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PatchingCacheContent {{")?;
        write!(f, "data address: 0x{:x}, ", self.data.as_ptr() as usize)?;
        write!(f, "total_size: {}, entry_data_offset: {}, op_data_offset: {}, entry_table_size: {}, op_table_size: {}", self.total_size, self.entry_data_offset, self.op_data_offset, self.entry_table_size, self.op_table_size)?;
        write!(f, " }}")
    }
}

impl PatchingCacheContent {
    pub fn entry_table_size(&self) -> usize {
        self.entry_table_size
    }
    pub fn op_table_size(&self) -> usize {
        self.op_table_size
    }

    pub fn iter(&self) -> BitmapIter {
        self.entry_bitmap.iter()
    }

    pub fn entry_data_offset(&self) -> usize {
        self.entry_data_offset
    }

    pub fn op_data_offset(&self) -> usize {
        self.op_data_offset
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    // pub fn clean_load_patching_table(
    //     &mut self,
    //     ctx: Rc<RefCell<AnalysisContext>>,
    //     table: &PatchingTable,
    // ) -> Result<()> {
    //     self.clear();

    //     self.init(table.entry_size, table.op_size, true);

    //     for (i, (pp_id, ops)) in table.tbl.iter().enumerate() {
    //         *self.entry_mut(i) = entry.clone();
    //         self.entry_bitmap.set(i)?;
    //     }

    //     for (idx, op) in package.ops.iter() {
    //         *self.op_mut(*idx) = op.clone();
    //         self.op_bitmap.set(*idx)?;
    //     }

    //     Ok(())
    // }

    #[inline]
    unsafe fn entry_data_ptr<T>(&self) -> *const T {
        self.data.as_ptr().offset(self.entry_data_offset as isize) as *const T
    }

    #[inline]
    unsafe fn op_data_ptr<T>(&self) -> *const T {
        self.data.as_ptr().offset(self.op_data_offset as isize) as *const T
    }

    #[inline]
    unsafe fn entry_ptr(&self, idx: usize) -> *const PatchingCacheEntry {
        let addr = self.entry_data_ptr::<u8>();
        let addr = addr.offset((idx * mem::size_of::<PatchingCacheEntry>()) as isize);
        addr as *const PatchingCacheEntry
    }

    #[inline]
    pub fn entry_ref(&self, idx: usize) -> &PatchingCacheEntry {
        unsafe {
            let addr = self.entry_data_ptr::<u8>();
            let addr = addr.offset((idx * mem::size_of::<PatchingCacheEntry>()) as isize);
            &*(addr as *const PatchingCacheEntry)
        }
    }

    #[inline]
    pub fn entry_mut(&self, idx: usize) -> &mut PatchingCacheEntry {
        let addr = unsafe { self.entry_data_ptr::<u8>() };
        unsafe {
            let addr = addr.offset((idx * mem::size_of::<PatchingCacheEntry>()) as isize);
            &mut *(addr as *mut PatchingCacheEntry)
        }
    }

    #[inline]
    pub fn op_ref(&self, idx: usize) -> &PatchingOperation {
        unsafe {
            let addr = self
                .op_data_ptr::<u8>()
                .offset((idx * mem::size_of::<PatchingOperation>()) as isize);
            &*(addr as *const PatchingOperation)
        }
    }

    #[inline]
    unsafe fn op_ptr(&self, idx: usize) -> *const PatchingOperation {
        let addr = self
            .op_data_ptr::<u8>()
            .offset((idx * mem::size_of::<PatchingOperation>()) as isize);
        addr as *const PatchingOperation
    }

    #[inline]
    pub fn op_mut(&self, idx: usize) -> &mut PatchingOperation {
        unsafe {
            let addr = self
                .op_data_ptr::<u8>()
                .offset((idx * mem::size_of::<PatchingOperation>()) as isize);
            &mut *(addr as *mut PatchingOperation)
        }
    }

    pub fn estimate_memory_occupied(entry_size: usize, op_size: usize) -> usize {
        let entry_bitmap_len = entry_size.div_ceil(8);
        let op_bitmap_len = op_size.div_ceil(8);

        // 计算内存布局，与 init 方法完全一致
        let entry_align = mem::align_of::<PatchingCacheEntry>();
        let op_align = mem::align_of::<PatchingOperation>();
        let max_align = std::cmp::max(
            std::cmp::max(entry_align, op_align), 
            mem::align_of::<PatchingCacheContent>()
        );

        // 基础结构大小（包括 PatchingCacheContent 本身）
        let base_size = mem::size_of::<Self>();

        // 计算 entry 数据偏移（从 data 开始计算，考虑对齐）
        let entry_data_offset = (entry_bitmap_len + entry_align - 1) & !(entry_align - 1);

        // 计算 op bitmap 偏移
        let op_bitmap_offset = entry_data_offset + entry_size * mem::size_of::<PatchingCacheEntry>();

        // 计算 op 数据偏移（考虑对齐）
        let op_data_offset = (op_bitmap_offset + op_bitmap_len + op_align - 1) & !(op_align - 1);

        // 总大小：从 PatchingCacheContent 开始到 op 数据结束
        let total_size = base_size + op_data_offset + op_size * mem::size_of::<PatchingOperation>();

        // 确保对齐到最大对齐要求（与 shm_open 一致）
        let aligned_size = (total_size + max_align - 1) & !(max_align - 1);
        
        // 确保最小内存大小不小于 PatchingCacheContent 本身的大小
        std::cmp::max(aligned_size, mem::size_of::<Self>())
    }

    /// Initialize the content.
    /// NOTE: This function must be called before any other!
    pub fn init(&mut self, entry_size: usize, op_size: usize, create: bool) {
        let entry_bitmap_len = entry_size.div_ceil(8);
        let op_bitmap_len = op_size.div_ceil(8);

        // Memory layout
        // - data address (aligned)
        // - entry bitmap [u8]
        // - padding for PatchingCacheEntry alignment
        // - [PatchingCacheEntry]
        // - op bitmap [u8]
        // - padding for PatchingOperation alignment
        // - [PatchingOperation]

        // Since the shared memory base address is now guaranteed to be aligned
        // to the maximum alignment requirement, we can use simpler calculations

        // Calculate entry data offset with proper alignment
        let entry_align = mem::align_of::<PatchingCacheEntry>();
        self.entry_data_offset = (entry_bitmap_len + entry_align - 1) & !(entry_align - 1);

        // Calculate op bitmap offset
        let op_bitmap_offset =
            self.entry_data_offset + entry_size * mem::size_of::<PatchingCacheEntry>();

        // Calculate op data offset with proper alignment
        let op_align = mem::align_of::<PatchingOperation>();
        self.op_data_offset = (op_bitmap_offset + op_bitmap_len + op_align - 1) & !(op_align - 1);

        // #[cfg(debug_assertions)]
        // {
        //     println!(
        //         "mem::size_of::<PatchingCacheEntry>(): {}",
        //         mem::size_of::<PatchingCacheEntry>()
        //     );
        //     println!(
        //         "mem::align_of::<PatchingCacheEntry>(): {}",
        //         mem::align_of::<PatchingCacheEntry>()
        //     );
        //     println!(
        //         "mem::size_of::<PatchingOperation>(): {}",
        //         mem::size_of::<PatchingOperation>()
        //     );
        //     println!(
        //         "mem::align_of::<PatchingOperation>(): {}",
        //         mem::align_of::<PatchingOperation>()
        //     );
        //     println!("data: {:p}", self.data.as_ptr());
        //     println!("data + entry_data_offset: {:p}", unsafe {
        //         self.data.as_ptr().offset(self.entry_data_offset as isize)
        //     });
        //     println!("data + op_bitmap_offset: {:p}", unsafe {
        //         self.data.as_ptr().offset(op_bitmap_offset as isize)
        //     });
        //     println!("data + op_data_offset: {:p}", unsafe {
        //         self.data.as_ptr().offset(self.op_data_offset as isize)
        //     });

        //     // Verify alignment
        //     let entry_ptr = unsafe { self.data.as_ptr().offset(self.entry_data_offset as isize) };
        //     let op_ptr = unsafe { self.data.as_ptr().offset(self.op_data_offset as isize) };
        //     assert_eq!(
        //         entry_ptr as usize % entry_align,
        //         0,
        //         "Entry data not aligned"
        //     );
        //     assert_eq!(op_ptr as usize % op_align, 0, "Op data not aligned");

        //     println!(
        //         "Entry data alignment verified: {}",
        //         entry_ptr as usize % entry_align == 0
        //     );
        //     println!(
        //         "Op data alignment verified: {}",
        //         op_ptr as usize % op_align == 0
        //     );
        // }

        self.entry_table_size = entry_size;
        self.op_table_size = op_size;

        self.entry_bitmap
            .init(self.data.as_mut_ptr(), entry_bitmap_len, create);
        self.op_bitmap.init(
            unsafe { self.data.as_mut_ptr().offset(op_bitmap_offset as isize) },
            op_bitmap_len,
            create,
        );
    }

    fn entry_space_size(&self) -> usize {
        self.entry_table_size * mem::size_of::<PatchingCacheEntry>()
    }

    fn entry_space_left(&self) -> usize {
        self.entry_space_size() - self.entry_bitmap.size() * mem::size_of::<PatchingCacheEntry>()
    }

    fn op_space_size(&self) -> usize {
        self.op_table_size * mem::size_of::<PatchingOperation>()
    }

    fn op_space_left(&self) -> usize {
        self.op_space_size() - self.op_bitmap.size() * mem::size_of::<PatchingOperation>()
    }

    pub fn clear(&mut self) {
        self.entry_bitmap.clear();
        self.op_bitmap.clear();
    }

    pub fn ops(&self, idx: usize) -> Vec<PatchingOperation> {
        let mut ops = vec![];
        let mut op_idx = self.entry_ref(idx).op_head_idx;
        while let Some(idx) = op_idx {
            ops.push(self.op_ref(idx).clone());
            op_idx = self.op_ref(idx).next_idx;
        }
        ops
    }

    pub fn find_ops(&self, id: PatchPointID) -> Result<Vec<&PatchingOperation>> {
        let entry = self.find_entry(id);
        if entry.is_none() {
            return Err(anyhow::anyhow!("Entry not found"));
        }
        let (_, entry) = entry.unwrap();
        let mut ops = vec![];
        let mut op_idx = entry.op_head_idx;
        while let Some(idx) = op_idx {
            ops.push(self.op_ref(idx));
            op_idx = self.op_ref(idx).next_idx;
        }
        Ok(ops)
    }

    /// NOTE: The returned referencers are only valid as long as no entries are added
    /// or removed.
    fn entries_raw(&self) -> Vec<*const PatchingCacheEntry> {
        let mut ret = Vec::new();
        self.entry_bitmap.iter().for_each(|idx| {
            unsafe {
                let addr = self.entry_ptr(idx);
                ret.push(addr);
            };
        });
        ret
    }

    fn allocate_entry(&mut self) -> Result<(usize, &mut PatchingCacheEntry)> {
        let free_idx = self.entry_bitmap.find_first_clear()?;

        if mem::size_of::<PatchingCacheEntry>() * (free_idx + 1) > self.entry_space_size() {
            log::error!("Found free entry slot, but not enough space to store the entry");
            return Err(anyhow::anyhow!(
                "Found free entry slot, but not enough space to store the entry"
            ));
        }

        self.entry_bitmap.set(free_idx)?;

        let entry_mut = self.entry_mut(free_idx);

        Ok((free_idx, entry_mut))
    }

    fn allocate_op(&mut self) -> Result<(usize, &mut PatchingOperation)> {
        let free_idx = self.op_bitmap.find_first_clear()?;

        if mem::size_of::<PatchingOperation>() * (free_idx + 1) > self.op_space_size() {
            log::error!("Found free op slot, but not enough space to store the operation");
            return Err(anyhow::anyhow!(
                "Found free op slot, but not enough space to store the operation"
            ));
        }

        self.op_bitmap.set(free_idx)?;

        let op_mut = self.op_mut(free_idx);

        Ok((free_idx, op_mut))
    }

    pub fn reset_all_entry_op(&mut self) {
        self.entry_bitmap.iter().for_each(|idx| {
            let entry = self.entry_mut(idx);
            entry.metadata.op_idx = entry.op_head_idx;
        });
    }

    pub fn push_op_batch(
        &mut self,
        idx: usize,
        ops: &Vec<PatchingOperation>,
    ) -> Result<Option<usize>> {
        if ops.len() == 0 {
            return Ok(None);
        }

        if ops.len() * mem::size_of::<PatchingOperation>() > self.op_space_left() {
            return Err(anyhow::anyhow!("Not enough space to store the operations"));
        }

        let free_op_indices = self.op_bitmap.find_n_clear(ops.len())?;

        if free_op_indices.len() * mem::size_of::<PatchingOperation>() > self.op_space_left() {
            return Err(anyhow::anyhow!("Not enough space to store the operations"));
        }

        let mut prev_op_idx = None;
        for (i, op) in ops.iter().enumerate() {
            let op_mut = self.op_mut(free_op_indices[i]);
            *op_mut = *op;
            self.op_bitmap.set(free_op_indices[i])?;
            if prev_op_idx.is_none() {
                prev_op_idx = Some(free_op_indices[i]);
            } else {
                self.op_mut(prev_op_idx.unwrap()).next_idx = Some(free_op_indices[i]);
            }
        }

        let entry = self.entry_mut(idx);
        if entry.op_head_idx.is_none() {
            // The entry does not have any operations before.
            entry.op_head_idx = Some(free_op_indices[0]);
            entry.op_tail_idx = Some(free_op_indices[ops.len() - 1]);
        } else {
            let tail_op_idx = entry.op_tail_idx.unwrap();
            entry.op_tail_idx = Some(free_op_indices[0]);
            self.op_mut(tail_op_idx).next_idx = Some(free_op_indices[0]);
        }

        Ok(Some(free_op_indices[0]))
    }

    pub fn push_op(&mut self, idx: usize, op: &PatchingOperation) -> Result<usize> {
        // Check if the entry of idx exists.
        if !self.entry_bitmap.is_set(idx)? {
            return Err(anyhow::anyhow!(format!("Entry of idx {} not found", idx)));
        }

        let entry = self.entry_ref(idx);

        if entry.op_head_idx.is_none() {
            // The operation to be pushed is the first one.
            let (op_idx, op_mut) = self.allocate_op()?;
            *op_mut = *op;
            let entry = self.entry_mut(idx);
            entry.op_head_idx = Some(op_idx);
            entry.op_tail_idx = Some(op_idx);
            Ok(op_idx)
        } else {
            let tail_op_idx = entry.op_tail_idx.unwrap();
            let (op_idx, op_mut) = self.allocate_op()?;
            *op_mut = *op;
            self.op_mut(tail_op_idx).next_idx = Some(op_idx);
            self.entry_mut(idx).op_tail_idx = Some(op_idx);
            Ok(op_idx)
        }
    }

    /// NOTE: The returned referencers are only valid as long as no entries are added
    /// or removed.
    pub fn entries(&self) -> Vec<&PatchingCacheEntry> {
        self.entries_raw()
            .into_iter()
            .map(|e| unsafe { &*e })
            .collect()
    }

    /// NOTE: The returned referencers are only valid as long as no entries are added
    /// or removed.
    pub fn entries_mut(&mut self) -> Vec<&mut PatchingCacheEntry> {
        self.entries_raw()
            .into_iter()
            .map(|e| unsafe { &mut *(e as *mut PatchingCacheEntry) })
            .collect()
    }

    pub fn push(&mut self, entry: PatchingCacheEntry) -> Result<usize> {
        if entry.size() > self.entry_space_left() {
            return Err(anyhow::anyhow!("Not enough space to store the entry"));
        }

        let (idx, entry_mut) = self.allocate_entry()?;
        *entry_mut = entry;
        entry_mut.op_head_idx = None;
        entry_mut.op_tail_idx = None;

        Ok(idx)
    }

    pub fn find_entry(&self, id: PatchPointID) -> Option<(usize, &PatchingCacheEntry)> {
        self.entry_bitmap
            .iter()
            .find(|idx| {
                let entry = self.entry_ref(*idx);
                entry.id() == id
            })
            .map(|idx| (idx, self.entry_ref(idx)))
    }

    fn find_entry_mut(&mut self, id: PatchPointID) -> Option<(usize, &mut PatchingCacheEntry)> {
        self.entry_bitmap
            .iter()
            .find(|idx| {
                let entry = self.entry_ref(*idx);
                entry.id() == id
            })
            .map(|idx| (idx, self.entry_mut(idx)))
    }

    pub fn remove_op(&mut self, id: PatchPointID, op_idx: usize) -> Result<()> {
        let entry = self.find_entry(id);
        if entry.is_none() {
            return Err(anyhow::anyhow!(
                "Entry with patchpoint id {:?} not found",
                id
            ));
        }
        let (_, entry) = entry.unwrap();
        if entry.op_head_idx.is_none() {
            return Err(anyhow::anyhow!(
                "Entry with patchpoint id {:?} has no operations",
                id
            ));
        }

        // Find the prev slot and the slot to be removed.
        let mut prev_idx = None;
        let mut current_idx = entry.op_head_idx;
        for i in 0..op_idx {
            if let Some(idx) = current_idx {
                if self.op_bitmap.is_set(idx).unwrap() {
                    prev_idx = Some(idx);
                    current_idx = self.op_ref(idx).next_idx;
                } else {
                    return Err(anyhow::anyhow!(
                            "Operation at index {} of patchpoint id {:?} does not exist in the descriptor table",
                            i,
                            op_idx
                        ));
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Operation at index {:?} of patchpoint id {:?} does not have next slot",
                    i - 1,
                    op_idx
                ));
            }
        }

        if current_idx.is_none() {
            return Err(anyhow::anyhow!(
                "Operation at index {:?} of patchpoint id {:?} does not exist",
                id,
                op_idx
            ));
        }

        let next_idx = if let Some(idx) = current_idx {
            self.op_ref(idx).next_idx
        } else {
            return Err(anyhow::anyhow!(
                "Operation at index {:?} of patchpoint id {:?} descriptor not found in table",
                op_idx,
                id
            ));
        };

        self.op_bitmap.clear_at(current_idx.unwrap())?;

        // The operation to be removed is the first one.
        if prev_idx.is_none() {
            self.find_entry_mut(id).unwrap().1.op_head_idx = next_idx;
        }

        // The operation to be removed is the last one.
        if next_idx.is_none() {
            self.find_entry_mut(id).unwrap().1.op_tail_idx = prev_idx;
        }

        if prev_idx.is_some() && next_idx.is_some() {
            if self.op_bitmap.is_set(prev_idx.unwrap())? {
                self.op_mut(prev_idx.unwrap()).next_idx = next_idx;
            } else {
                return Err(anyhow::anyhow!(
                    "Prev operation of the operation at index {:?} of patchpoint id {:?} not found in table",
                    op_idx,
                    id
                ));
            }
        }

        return Ok(());
    }

    pub fn clear_entry_ops(&mut self, idx: usize) -> Result<()> {
        let entry = self.entry_mut(idx);
        let mut op_idx = entry.op_head_idx;
        while let Some(idx) = op_idx {
            self.op_bitmap.clear_at(idx)?;
            op_idx = self.op_ref(idx).next_idx;
        }
        let entry = self.entry_mut(idx);
        entry.op_head_idx = None;
        entry.op_tail_idx = None;

        Ok(())
    }

    pub fn remove_entry_idx(&mut self, idx: usize) -> Result<()> {
        self.clear_entry_ops(idx)?;
        self.entry_bitmap.clear_at(idx)?;

        Ok(())
    }

    pub fn retain<F>(&mut self, mut f: F) -> usize
    where
        F: FnMut(&PatchingCacheEntry) -> bool,
    {
        let mut removed_count = 0;
        self.entry_bitmap.iter().for_each(|idx| {
            if !f(self.entry_ref(idx)) {
                self.remove_entry_idx(idx).unwrap();
                removed_count += 1;
            }
        });

        removed_count
    }

    pub fn remove(&mut self, id: PatchPointID) -> Result<usize> {
        let res = self.find_entry(id);
        if res.is_none() {
            return Err(anyhow::anyhow!("Entry not found"));
        }

        let (idx, _) = res.unwrap();

        self.clear_entry_ops(idx)?;
        self.remove_entry_idx(idx)?;

        Ok(idx)
    }

    pub fn op_count(&self) -> usize {
        self.op_bitmap.size()
    }

    pub fn entry_count(&self) -> usize {
        self.entry_bitmap.size()
    }

    fn print_entries(&self) {
        println!("entries: ");
        self.entry_bitmap.iter().for_each(|idx| {
            let entry = self.entry_ref(idx);
            println!("#{idx}: {:?}", entry);
        });
        println!();
    }

    fn print_ops(&self) {
        println!("ops: ");
        self.op_bitmap.iter().for_each(|idx| {
            let op = self.op_ref(idx);
            println!("#{idx}: {:?}", op);
        });
        println!();
    }

    pub fn print_entry_bitmap(&self, start: usize, length: usize) {
        self.entry_bitmap.print(start, length);
    }

    pub fn print_op_bitmap(&self, start: usize, length: usize) {
        self.op_bitmap.print(start, length);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        patching_cache::{PatchingCache, PatchingCacheEntryFlags},
        util,
    };
    use std::assert_matches::assert_matches;

    fn dummy_op(operand: u64) -> PatchingOperation {
        PatchingOperation {
            op: crate::patching_cache_entry::PatchingOperator::Add,
            operand,
            next_idx: None,
        }
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
    fn test_push_remove() {
        let size = PatchingCacheContent::estimate_memory_occupied(100, 100);
        let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        content.init(100, 100, true);

        let init_space_left = content.entry_space_left();

        // Test if empty
        let ret = content.entries();
        assert_eq!(ret.len(), 0);

        // Test push e0
        let mut e0 = dummy_entry(0);
        let _ret = content.push(e0.clone()).expect("Failed to push e0");
        assert_eq!(content.entry_count(), 1);
        assert_eq!(content.entries().len(), 1);

        // Test push e1
        let mut e1 = dummy_entry(1);
        let _ret = content.push(e1.clone()).expect("Failed to push e1");
        assert_eq!(content.entries().len(), 2);

        // Test query e1
        let (_, e1_ref) = content.find_entry(e1.id()).expect("Failed to find e1");
        assert_eq!(e1_ref.id(), e1.id());

        // Test push e0 with op
        let op = dummy_op(1);
        content.push_op(0, &op).expect("Failed to push op");
        let (_, e0_ref) = content.find_entry(e0.id()).expect("Failed to find e0");
        assert_eq!(e0_ref.op_head_idx.is_some(), true);
        assert_eq!(e0_ref.op_head_idx.unwrap(), 0);
        assert_eq!(content.op_count(), 1);
        assert_eq!(content.entry_count(), 2);

        let (e1_idx, _) = content.find_entry(e1.id()).expect("Failed to find e1");
        content
            .push_op(e1_idx, &dummy_op(0))
            .expect("Failed to push op");
        content
            .push_op(e1_idx, &dummy_op(1))
            .expect("Failed to push op");
        content
            .push_op(e1_idx, &dummy_op(2))
            .expect("Failed to push op");
        assert_eq!(content.op_count(), 4);
        assert_eq!(content.entry_count(), 2);
        content.print_entries();
        content.print_ops();

        content.remove(e0.id()).expect("Failed to remove e0");
        assert_eq!(content.entries().len(), 1);
        assert_eq!(content.op_count(), 3);
        assert_eq!(content.entry_count(), 1);
        content.print_entries();
        content.print_ops();

        let ops = content.find_ops(e1.id()).expect("Failed to find ops");
        let (_, e1_ref) = content.find_entry(e1.id()).expect("Failed to find e1");
        assert_eq!(e1_ref.op_head_idx.is_some(), true);
        assert_eq!(e1_ref.op_head_idx.unwrap(), 1);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].operand, 0);
        assert_eq!(ops[1].operand, 1);
        assert_eq!(ops[2].operand, 2);
        content.print_entries();
        content.print_ops();

        content.remove_op(e1.id(), 1).expect("Failed to remove op");
        let ops = content.find_ops(e1.id()).expect("Failed to find ops");
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].operand, 0);
        assert_eq!(ops[1].operand, 2);
        let e1_ref = content.find_entry(e1.id()).unwrap().1;
        assert_eq!(e1_ref.op_head_idx.unwrap(), 1);
        assert_eq!(e1_ref.op_tail_idx.unwrap(), 3);
        println!("after remove index 1 ops: ");
        content.print_entries();
        content.print_ops();

        content.remove_op(e1.id(), 0).expect("Failed to remove op");
        let ops = content.find_ops(e1.id()).expect("Failed to find ops");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].operand, 2);
        println!("after remove index 0 ops: {:?}", ops);
        content.print_entries();
        content.print_ops();

        content.remove_op(e1.id(), 0).expect("Failed to remove op");
        let ops = content.find_ops(e1.id()).expect("Failed to find ops");
        assert_eq!(ops.len(), 0);
        assert_eq!(content.op_count(), 0);
        let e1_ref = content.find_entry(e1.id()).unwrap().1;
        assert_eq!(e1_ref.op_head_idx.is_none(), true);
        assert_eq!(e1_ref.op_tail_idx.is_none(), true);
        println!("after remove index 0 ops: ");
        content.print_entries();
        content.print_ops();

        // Test query e0 after remove
        let e0_ref = content.find_entry(e0.id());
        assert_eq!(e0_ref.is_none(), true);

        content
            .push_op(e1_idx, &dummy_op(0))
            .expect("Failed to push op");
        content
            .push_op(e1_idx, &dummy_op(5))
            .expect("Failed to push op");
        content
            .push_op(e1_idx, &dummy_op(6))
            .expect("Failed to push op");
        content.remove_op(e1.id(), 0).expect("Failed to remove op");
        println!("after push ops on e1: ");
        content.print_entries();
        content.print_ops();

        content.remove(e1.id()).expect("Failed to remove e1");
        assert_eq!(content.entries().len(), 0);
        // content.consolidate();
        assert_eq!(content.entry_space_left(), init_space_left);
    }

    #[test]
    fn test_union() {
        let mut cache = PatchingCache::new(100, 100).unwrap();

        for i in 0..10 {
            let mut e = dummy_entry(i as u64);
            e.set_flag(PatchingCacheEntryFlags::Patching);
            cache.content_mut().push(e).expect("Failed to push entry");
            for j in 0..i {
                cache
                    .content_mut()
                    .push_op(i, &dummy_op(j as u64))
                    .expect("Failed to push op");
            }
        }

        let mut another_cache = PatchingCache::new(100, 100).unwrap();
        for i in 10..20 {
            let mut e = dummy_entry(i as u64);
            e.set_flag(PatchingCacheEntryFlags::Patching);
            another_cache
                .content_mut()
                .push(e)
                .expect("Failed to push entry");
        }
        for i in 10..15 {
            let (idx, _) = another_cache
                .content()
                .find_entry(PatchPointID::from(i as u64))
                .unwrap();
            another_cache
                .content_mut()
                .entry_mut(idx)
                .set_flag(PatchingCacheEntryFlags::Tracing);
        }

        cache.union(&another_cache).expect("Failed to union");
        assert_eq!(cache.content().entry_count(), 20);
        cache.content().print_entries();
        cache.content().print_ops();
    }

    #[test]
    fn test_max_size() {
        let size = PatchingCacheContent::estimate_memory_occupied(100, 100);
        let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        content.init(100, 100, true);

        for i in 0..100 {
            let e = dummy_entry(i as u64);
            println!("pushing entry: {:?}", i);
            assert_matches!(content.push(e), Ok(..));
        }

        let e = dummy_entry(u64::MAX);
        assert_matches!(content.push(e), Err(..));

        let (e0_idx, e0) = content.find_entry(PatchPointID::from(0u32)).unwrap();
        let e0_id = e0.id();
        for i in 0..100 {
            assert_matches!(content.push_op(e0_idx, &dummy_op(i as u64)), Ok(..));
        }

        content.remove(e0_id).expect("Failed to remove entry");
        assert_eq!(content.entry_count(), 99);
        assert_eq!(content.op_count(), 0);

        assert_matches!(content.push(dummy_entry(101)), Ok(..));
        assert_matches!(content.push(dummy_entry(1)), Err(..));
    }

    #[test]
    fn test_consolidate_and_restore() {
        let size = PatchingCacheContent::estimate_memory_occupied(100, 100);
        let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        content.init(100, 100, true);

        for i in 0..10 {
            let e = dummy_entry(i as u64);
            let idx = content.push(e.clone()).expect("Failed to push entry");
            for j in 0..i {
                content
                    .push_op(idx, &dummy_op(j as u64))
                    .expect("Failed to push op");
            }
            for j in 0..(i / 2) {
                content.remove_op(e.id(), 0).expect("Failed to remove op");
            }
        }

        content.print_entries();
        content.print_ops();

        let entry_cnt = content.entry_count();
        let op_cnt = content.op_count();
        println!(
            "before consolidation, entry_cnt: {}, op_cnt: {}",
            entry_cnt, op_cnt
        );
        // let package: PatchingTable = content.as_ref().into();
        // println!("package: {:#?}", package);

        // let size = PatchingCacheContent::estimate_memory_occupied(100, 100);
        // let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        // content.init(100, 100, true);

        // content
        //     .clean_load_patching_table(&package)
        //     .expect("Failed to load package");

        // assert_eq!(content.entry_count(), entry_cnt);
        // assert_eq!(content.op_count(), op_cnt);

        // content.print_entries();
        // content.print_ops();

        // println!(
        //     "after consolidation, entry_cnt: {}, op_cnt: {}",
        //     content.entry_count(),
        //     content.op_count()
        // );

        // for i in 0..entry_cnt {
        //     let e = content.find_entry(PatchPointID::from(i as u64)).unwrap().1;
        //     let ops = content.find_ops(e.id()).expect("Failed to find ops");
        //     println!("ops: {:?}", ops);
        //     assert_eq!(ops.len(), i - i / 2);
        //     for j in (i / 2)..i {
        //         assert_eq!(ops[j - i / 2].operand, j as u64);
        //     }
        // }
    }
}
