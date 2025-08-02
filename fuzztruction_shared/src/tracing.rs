use anyhow::{anyhow, Result};
use core::slice;
use serde::{Deserialize, Serialize};
use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet},
    ffi::CString,
    fmt::{self, Display},
    fs::File,
    io::{self, Write},
    mem::size_of,
};

use crate::{constants::ENV_FT_SET_SHM, types::PatchPointID};

use super::shared_memory::MmapShMem;

pub const ENV_FT_TRACE_SHM: &str = "TRACE";
pub const DEFAULT_TRACE_MAP_LEN: usize = 0x10000;

#[derive(Clone)]
pub struct TraceVector {
    // TODO:
    // struct field and impl should be consistent with StdMapObserver
    header: *mut TraceVectorHeader,
    _shared_memory: MmapShMem,
}

unsafe impl Send for TraceVector {}

unsafe impl Sync for TraceVector {}

pub struct SyncTraceMap(pub UnsafeCell<TraceVector>);

unsafe impl Sync for SyncTraceMap {}

impl Serialize for TraceVector {
    fn serialize<S>(&self, _serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // TODO:
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for TraceVector {
    fn deserialize<D>(_deserializer: D) -> std::prelude::v1::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // TODO:
        unimplemented!()
    }
}

impl std::fmt::Debug for TraceVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TraceVector {{ header: {:#x?}, shared_memory: {:#x?} }}",
            self.header, self._shared_memory
        )
    }
}

impl TraceVector {
    pub fn dump(&self, path: &str, bytes: usize) -> Result<()> {
        assert!(bytes <= self.memory_capacity());
        assert!(bytes >= size_of::<TraceVectorHeader>());

        let mut file = File::create(path)?;
        let mut buffer = vec![0; bytes];
        // First dump the header
        let header_slice = unsafe {
            std::slice::from_raw_parts(
                self.header as *const u8,
                std::mem::size_of::<TraceVectorHeader>(),
            )
        };
        file.write_all(header_slice)?;

        let mut offset = size_of::<TraceVectorHeader>();
        // Then dump the remaining bytes
        while offset < bytes {
            let read_size = std::cmp::min(buffer.len(), bytes - offset);
            let slice = unsafe {
                std::slice::from_raw_parts((self.header as *const u8).add(offset), read_size)
            };
            buffer[..read_size].copy_from_slice(slice);
            file.write_all(&buffer[..read_size])?;
            offset += read_size;
        }
        Ok(())
    }

    pub fn frequent_set(&self, threshold: usize) -> Vec<u32> {
        let mut trace_cnt = self.hit_counts();
        trace_cnt.retain(|_, cnt| *cnt as usize > threshold);
        trace_cnt.keys().cloned().collect::<Vec<_>>()
    }

    pub fn memory_ratio(&self) -> f64 {
        (self.len() * size_of::<TraceVectorEntry>() + size_of::<TraceVectorHeader>()) as f64
            / self.memory_capacity() as f64
    }

    pub fn hit_set(&self) -> HashSet<u32> {
        self.entries_slice().iter().map(|e| e.id).collect()
    }

    pub fn hit_counts(&self) -> HashMap<u32, u32> {
        let mut counts = HashMap::new();
        for entry in self.entries_slice() {
            *counts.entry(entry.id).or_insert(0) += 1;
        }
        counts
    }

    pub fn memory_capacity(&self) -> usize {
        self._shared_memory.size()
    }

    fn header(&self) -> &TraceVectorHeader {
        unsafe { &*self.header }
    }

    fn header_mut(&mut self) -> &mut TraceVectorHeader {
        unsafe { &mut *self.header }
    }

    pub fn new(len: usize, tag: &str) -> Result<TraceVector> {
        log::trace!("Creating trace vector with length: {}, tag: {}", len, tag);
        let total_size = size_of::<TraceVectorHeader>() + len * size_of::<TraceVectorEntry>();
        let label = format!("trace_{}", tag);
        let mut shared_memory = MmapShMem::new_shmem(total_size, &label)?;
        let header = TraceVectorHeader::new(true, total_size, &mut shared_memory)?;

        let env_key = format!("{}_{}", ENV_FT_TRACE_SHM, tag.to_uppercase());
        shared_memory.write_to_env(&env_key)?;

        Ok(TraceVector {
            header,
            _shared_memory: shared_memory,
        })
    }

    pub fn from_env(tag: &str) -> Result<TraceVector> {
        let env_key = format!("{}_{}", ENV_FT_TRACE_SHM, tag.to_uppercase());
        let mut shared_memory = match MmapShMem::shmem_from_env(&env_key) {
            Ok(shm) => shm,
            Err(_) => {
                let err = io::Error::last_os_error();
                log::error!("Last io os error: {}", err);
                return Err(anyhow!(err).context(format!("When getting trace shm: {}", env_key)));
            }
        };
        let len =
            (shared_memory.size() - size_of::<TraceVectorHeader>()) / size_of::<TraceVectorEntry>();
        let header = TraceVectorHeader::new(false, len, &mut shared_memory)?;

        Ok(TraceVector {
            header,
            _shared_memory: shared_memory,
        })
    }

    pub fn first(&self) -> Option<&TraceVectorEntry> {
        self.entry(0)
    }

    pub fn last(&self) -> Option<&TraceVectorEntry> {
        if self.len() == 0 {
            return None;
        }
        self.entry(self.len() - 1)
    }

    pub fn hit_value<T>(&mut self, id: u32, value: T) {
        if id == 53342 {
            println!("a");
        }

        let value_bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };
        if id == 53342 {
            println!("b");
        }

        self.hit_slice(id, value_bytes);
        
        if id == 53342 {
            println!("c");
        }
    }

    pub fn hit_slice(&mut self, id: u32, value: &[u8]) {
        let align = std::mem::align_of::<TraceVectorEntry>();
        let base_ptr = unsafe { self.data_mut_ptr().add(self.header().offset) };
        let alignment_offset = base_ptr.align_offset(align) as usize;

        if alignment_offset == usize::MAX {
            // 无法对齐，需要处理这种情况
            println!("Cannot align pointer");
            panic!("Cannot align pointer");
        }

        let entry_size = size_of::<TraceVectorEntry>() + value.len();
        let total_size = alignment_offset + entry_size;

        if self.header().offset + total_size > self.memory_capacity() {
            println!("TraceVector is full of capacity after pushing this item: ");
            println!("current offset: {}", self.header().offset);
            println!("entry total size: {}", total_size);
            println!("memory capacity: {}", self.memory_capacity());
            panic!("TraceVector is full of capacity: {}", self.capacity());
        }

        // 写入 entry
        unsafe {
            let aligned_ptr = (base_ptr as *mut u8).add(alignment_offset);
            let entry_ptr = aligned_ptr as *mut TraceVectorEntry;

            assert_eq!(
                entry_ptr as usize % align,
                0,
                "TraceVectorEntry pointer is not properly aligned"
            );

            (*entry_ptr).id = id;
            (*entry_ptr).length = value.len() as u32;

            // 复制 value 数据
            let value_offset = std::mem::size_of::<TraceVectorEntry>();
            let value_ptr = (entry_ptr as *mut u8).add(value_offset);
            std::ptr::copy_nonoverlapping(value.as_ptr(), value_ptr, value.len());
        }

        // 更新 header，包括对齐偏移量
        self.header_mut().len += 1;
        self.header_mut().offset += total_size;

        // println!(
        //     "Hit: id: {}, length: {}, offset: {}",
        //     id,
        //     value.len(),
        //     self.header().offset
        // );
    }

    fn entry(&self, idx: usize) -> Option<&TraceVectorEntry> {
        unimplemented!();
        if idx >= self.len() {
            return None;
        }

        let mut offset = 0;
        for i in 0..=idx {
            let entry = unsafe {
                let ptr = self.data_ptr().add(offset) as *const TraceVectorEntry;
                &*ptr
            };
            if i == idx {
                return Some(entry);
            }
            offset += size_of::<TraceVectorEntry>() + entry.length as usize;
        }
        None
    }

    pub fn get_value(&self, idx: usize) -> Option<&[u8]> {
        unimplemented!();
        let entry = self.entry(idx)?;
        Some(unsafe { slice::from_raw_parts(entry.value.as_ptr(), entry.length as usize) })
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.header().data.as_ptr() as *const u8
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        self.header_mut().data.as_mut_ptr() as *mut u8
    }

    pub fn clear(&mut self) {
        self.header_mut().len = 0;
        self.header_mut().offset = 0;
    }

    pub fn items(&self) -> Vec<TraceItem> {
        self.iter()
            .map(|e| TraceItem {
                id: e.id,
                value: e.raw_value().to_vec(),
            })
            .collect()
    }

    pub fn entries_slice(&self) -> Vec<&TraceVectorEntry> {
        self.iter().collect()
    }

    pub fn len_mut(&mut self) -> &mut usize {
        &mut self.header_mut().len
    }

    pub fn len(&self) -> usize {
        self.header().len
    }

    pub fn capacity(&self) -> usize {
        self.header().capacity
    }

    pub fn unlink(&self) {
        unsafe {
            let cname = CString::new(self._shared_memory.path()).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }
    }

    pub fn iter(&self) -> TraceVectorIterator {
        TraceVectorIterator {
            trace_vector: self,
            current_idx: 0,
            current_offset: 0,
        }
    }
}

pub struct TraceVectorIterator<'a> {
    trace_vector: &'a TraceVector,
    current_idx: usize,
    current_offset: usize,
}

impl<'a> Iterator for TraceVectorIterator<'a> {
    type Item = &'a TraceVectorEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.trace_vector.len() {
            return None;
        }

        let align = std::mem::align_of::<TraceVectorEntry>();

        // 计算对齐偏移
        let base_ptr = unsafe { self.trace_vector.data_ptr().add(self.current_offset) };
        let alignment_offset = base_ptr.align_offset(align) as usize;

        if alignment_offset == usize::MAX {
            panic!("Cannot align pointer in iterator");
        }

        // 应用对齐偏移
        self.current_offset += alignment_offset;

        let entry = unsafe {
            let ptr =
                self.trace_vector.data_ptr().add(self.current_offset) as *const TraceVectorEntry;
            &*ptr
        };

        // 更新索引和偏移量，为下一次迭代做准备
        self.current_idx += 1;
        self.current_offset += size_of::<TraceVectorEntry>() + entry.length as usize;

        Some(entry)
    }
}

// 为了方便使用，实现 IntoIterator trait
impl<'a> IntoIterator for &'a TraceVector {
    type Item = &'a TraceVectorEntry;
    type IntoIter = TraceVectorIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
#[repr(C)]
struct TraceVectorHeader {
    capacity: usize,
    len: usize,
    offset: usize,
    data: [TraceVectorEntry; 0],
}

impl TraceVectorHeader {
    pub fn new(create: bool, total_size: usize, memory: &mut MmapShMem) -> Result<*mut Self> {
        assert!(memory.size() >= size_of::<TraceVectorEntry>());

        unsafe {
            let header: &mut TraceVectorHeader = &mut *memory
                .as_mut_ptr_of::<TraceVectorHeader>()
                .expect("Failed to get header pointer");

            if create {
                header.capacity = total_size - size_of::<Self>();
                header.len = 0;
                header.offset = 0;
            }

            return Ok(header as *mut TraceVectorHeader);
        };
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(C)]
pub struct TraceVectorEntry {
    /// Some value that is used to map the `TraceEntry` back to another object
    /// after execution (e.g., the VMA).
    pub id: u32,
    pub length: u32,
    pub value: [u8; 0],
}

impl Display for TraceVectorEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TraceVectorEntry {{ id: {}, length: {}, value: {:?} }}",
            self.id,
            self.length,
            self.raw_value()
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Default)]
pub struct Trace {
    items: Vec<TraceItem>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Default)]

pub struct TraceItem {
    pub id: u32,
    pub value: Vec<u8>,
}

// 获取 value 的低 bits 位
// 例如：value = 0x09, bits = 2, 则返回 0x01
pub fn int_low_bits(value: u64, bits: u16) -> u64 {
    assert!(bits <= 64);
    if bits == 64 {
        return value;
    }
    value & ((1 << bits) - 1)
}

impl TraceVectorEntry {
    pub fn raw_value(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.value.as_ptr(), self.length as usize) }
    }
}

pub const TRACE_HIT_CNT_THRESHOLD: usize = 1000;

impl Trace {
    pub fn from_items(items: &[TraceItem]) -> Self {
        Self {
            items: items.to_vec(),
        }
    }

    pub fn items(&self) -> &Vec<TraceItem> {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[derive(Clone)]
pub struct TraceSet {
    header: *mut TraceSetHeader,
    _shared_memory: MmapShMem,
}

impl std::fmt::Debug for TraceSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TraceSet {{ capacity: {}, len: {} }}",
            self.header().capacity,
            self.header().len
        )
    }
}

#[repr(C)]
pub struct TraceSetHeader {
    // 使用 u64 数组作为位图，每个 u64 可以表示64个数字
    capacity: usize,
    len: usize,
    bitmap: [u64; 0], // 柔性数组
}

impl TraceSetHeader {
    pub fn new(create: bool, len: usize, memory: &mut MmapShMem) -> Result<*mut Self> {
        assert!(memory.size() >= Self::required_size(len));

        unsafe {
            let header: &mut TraceSetHeader = &mut *memory
                .as_mut_ptr_of::<TraceSetHeader>()
                .expect("Failed to get header pointer");

            if create {
                header.capacity = len;
                header.len = 0;
            }

            return Ok(header as *mut TraceSetHeader);
        };
    }

    pub fn required_size(max_elements: usize) -> usize {
        // 计算需要多少个u64来存储
        let bitmap_len = (max_elements + 63) / 64;
        size_of::<usize>() * 2 + bitmap_len * size_of::<u64>()
    }
}

impl TraceSet {
    pub fn new_with_env(max_value: usize, tag: &str) -> Result<Self> {
        let size = TraceSetHeader::required_size(max_value);
        let label = format!("set_{}", tag);
        let mut shared_memory = MmapShMem::new_shmem(size, &label)?;

        let env_key = format!("{}_{}", ENV_FT_SET_SHM, tag.to_uppercase());
        shared_memory.write_to_env(&env_key)?;

        let header = TraceSetHeader::new(true, max_value, &mut shared_memory)?;

        let set = Self {
            header,
            _shared_memory: shared_memory,
        };

        Ok(set)
    }

    pub fn from_env(tag: &str) -> Result<Self> {
        let env_key = format!("{}_{}", ENV_FT_SET_SHM, tag.to_uppercase());
        let mut shared_memory = match MmapShMem::shmem_from_env(&env_key) {
            Ok(shm) => shm,
            Err(_) => {
                let err = io::Error::last_os_error();
                log::error!("Last io os error: {}", err);
                return Err(anyhow!(err).context(format!("When getting set shm: {}", env_key)));
            }
        };

        // 从共享内存大小计算容量
        let max_elements = (shared_memory.size() - size_of::<usize>() * 2) * 64 / size_of::<u64>();

        let header = TraceSetHeader::new(false, max_elements, &mut shared_memory)?;

        let set = Self {
            header,
            _shared_memory: shared_memory,
        };

        Ok(set)
    }

    pub fn len(&self) -> usize {
        self.header().len
    }

    pub fn capacity(&self) -> usize {
        self.header().capacity
    }

    pub fn header(&self) -> &TraceSetHeader {
        unsafe { &*self.header }
    }

    pub fn header_mut(&mut self) -> &mut TraceSetHeader {
        unsafe { &mut *self.header }
    }

    pub fn bitmap_ptr(&self) -> *mut u64 {
        self.header().bitmap.as_ptr() as *mut u64
    }

    pub fn add(&mut self, value: u32) {
        let idx = (value / 64) as usize;
        let bit = value % 64;
        unsafe {
            let bitmap_ptr = self.bitmap_ptr().add(idx);
            if (*bitmap_ptr & (1 << bit)) == 0 {
                *bitmap_ptr |= 1 << bit;
                self.header_mut().len += 1;
            }
        }
    }

    pub fn contains(&self, value: u32) -> bool {
        let idx = (value / 64) as usize;
        let bit = value % 64;
        unsafe { (*self.bitmap_ptr().add(idx) & (1 << bit)) != 0 }
    }

    pub fn remove(&mut self, value: u32) {
        let idx = (value / 64) as usize;
        let bit = value % 64;
        unsafe {
            *self.bitmap_ptr().add(idx) &= !(1 << bit);
        }
    }

    pub fn clear(&mut self) {
        let bitmap_len = (self.header().capacity + 63) / 64;
        unsafe {
            std::ptr::write_bytes(self.bitmap_ptr(), 0, bitmap_len);
        }
    }

    pub fn unlink(&self) {
        unsafe {
            let cname = CString::new(self._shared_memory.path()).unwrap();
            libc::shm_unlink(cname.as_ptr());
        }
    }
}

unsafe impl Send for TraceSet {}

unsafe impl Sync for TraceSet {}

pub struct SyncTraceSet(pub UnsafeCell<TraceSet>);

unsafe impl Sync for SyncTraceSet {}

impl Serialize for TraceSet {
    fn serialize<S>(&self, _serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // TODO:
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for TraceSet {
    fn deserialize<D>(_deserializer: D) -> std::prelude::v1::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // TODO:
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::tracing::TraceSet;

    #[test]
    fn trace_set() {
        let mut set = TraceSet::new_with_env(u32::MAX as usize, "test").unwrap();
        let mut rng = rand::thread_rng();
        let nums: Vec<u32> = (0..900).map(|_| rng.gen::<u32>()).collect();

        for &num in nums.iter() {
            set.add(num);
        }

        for &num in nums.iter() {
            assert!(set.contains(num));
        }

        set.unlink();

        println!("set: {:?}", set);
        println!("all nums are in set");
    }
}
