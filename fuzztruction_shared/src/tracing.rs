use anyhow::{anyhow, Result};
use core::slice;
use serde::{Deserialize, Serialize};
use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet},
    ffi::CString,
    io,
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
    pub fn frequent_set(&self, threshold: usize) -> Vec<u32> {
        let mut trace_cnt = self.hit_counts();
        trace_cnt.retain(|_, cnt| *cnt as usize > threshold);
        trace_cnt.keys().cloned().collect::<Vec<_>>()
    }

    pub fn memory_ratio(&self) -> f64 {
        (self.len() * size_of::<TraceEntry>() + size_of::<TraceVectorHeader>()) as f64
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
        let total_size = size_of::<TraceVectorHeader>() + len * size_of::<TraceEntry>();
        let label = format!("trace_{}", tag);
        let mut shared_memory = MmapShMem::new_shmem(total_size, &label)?;
        let header = TraceVectorHeader::new(true, len, &mut shared_memory)?;

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
        let len = (shared_memory.size() - size_of::<TraceVectorHeader>()) / size_of::<TraceEntry>();
        let header = TraceVectorHeader::new(false, len, &mut shared_memory)?;

        Ok(TraceVector {
            header,
            _shared_memory: shared_memory,
        })
    }

    pub fn first(&self) -> Option<&TraceEntry> {
        self.entry(0)
    }

    pub fn last(&self) -> Option<&TraceEntry> {
        if self.len() == 0 {
            return None;
        }
        self.entry(self.len() - 1)
    }

    pub fn hit(&mut self, id: u32, value: &[u8]) {
        let entry_size = size_of::<TraceEntry>() + value.len();
        if self.header().offset + entry_size > self.memory_capacity() {
            println!("TraceVector is full of capacity: {}", self.capacity());
            panic!("TraceVector is full of capacity: {}", self.capacity());
        }

        // 写入 entry
        unsafe {
            let entry_ptr = self.data_mut_ptr().add(self.header().offset) as *mut TraceEntry;
            (*entry_ptr).id = id;
            (*entry_ptr).length = value.len() as u32;

            // 复制 value 数据
            let value_ptr = entry_ptr.add(1) as *mut u8;
            std::ptr::copy_nonoverlapping(value.as_ptr(), value_ptr, value.len());
        }

        // 更新 header
        self.header_mut().len += 1;
        self.header_mut().offset += entry_size;
    }

    fn entry(&self, idx: usize) -> Option<&TraceEntry> {
        if idx >= self.len() {
            return None;
        }

        let mut offset = 0;
        for i in 0..=idx {
            let entry = unsafe {
                let ptr = self.data_ptr().add(offset) as *const TraceEntry;
                &*ptr
            };
            if i == idx {
                return Some(entry);
            }
            offset += size_of::<TraceEntry>() + entry.length as usize;
        }
        None
    }

    pub fn get_value(&self, idx: usize) -> Option<&[u8]> {
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

    pub fn entries_slice(&self) -> Vec<&TraceEntry> {
        let mut entries = Vec::new();
        let mut offset = 0;

        for _ in 0..self.len() {
            let entry = unsafe {
                let ptr = self.data_ptr().add(offset) as *const TraceEntry;
                &*ptr
            };
            entries.push(entry);
            offset += size_of::<TraceEntry>() + entry.length as usize;
        }

        entries
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
}

#[derive(Debug, Clone)]
#[repr(C)]
struct TraceVectorHeader {
    capacity: usize,
    len: usize,
    offset: usize,
    data: [TraceEntry; 0],
}

impl TraceVectorHeader {
    pub fn new(create: bool, len: usize, memory: &mut MmapShMem) -> Result<*mut Self> {
        assert!(memory.size() >= size_of::<TraceEntry>());

        unsafe {
            let header: &mut TraceVectorHeader = memory.as_object_mut();

            if create {
                header.capacity = len;
                header.len = 0;
                header.offset = 0;
            }

            return Ok(header as *mut TraceVectorHeader);
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[repr(C)]
pub struct TraceEntry {
    /// Some value that is used to map the `TraceEntry` back to another object
    /// after execution (e.g., the VMA).
    pub id: u32,
    pub length: u32,
    pub value: [u8; 0],
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

impl TraceEntry {
    pub fn raw_value(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.value.as_ptr(), self.length as usize) }
    }
}

pub const TRACE_HIT_CNT_THRESHOLD: usize = 1000;

pub fn remove_frequent_trace_entries<'a>(
    entries: &[&'a TraceEntry],
    threshold: usize,
    too_freq_pp_ids: Option<&mut HashSet<PatchPointID>>,
) -> Vec<&'a TraceEntry> {
    let len = entries.len();
    let mut trace_cnt: HashMap<u32, usize> = HashMap::new();
    for entry in entries {
        *trace_cnt.entry(entry.id).or_insert(0) += 1;
    }

    trace_cnt.retain(|_, cnt| *cnt <= threshold);
    let tracable_ids = trace_cnt.keys().cloned().collect::<HashSet<_>>();

    let entries: Vec<&TraceEntry> = entries
        .into_iter()
        .filter(|entry| tracable_ids.contains(&entry.id))
        .cloned()
        .collect();

    log::trace!(
        "remove_frequent_trace_entries: {}({:2.2}%)",
        len - entries.len(),
        (len - entries.len()) as f64 / len as f64 * 100.0
    );

    entries
}

impl Trace {
    pub fn from_entries(entries: &[&TraceEntry]) -> Self {
        // TODO:
        // remove frequent trace entries
        let entries = remove_frequent_trace_entries(entries, TRACE_HIT_CNT_THRESHOLD, None);

        let items: Vec<TraceItem> = entries
            .iter()
            .map(|entry| TraceItem {
                id: entry.id,
                value: entry.raw_value().to_vec(),
            })
            .collect();

        Self { items }
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
            let header: &mut TraceSetHeader = memory.as_object_mut();

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
