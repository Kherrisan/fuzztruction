use anyhow::{anyhow, Context, Result};
use core::slice;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, io, mem::size_of, num::NonZeroU64};

use crate::shared_memory::{MmapShMem, MmapShMemProvider};

pub const ENV_FT_TRACE_SHM: &str = "TRACE";
pub const DEFAULT_TRACE_MAP_LEN: usize = 0x1000000;

#[derive(Debug, Clone)]
pub struct TraceMap {
    // TODO:
    // struct field and impl should be consistent with StdMapObserver
    header: *mut TraceMapHeader,
    _shared_memory: MmapShMem,
    allocated_slots: HashSet<u64>,
    total_hits: u64,
}

unsafe impl Send for TraceMap {}

impl Serialize for TraceMap {
    fn serialize<S>(&self, _serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unimplemented!()
        // TODO:
        // 随便写的，要不然编译不通过
        // MmapShMem is not serialized?
    }
}

impl<'de> Deserialize<'de> for TraceMap {
    // TODO:
    // 随便写的，要不然编译不通过
    fn deserialize<D>(_deserializer: D) -> std::prelude::v1::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        unimplemented!()
    }
}

impl TraceMap {
    fn header(&self) -> &TraceMapHeader {
        unsafe { &*self.header }
    }

    fn header_mut(&mut self) -> &mut TraceMapHeader {
        unsafe { &mut *self.header }
    }

    pub fn new(len: usize) -> Result<TraceMap> {
        let mut shm_provider = MmapShMemProvider::new()?;
        let total_size = size_of::<TraceMapHeader>() + len * size_of::<TraceEntry<u64>>();
        let mut shared_memory = shm_provider.new_shmem(total_size, "trace")?;
        let header = TraceMapHeader::new(true, len, &mut shared_memory)?;

        shared_memory.write_to_env(ENV_FT_TRACE_SHM)?;

        Ok(TraceMap {
            header,
            _shared_memory: shared_memory,
            allocated_slots: HashSet::new(),
            total_hits: 0,
        })
    }

    pub fn finalize(&mut self) {
        if self.allocated_slots.is_empty() {
            return;
        }
        *self.len_mut() = self.allocated_slots.len();

        let mut sorted_ids: Vec<_> = self.allocated_slots.clone().into_iter().collect();
        sorted_ids.sort();
        for (idx, id) in sorted_ids.iter().enumerate() {
            let entry = self
                .entry_mut(idx)
                .expect(format!("TraceEntry with idx {idx} not found in the TraceMap").as_str());
            entry.value = *id;
        }

        self.clear_hits();
        self.allocated_slots.clear();
        self.allocated_slots.shrink_to(0);
    }

    pub fn from_env() -> Result<TraceMap> {
        let mut shared_memory = match MmapShMemProvider::shmem_from_env(ENV_FT_TRACE_SHM) {
            Ok(shm) => shm,
            Err(_) => {
                let err = io::Error::last_os_error();
                log::error!("Last io os error: {}", err);
                return Err(anyhow!(err));
            }
        };
        let len =
            (shared_memory.size() - size_of::<TraceMapHeader>()) / size_of::<TraceEntry<u64>>();
        let header = TraceMapHeader::new(false, len, &mut shared_memory)?;

        Ok(TraceMap {
            header,
            _shared_memory: shared_memory,
            allocated_slots: HashSet::new(),
            total_hits: 0,
        })
    }

    pub fn alloc_slot(&mut self, id: u64) {
        self.allocated_slots.insert(id);
    }

    pub fn hit(&mut self, id: u64) {
        let idx = self
            .entries_slice()
            .binary_search_by(|e| e.value.cmp(&id))
            .expect("Trying to report hit for unallocated element");

        self.total_hits += 1;
        let order = NonZeroU64::new(self.total_hits);
        let entry = self
            .entry_mut(idx)
            .context(format!("When hitting trace_point with id {id}, idx: {idx}"))
            .expect("TraceEntry not found in the map");
        if entry.order.is_none() {
            entry.order = order;
        }
        entry.hits += 1;
    }

    fn entry(&self, idx: usize) -> Option<&TraceEntry<u64>> {
        if idx >= self.len() {
            return None;
        }
        let entry = unsafe {
            let ptr = self
                .data_ptr()
                .offset((idx * size_of::<TraceEntry<u64>>()) as isize);
            &*(ptr as *const TraceEntry<u64>)
        };
        Some(entry)
    }

    fn entry_mut(&mut self, idx: usize) -> Option<&mut TraceEntry<u64>> {
        if idx >= self.len() {
            return None;
        }
        let entry = unsafe {
            let ptr = self
                .data_mut_ptr()
                .offset((idx * size_of::<TraceEntry<u64>>()) as isize);
            &mut *(ptr as *mut TraceEntry<u64>)
        };
        Some(entry)
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.header().data.as_ptr() as *const u8
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        self.header_mut().data.as_mut_ptr() as *mut u8
    }

    pub fn clear(&mut self) {
        self.header_mut().len = 0;
        self.allocated_slots.clear();
        self.total_hits = 0;
    }

    pub fn clear_hits(&mut self) {
        for i in 0..self.len() {
            let e = self.entry_mut(i).unwrap();
            e.hits = 0;
            e.order = None;
        }
    }

    pub fn entries_slice(&self) -> &[TraceEntry<u64>] {
        unsafe { slice::from_raw_parts(self.data_ptr() as *const TraceEntry<u64>, self.len()) }
    }

    pub fn entries_mut(&mut self) -> &mut [TraceEntry<u64>] {
        unsafe {
            slice::from_raw_parts_mut(self.data_mut_ptr() as *mut TraceEntry<u64>, self.len())
        }
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
}

#[derive(Debug, Clone)]
pub struct TraceMapHeader {
    capacity: usize,
    len: usize,
    data: [TraceEntry<u64>; 0],
}

impl TraceMapHeader {
    pub fn new(create: bool, len: usize, memory: &mut MmapShMem) -> Result<*mut Self> {
        assert!(memory.size() >= size_of::<TraceEntry<u64>>());

        unsafe {
            let header: &mut TraceMapHeader = memory.as_object_mut();

            if create {
                header.capacity = len;
                header.len = 0;
            }

            return Ok(header as *mut TraceMapHeader);
        };
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TraceEntry<T> {
    /// Some value that is used to map the `TraceEntry` back to another object
    /// after execution (e.g., the VMA).
    pub value: T,
    /// Number of times this entry was hit.
    pub hits: u64,
    /// A value that can be used to order `TraceEntry`s according to their
    /// time of discovery. Odering entries ascending according to their `order`
    /// id allows to determine whether a entry was covered before or after anotherone.
    pub order: Option<NonZeroU64>,
}
