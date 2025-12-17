// Copyright © 2024 Pingu Fuzzer. All rights reserved.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 扁平化的 PatchPoint 共享内存缓存
//!
//! 内存布局：
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │ Header (16 bytes)                                │
//! ├──────────────────────────────────────────────────┤
//! │ PatchPoint Array（固定大小，可直接索引）         │
//! │  [0] PatchPointFlat                              │
//! │  [1] PatchPointFlat                              │
//! │  ...                                             │
//! ├──────────────────────────────────────────────────┤
//! │ Variable Data Arena（可变数据区）                │
//! │  - 函数名（null-terminated strings）             │
//! │  - LiveOuts 数组                                 │
//! │  ...                                             │
//! └──────────────────────────────────────────────────┘
//! ```

use crate::constants::ENV_PP_SHM;
use crate::dwarf::{self, DwarfReg};
use crate::patchpoint::PatchPoint;
use crate::types::PatchPointID;
use crate::var::VarType;
use anyhow::{Context, Result};
use derive_more::Display;
use std::collections::HashMap;
use std::ffi::CString;
use std::time::Instant;
use std::{env, ptr};

/// 共享内存头部
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CacheHeader {
    /// 魔数
    magic: u32,
    /// PatchPoint 数量
    patchpoint_count: u32,
    /// PatchPoint 数组起始偏移（相对 mmap 基地址）
    patchpoint_array_offset: u32,
    /// 可变数据区起始偏移（相对 mmap 基地址）
    vardata_offset: u32,
}

impl CacheHeader {
    const MAGIC: u32 = 0xDEAD0003;
    const SIZE: usize = 16;
}

/// 扁平化的 LiveOut
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveOutFlat {
    pub dwarf_regnum: u16,
    pub size: u8,
    _padding: u8,
}

impl From<&llvm_stackmap::LiveOut> for LiveOutFlat {
    fn from(lo: &llvm_stackmap::LiveOut) -> Self {
        Self {
            dwarf_regnum: lo.dwarf_regnum(),
            size: lo.size(),
            _padding: 0,
        }
    }
}

/// VarType 的类型枚举
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum VarTypeKind {
    None = 0,
    Int = 1,
    Array = 2,
}

/// VarType 的扁平化信息（包含 agent 需要的所有信息）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarTypeFlat {
    /// 是否可追踪（val_tracable）
    pub is_tracable: bool,
    /// Whether it is a register variable.
    pub is_reg: bool,
    /// If it is a variable in memory (not a register), whether it is a pointer.
    pub is_ptr: bool,
    /// dereference 后的类型种类
    pub deref_kind: VarTypeKind,
    /// dereference 后的字节长度
    pub deref_bytes: u16,
}

impl From<Option<&VarType>> for VarTypeFlat {
    fn from(vt: Option<&VarType>) -> Self {
        if vt.is_none() {
            return Self {
                is_tracable: false,
                is_ptr: false,
                is_reg: false,
                deref_kind: VarTypeKind::None,
                deref_bytes: 0,
            };
        }

        let vt = vt.unwrap();
        let is_tracable = vt.val_tracable();

        // 提取 dereference 信息
        let (deref_kind, num_bytes, is_ptr, is_reg) = if let Some(deref) = vt.dereference() {
            match deref {
                VarType::Int { .. }
                | VarType::Float { .. }
                | VarType::Struct { .. }
                | VarType::Enum { .. }
                | VarType::Union { .. }
                | VarType::Bitfield { .. } => {
                    (VarTypeKind::Int, deref.num_bytes() as u16, false, false)
                }
                VarType::Array { .. } => {
                    (VarTypeKind::Array, deref.num_bytes() as u16, false, false)
                }
                VarType::Pointer { pointee: None, .. } => {
                    // 64 bytes for opaque pointer
                    // Same as i64
                    (VarTypeKind::Int, 8, false, false)
                }
                VarType::Pointer {
                    pointee: Some(pointee),
                    ..
                } => match pointee.as_ref() {
                    VarType::Int { .. }
                    | VarType::Float { .. }
                    | VarType::Struct { .. }
                    | VarType::Enum { .. }
                    | VarType::Union { .. }
                    | VarType::Bitfield { .. } => {
                        (VarTypeKind::Int, pointee.num_bytes() as u16, true, false)
                    }
                    VarType::Array { .. } => {
                        (VarTypeKind::Array, pointee.num_bytes() as u16, true, false)
                    }
                    _ => (VarTypeKind::None, 0, true, false),
                },
                _ => (VarTypeKind::None, 0, false, false),
            }
        } else {
            // deference() returns none, means that it is a register variable.
            match vt {
                VarType::Int { .. }
                | VarType::Float { .. }
                | VarType::Pointer { pointee: None, .. } => {
                    (VarTypeKind::Int, vt.num_bytes() as u16, false, true)
                }
                _ => {
                    log::error!("Wrong var type in LLVM IR register: {:#?}", vt);
                    panic!("Wrong var type in LLVM IR register: {:#?}", vt);
                }
            }
        };

        Self {
            is_tracable,
            is_ptr,
            is_reg,
            deref_kind,
            deref_bytes: num_bytes,
        }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapRangeFlat {
    pub start: u64,
    pub end: u64
}

/// 扁平化的 PatchPoint（固定大小，使用偏移量引用可变数据）
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct PatchPointFlat {
    // ===== 基本标识 =====
    pub id: PatchPointID, // 8 bytes
    pub ir_id: u32,       // 4 bytes（与 ir_id 相同）
    pub func_idx: u32,    // 4 bytes

    // VMA
    pub vma: u64,               // 8 bytes
    pub range: MapRangeFlat,    // 16 bytes

    // ===== 函数名信息（使用偏移量）=====
    /// 函数名在可变数据区的偏移量（相对于 vardata_offset）
    pub function_name_offset: u32, // 4 bytes
    /// 函数名长度（字节数）
    pub function_name_len: u16, // 2 bytes

    // PatchPoint information
    pub loc_type: llvm_stackmap::LocationType,
    pub loc_size: u16,
    pub dwarf_regnum: dwarf::DwarfReg,
    pub offset_or_constant: i32,

    // ===== LiveOuts 信息（使用偏移量）=====
    /// LiveOuts 在可变数据区的偏移量（相对于 vardata_offset）
    pub liveout_offset: u32, // 4 bytes
    /// LiveOut 数量
    pub liveout_count: u8, // 1 byte

    // ===== VarType 信息 =====
    pub var_type: VarTypeFlat, // 12 bytes

    // ===== 对齐填充 =====
    _padding: [u8; 3], // 3 bytes，总计对齐到 48 bytes
}

impl PatchPointFlat {
    const SIZE: usize = std::mem::size_of::<Self>();

    /// 获取函数名
    pub fn function_name<'a>(&self, vardata_base: *const u8) -> &'a str {
        if self.function_name_len == 0 {
            return "";
        }

        unsafe {
            let name_ptr = vardata_base.add(self.function_name_offset as usize);
            let name_slice = std::slice::from_raw_parts(name_ptr, self.function_name_len as usize);
            std::str::from_utf8_unchecked(name_slice)
        }
    }

    /// 获取 LiveOuts
    pub fn live_outs<'a>(&self, vardata_base: *const u8) -> &'a [LiveOutFlat] {
        if self.liveout_count == 0 {
            return &[];
        }

        unsafe {
            let liveout_ptr = vardata_base.add(self.liveout_offset as usize) as *const LiveOutFlat;
            // println!("vardata_base: {:#x}", vardata_base as usize);
            // println!("self.liveout_offset: {}", self.liveout_offset);
            // println!("self.liveout_count: {}", self.liveout_count);
            // 打印内存字节用于调试
            // let byte_ptr = liveout_ptr as *const u8;
            // print!("LiveOut memory bytes: ");
            // for i in 0..(self.liveout_count as usize * std::mem::size_of::<LiveOutFlat>()) {
            //     print!("{:02x} ", *byte_ptr.add(i));
            // }
            // println!();
            std::slice::from_raw_parts(liveout_ptr, self.liveout_count as usize)
        }
    }
}

// 这些结构体都是 POD 类型（Plain Old Data）
// 使用 #[repr(C)] 确保内存布局

// 编译时检查
const _: () = {
    // 确保大小合理（不超过 64 bytes）
    const PP_SIZE: usize = std::mem::size_of::<PatchPointFlat>();
    assert!(PP_SIZE <= 80);
    assert!(PP_SIZE % 8 == 0); // 8 字节对齐

    assert!(std::mem::size_of::<LiveOutFlat>() == 4);
    assert!(std::mem::size_of::<VarTypeFlat>() == 6);
    assert!(std::mem::size_of::<MapRangeFlat>() == 16);
    assert!(std::mem::size_of::<CacheHeader>() == 16);
};

/// PatchPoint 缓存（Agent 端使用）
pub struct PatchPointCache {
    /// mmap 基地址
    mmap_ptr: *const u8,
    /// mmap 总大小
    mmap_len: usize,
    shm_name: String,
    /// PatchPoint 数组基地址
    patchpoint_array: *const PatchPointFlat,
    /// 可变数据区基地址
    vardata_base: *const u8,
    /// PatchPoint 数量
    count: usize,
    /// ID 到索引的映射
    id_to_idx: HashMap<PatchPointID, usize>,
}

unsafe impl Send for PatchPointCache {}
unsafe impl Sync for PatchPointCache {}

impl PatchPointCache {
    /// 从共享内存打开
    pub fn open(shm_name: &str) -> Result<Self> {
        // 1. 打开共享内存
        let shm_name_c = CString::new(shm_name)?;
        let shm_fd = unsafe { libc::shm_open(shm_name_c.as_ptr(), libc::O_RDONLY, 0) };
        if shm_fd < 0 {
            return Err(anyhow::anyhow!(
                "Failed to open shared memory: {}",
                shm_name
            ));
        }

        // 2. 获取大小
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::fstat(shm_fd, &mut stat) };
        if ret < 0 {
            unsafe {
                libc::close(shm_fd);
            }
            return Err(anyhow::anyhow!("Failed to fstat"));
        }
        let total_size = stat.st_size as usize;

        // 3. mmap（MAP_SHARED，关键！）
        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                shm_fd,
                0,
            )
        };
        unsafe {
            libc::close(shm_fd);
        }

        if mmap_ptr == libc::MAP_FAILED {
            return Err(anyhow::anyhow!("Failed to mmap"));
        }

        // 4. 读取并校验 Header
        let header = unsafe { ptr::read(mmap_ptr as *const CacheHeader) };
        if header.magic != CacheHeader::MAGIC {
            unsafe {
                libc::munmap(mmap_ptr, total_size);
            }
            return Err(anyhow::anyhow!("Invalid magic: {:#x}", header.magic));
        }

        // 5. 计算各区域地址
        let patchpoint_array = unsafe {
            (mmap_ptr as *const u8).add(header.patchpoint_array_offset as usize)
                as *const PatchPointFlat
        };
        let vardata_base = unsafe { (mmap_ptr as *const u8).add(header.vardata_offset as usize) };

        // 6. 构建 ID 到索引的映射
        let mut id_to_idx = HashMap::new();
        for i in 0..header.patchpoint_count as usize {
            let pp = unsafe { &*patchpoint_array.add(i) };
            id_to_idx.insert(pp.id, i);
        }

        log::info!(
            "PatchPointCache opened: {} patchpoints, {} bytes, array at +{}, vardata at +{}",
            header.patchpoint_count,
            total_size,
            header.patchpoint_array_offset,
            header.vardata_offset
        );

        Ok(Self {
            mmap_ptr: mmap_ptr as *const u8,
            mmap_len: total_size,
            shm_name: shm_name.to_string(),
            patchpoint_array,
            vardata_base,
            count: header.patchpoint_count as usize,
            id_to_idx,
        })
    }

    /// 从环境变量打开
    pub fn open_shm_from_env() -> Result<Self> {
        let shm_name = std::env::var(ENV_PP_SHM).context(format!("{ENV_PP_SHM} not set"))?;
        Self::open(&shm_name)
    }

    /// 根据 ID 获取 PatchPoint
    pub fn get(&self, id: &PatchPointID) -> Option<PatchPointRef> {
        let idx = self.id_to_idx.get(id)?;
        Some(PatchPointRef {
            pp: unsafe { &*self.patchpoint_array.add(*idx) },
            vardata_base: self.vardata_base,
        })
    }

    /// 根据索引获取 PatchPoint
    pub fn get_by_index(&self, idx: usize) -> Option<PatchPointRef> {
        if idx >= self.count {
            return None;
        }
        Some(PatchPointRef {
            pp: unsafe { &*self.patchpoint_array.add(idx) },
            vardata_base: self.vardata_base,
        })
    }

    /// 获取数量
    pub fn len(&self) -> usize {
        self.count
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// 判断是否包含某个 ID
    pub fn contains(&self, id: &PatchPointID) -> bool {
        self.id_to_idx.contains_key(id)
    }

    /// 迭代所有 PatchPoint
    pub fn iter(&self) -> PatchPointCacheIter {
        PatchPointCacheIter {
            patchpoint_array: self.patchpoint_array,
            vardata_base: self.vardata_base,
            current_idx: 0,
            count: self.count,
        }
    }

    pub fn unlink(&self) {
        if env::var("PINGU_GDB").is_ok() {
            log::info!("PINGU_GDB is set, skipping unlinking of patchpint cache");
        } else {
            let name = CString::new(self.shm_name.clone()).unwrap();
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
        }
    }
}

impl Drop for PatchPointCache {
    fn drop(&mut self) {
        if !self.mmap_ptr.is_null() {
            unsafe {
                libc::munmap(self.mmap_ptr as *mut libc::c_void, self.mmap_len);
            }
        }
    }
}

/// PatchPoint 的引用（包含可变数据区的访问）
#[derive(Debug, Clone, Copy)]
pub struct PatchPointRef<'a> {
    pp: &'a PatchPointFlat,
    vardata_base: *const u8,
}

impl<'a> PatchPointRef<'a> {
    /// 获取 ID
    pub fn id(&self) -> &PatchPointID {
        &self.pp.id
    }

    /// 获取 LLVM ID（也是 IR ID）
    pub fn ir_id(&self) -> u32 {
        self.pp.ir_id
    }

    /// 获取函数索引
    pub fn func_idx(&self) -> u32 {
        self.pp.func_idx
    }

    /// 获取函数名
    pub fn function_name(&self) -> &str {
        self.pp.function_name(self.vardata_base)
    }

    pub fn vma(&self) -> u64 {
        self.pp.vma
    }

    pub fn range(&self) -> &MapRangeFlat {
        &self.pp.range
    }

    /// 获取 LiveOuts
    pub fn live_outs(&self) -> &[LiveOutFlat] {
        self.pp.live_outs(self.vardata_base)
    }

    /// 判断是否可追踪
    pub fn is_var_tracable(&self) -> bool {
        self.pp.var_type.is_tracable
    }

    pub fn is_deref_ptr(&self) -> bool {
        self.pp.var_type.is_ptr
    }

    /// 获取 dereference 后的类型种类
    pub fn deref_kind(&self) -> VarTypeKind {
        self.pp.var_type.deref_kind
    }

    /// 获取 dereference 后的字节长度
    pub fn deref_size(&self) -> u16 {
        self.pp.var_type.deref_bytes
    }

    pub fn var_type(&self) -> &VarTypeFlat {
        &self.pp.var_type
    }

    /// 获取底层的 PatchPointFlat
    pub fn as_flat(&self) -> &PatchPointFlat {
        self.pp
    }

    pub fn loc_type(&self) -> llvm_stackmap::LocationType {
        self.pp.loc_type
    }

    pub fn loc_size(&self) -> u16 {
        self.pp.loc_size
    }

    pub fn dwarf_regnum(&self) -> dwarf::DwarfReg {
        self.pp.dwarf_regnum
    }

    pub fn offset_or_constant(&self) -> i32 {
        self.pp.offset_or_constant
    }
}

/// 迭代器
pub struct PatchPointCacheIter {
    patchpoint_array: *const PatchPointFlat,
    vardata_base: *const u8,
    current_idx: usize,
    count: usize,
}

impl Iterator for PatchPointCacheIter {
    type Item = PatchPointRef<'static>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.count {
            return None;
        }

        let pp = unsafe { &*self.patchpoint_array.add(self.current_idx) };
        self.current_idx += 1;

        Some(PatchPointRef {
            pp,
            vardata_base: self.vardata_base,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PatchPointCacheIter {}

/// 构建器（Fuzzer 端使用）
pub struct PatchPointCacheBuilder {
    /// PatchPoint 数组
    patchpoints: Vec<PatchPointFlat>,
    /// 可变数据缓冲区
    vardata: Vec<u8>,
}

impl PatchPointCacheBuilder {
    pub fn new() -> Self {
        Self {
            patchpoints: Vec::new(),
            vardata: Vec::new(),
        }
    }

    pub fn from(patchpoints: &Vec<&PatchPoint>) -> Result<Self> {
        let start = Instant::now();

        let mut builder = Self::new();
        for pp in patchpoints {
            builder.add(pp);
        }

        let end = Instant::now();
        log::info!(
            "Time taken to build PatchPointCache: {:?}",
            end.duration_since(start)
        );

        Ok(builder)
    }

    /// 添加一个 PatchPoint
    pub fn add(&mut self, pp: &PatchPoint) {
        // 1. 处理函数名
        let function_name_str = pp.ir().function.as_str();

        let function_name_offset = self.vardata.len() as u32;
        let function_name_len = function_name_str.len().min(u16::MAX as usize) as u16;
        self.vardata.extend_from_slice(function_name_str.as_bytes());

        // 2. 处理 LiveOuts
        let live_outs_vec = pp.live_outs();
        let liveout_align = std::mem::align_of::<LiveOutFlat>();
        let padding = (liveout_align - (self.vardata.len() % liveout_align)) % liveout_align;
        if padding != 0 {
            self.vardata.resize(self.vardata.len() + padding, 0);
        }

        let liveout_offset = self.vardata.len() as u32;
        let liveout_count = live_outs_vec.len().min(255) as u8;

        for lo in live_outs_vec.iter().take(liveout_count as usize) {
            let lo_flat = LiveOutFlat::from(lo);
            // 手动转换为 bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &lo_flat as *const LiveOutFlat as *const u8,
                    std::mem::size_of::<LiveOutFlat>(),
                )
            };
            self.vardata.extend_from_slice(bytes);
        }

        let var_type_flat = VarTypeFlat::from(pp.var_type());

        // 4. 创建 PatchPointFlat
        let loc = pp.location().as_ref().unwrap();
        let pp_flat = PatchPointFlat {
            id: pp.id().clone(),
            ir_id: pp.llvm_id(),
            func_idx: pp.func_idx(),
            vma: pp.vma(),
            range: MapRangeFlat {
                start: pp.mapping().start() as u64,
                end: pp.mapping().end() as u64,
            },
            function_name_offset,
            function_name_len,
            loc_type: loc.loc_type,
            loc_size: loc.loc_size,
            dwarf_regnum: DwarfReg::try_from(loc.dwarf_regnum).unwrap(),
            offset_or_constant: loc.offset_or_constant,
            liveout_offset,
            liveout_count,
            var_type: var_type_flat,
            _padding: [0; 3],
        };

        self.patchpoints.push(pp_flat);
    }

    /// 获取 PatchPoint 数量
    pub fn len(&self) -> usize {
        self.patchpoints.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.patchpoints.is_empty()
    }

    pub fn unlink(shm_name: &str) -> Result<()> {
        let shm_name_c = CString::new(shm_name)?;
        unsafe {
            libc::shm_unlink(shm_name_c.as_ptr());
        }
        Ok(())
    }

    /// 写入共享内存
    pub fn write_to_shm(&self, shm_name: &str) -> Result<()> {
        // 1. 计算布局
        let header_size = CacheHeader::SIZE;
        let patchpoint_array_size = self.patchpoints.len() * PatchPointFlat::SIZE;
        let vardata_size = self.vardata.len();
        let total_size = header_size + patchpoint_array_size + vardata_size;

        log::debug!(
            "PatchPointCache layout: header={}, pp_array={} ({} × {}), vardata={}, total={}",
            header_size,
            patchpoint_array_size,
            self.patchpoints.len(),
            PatchPointFlat::SIZE,
            vardata_size,
            total_size
        );

        // 2. 创建共享内存
        let shm_name_c = CString::new(shm_name)?;
        let shm_fd = unsafe {
            libc::shm_open(
                shm_name_c.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            )
        };
        if shm_fd < 0 {
            return Err(anyhow::anyhow!(
                "Failed to create shared memory: {}",
                shm_name
            ));
        }

        let ret = unsafe { libc::ftruncate(shm_fd, total_size as i64) };
        if ret < 0 {
            unsafe {
                libc::close(shm_fd);
            }
            return Err(anyhow::anyhow!("Failed to ftruncate"));
        }

        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            )
        };

        unsafe {
            libc::close(shm_fd);
        }

        if mmap_ptr == libc::MAP_FAILED {
            return Err(anyhow::anyhow!("Failed to mmap"));
        }

        // 3. 写入 Header
        let header = CacheHeader {
            magic: CacheHeader::MAGIC,
            patchpoint_count: self.patchpoints.len() as u32,
            patchpoint_array_offset: header_size as u32,
            vardata_offset: (header_size + patchpoint_array_size) as u32,
        };

        unsafe {
            ptr::write(mmap_ptr as *mut CacheHeader, header);
        }

        // 4. 写入 PatchPoint 数组
        if !self.patchpoints.is_empty() {
            unsafe {
                ptr::copy_nonoverlapping(
                    self.patchpoints.as_ptr(),
                    (mmap_ptr as *mut u8).add(header_size) as *mut PatchPointFlat,
                    self.patchpoints.len(),
                );
            }
        }

        // 5. 写入可变数据区
        if !self.vardata.is_empty() {
            unsafe {
                ptr::copy_nonoverlapping(
                    self.vardata.as_ptr(),
                    (mmap_ptr as *mut u8).add(header_size + patchpoint_array_size),
                    vardata_size,
                );
            }
        }

        // 6. 设置为只读
        unsafe {
            libc::mprotect(mmap_ptr, total_size, libc::PROT_READ);
            libc::munmap(mmap_ptr, total_size);
        }

        log::info!(
            "PatchPointCache created: {} patchpoints, {} bytes vardata, {} bytes total",
            self.patchpoints.len(),
            vardata_size,
            total_size
        );

        Ok(())
    }
}

impl Default for PatchPointCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<CacheHeader>(), 16);
        assert_eq!(std::mem::size_of::<LiveOutFlat>(), 4);
        // assert_eq!(std::mem::size_of::<VarTypeFlat>(), 8);
        assert!(std::mem::size_of::<PatchPointFlat>() <= 64);
        assert!(std::mem::size_of::<PatchPointFlat>() % 8 == 0);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(std::mem::align_of::<PatchPointFlat>(), 8);
    }
}
