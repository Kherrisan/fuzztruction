use std::{cmp::min, mem, ptr};

use crate::{
    patching_cache_entry::{PatchingCacheEntry, PatchingOperation},
    types::PatchPointID,
};

pub const MAX_PATCHING_CACHE_ENTRIES: usize = 400000;
pub const MAX_PATCHING_OPERATIONS: usize = 1000000;
const PENDING_DELETIONS_LIMIT: usize = 500;

/// Entry 描述符：记录 entry 在数据区的位置
#[derive(Debug, Clone, Copy)]
struct EntryDescriptor {
    start_offset: usize,
    op_head_idx: Option<usize>,
}

/// Operation 描述符：记录 operation 在数据区的位置
/// 注意：next_idx 存储在 PatchingOperation 数据本身中，不需要在描述符中重复
#[derive(Debug, Clone, Copy)]
struct OpDescriptor {
    start_offset: usize,
}

#[repr(C, align(8))]
pub struct PatchingCacheContent {
    /// 整体内存大小
    total_size: usize,

    /// Entry 相关
    entry_current_data_size: usize,
    entry_next_free_slot: usize,
    entry_pending_deletions: usize,
    entry_valid_count: usize, // ✅ 有效 entry 数量，用于优化遍历

    /// Operation 相关
    op_current_data_size: usize,
    op_next_free_slot: usize,
    op_pending_deletions: usize,
    op_valid_count: usize, // ✅ 有效 operation 数量，用于优化遍历

    /// Entry 数据区在 data 中的起始偏移
    entry_data_offset: usize,
    /// Operation 数据区在 data 中的起始偏移
    op_data_offset: usize,

    /// 是否需要使所有指针失效
    invalidate: bool,

    /// Entry 描述符表（固定大小，快速索引）
    entry_descriptor_tbl: [Option<EntryDescriptor>; MAX_PATCHING_CACHE_ENTRIES],

    /// Operation 描述符表（固定大小，快速索引）
    op_descriptor_tbl: [Option<OpDescriptor>; MAX_PATCHING_OPERATIONS],

    /// 动态数据区的起始位置（柔性数组成员）
    data: [u8; 0],
}

impl std::fmt::Debug for PatchingCacheContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PatchingCacheContent {{")?;
        write!(f, "total_size: {}, ", self.total_size)?;
        write!(f, "entry_count: {}, ", self.entry_count())?;
        write!(f, "op_count: {}, ", self.op_count())?;
        write!(f, "entry_data_used: {}, ", self.entry_current_data_size)?;
        write!(f, "op_data_used: {}", self.op_current_data_size)?;
        write!(f, "}}")
    }
}

impl PatchingCacheContent {
    /// 估算所需内存大小
    pub fn estimate_memory_occupied(entry_size: usize, op_size: usize) -> usize {
        let base_size = mem::size_of::<PatchingCacheContent>();

        // Entry 数据区预估
        let entry_data_size = entry_size * mem::size_of::<PatchingCacheEntry>();

        // Operation 数据区预估
        let op_data_size = op_size * mem::size_of::<PatchingOperation>();

        // 对齐填充
        let alignment =
            mem::align_of::<PatchingOperation>().max(mem::align_of::<PatchingCacheEntry>());
        let padding = alignment * 2;

        base_size + entry_data_size + op_data_size + padding
    }

    /// 初始化内容
    pub fn init(&mut self, entry_size: usize, op_size: usize, create: bool) {
        let size = Self::estimate_memory_occupied(entry_size, op_size);
        assert!(
            size >= mem::size_of_val(self),
            "The backing memory must be at least {} bytes large.",
            mem::size_of_val(self)
        );

        if create {
            self.entry_current_data_size = 0;
            self.entry_next_free_slot = 0;
            self.entry_pending_deletions = 0;
            self.entry_valid_count = 0;

            self.op_current_data_size = 0;
            self.op_next_free_slot = 0;
            self.op_pending_deletions = 0;
            self.op_valid_count = 0;

            self.total_size = size;
            self.invalidate = false;

            self.entry_descriptor_tbl.fill(None);
            self.op_descriptor_tbl.fill(None);

            // 计算数据区偏移（确保对齐）
            self.entry_data_offset = 0;
            let op_align = mem::align_of::<PatchingOperation>();
            self.op_data_offset = (entry_size * mem::size_of::<PatchingCacheEntry>() + op_align
                - 1)
                & !(op_align - 1);
        }
    }

    pub fn total_size(&self) -> usize {
        self.total_size
    }

    pub fn entry_table_size(&self) -> usize {
        MAX_PATCHING_CACHE_ENTRIES
    }

    pub fn op_table_size(&self) -> usize {
        MAX_PATCHING_OPERATIONS
    }

    pub fn entry_data_offset(&self) -> usize {
        self.entry_data_offset
    }

    pub fn op_data_offset(&self) -> usize {
        self.op_data_offset
    }

    /// 返回数据区的总空间（不包括 PatchingCacheContent 结构体本身）
    pub fn total_space(&self) -> usize {
        self.total_size.saturating_sub(mem::size_of_val(self))
    }

    /// 返回剩余可用空间
    pub fn space_left(&self) -> usize {
        self.total_space()
            .saturating_sub(self.entry_current_data_size)
            .saturating_sub(self.op_current_data_size)
    }

    pub fn clear(&mut self) {
        self.entry_current_data_size = 0;
        self.entry_next_free_slot = 0;
        self.entry_pending_deletions = 0;
        self.entry_valid_count = 0;

        self.op_current_data_size = 0;
        self.op_next_free_slot = 0;
        self.op_pending_deletions = 0;
        self.op_valid_count = 0;

        self.entry_descriptor_tbl.fill(None);
        self.op_descriptor_tbl.fill(None);

        self.invalidate = true;
    }

    pub fn is_invalidated(&self) -> bool {
        self.invalidate
    }

    pub fn clear_invalidated(&mut self) {
        self.invalidate = false;
    }

    /// 获取 entry 数据区起始指针
    pub unsafe fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// 获取 entry 数据区起始指针
    unsafe fn entry_data_ptr(&self) -> *const u8 {
        unsafe { self.data_ptr().offset(self.entry_data_offset as isize) }
    }

    /// 获取 operation 数据区起始指针
    unsafe fn op_data_ptr(&self) -> *const u8 {
        unsafe { self.data_ptr().offset(self.op_data_offset as isize) }
    }

    /// 通过偏移获取 entry 引用
    pub unsafe fn entry_ref_by_offset(&self, offset: usize) -> &PatchingCacheEntry {
        unsafe {
            let ptr = self.entry_data_ptr().offset(offset as isize);
            &*(ptr as *const PatchingCacheEntry)
        }
    }

    /// 通过偏移获取 entry 可变引用
    pub unsafe fn entry_mut_by_offset(&self, offset: usize) -> &mut PatchingCacheEntry {
        unsafe {
            let ptr = self.entry_data_ptr().offset(offset as isize);
            &mut *(ptr as *mut PatchingCacheEntry)
        }
    }

    /// 通过索引获取 entry 引用
    pub fn entry_ref(&self, idx: usize) -> &PatchingCacheEntry {
        let desc = self.entry_descriptor_tbl[idx]
            .as_ref()
            .expect("Entry descriptor not found");
        unsafe { self.entry_ref_by_offset(desc.start_offset) }
    }

    /// 通过索引获取 entry 可变引用
    pub fn entry_mut(&self, idx: usize) -> &mut PatchingCacheEntry {
        let desc = self.entry_descriptor_tbl[idx]
            .as_ref()
            .expect("Entry descriptor not found");
        unsafe { self.entry_mut_by_offset(desc.start_offset) }
    }

    /// 通过偏移获取 operation 引用
    unsafe fn op_ref_by_offset(&self, offset: usize) -> &PatchingOperation {
        unsafe {
            let ptr = self.op_data_ptr().offset(offset as isize);
            &*(ptr as *const PatchingOperation)
        }
    }

    /// ✅ 关键API：通过 descriptor 索引获取 operation 引用
    /// 供 pingu-agent stub 调用，保持 API 兼容性
    pub fn op_ref(&self, descriptor_idx: usize) -> &PatchingOperation {
        let op_desc = self.op_descriptor_tbl[descriptor_idx]
            .as_ref()
            .expect("Operation descriptor not found");
        unsafe { self.op_ref_by_offset(op_desc.start_offset) }
    }

    /// 通过 descriptor 索引获取 operation 可变引用
    pub fn op_mut(&self, descriptor_idx: usize) -> &mut PatchingOperation {
        let op_desc = self.op_descriptor_tbl[descriptor_idx]
            .as_ref()
            .expect("Operation descriptor not found");
        unsafe {
            let ptr = self.op_data_ptr().offset(op_desc.start_offset as isize);
            &mut *(ptr as *mut PatchingOperation)
        }
    }

    /// 获取所有 entries
    pub fn entries(&self) -> Vec<&PatchingCacheEntry> {
        let mut result = Vec::with_capacity(self.entry_valid_count);
        let mut found = 0;

        // ✅ 使用计数器提前退出，避免遍历所有槽位
        for desc in self.entry_descriptor_tbl.iter() {
            if found >= self.entry_valid_count {
                break;
            }
            if let Some(entry_desc) = desc {
                result.push(unsafe { self.entry_ref_by_offset(entry_desc.start_offset) });
                found += 1;
            }
        }
        result
    }

    /// 获取所有 entries（可变）
    pub fn entries_mut(&mut self) -> Vec<&mut PatchingCacheEntry> {
        let count = self.entry_valid_count;
        let mut result = Vec::with_capacity(count);
        let mut found = 0;

        // ✅ 使用计数器提前退出，避免遍历所有槽位
        for desc in self.entry_descriptor_tbl.iter() {
            if found >= count {
                break;
            }
            if let Some(entry_desc) = desc {
                result.push(unsafe { self.entry_mut_by_offset(entry_desc.start_offset) });
                found += 1;
            }
        }
        result
    }

    /// 添加一个 entry
    pub fn push(&mut self, entry: PatchingCacheEntry) -> anyhow::Result<usize> {
        if self.entry_next_free_slot >= MAX_PATCHING_CACHE_ENTRIES {
            anyhow::bail!("Entry descriptor table full, maybe consolidate() first");
        }

        let entry_size = mem::size_of::<PatchingCacheEntry>();
        if entry_size > self.space_left() {
            log::error!("Not enough space left in the cache to push entry");
            anyhow::bail!("Not enough space for entry");
        }

        let descriptor_idx = self.entry_next_free_slot;
        let descriptor = Some(EntryDescriptor {
            start_offset: self.entry_current_data_size,
            op_head_idx: None,
        });

        // 填充描述符表
        if let ref mut slot @ None = self.entry_descriptor_tbl[descriptor_idx] {
            *slot = descriptor;
            self.entry_next_free_slot = min(
                self.entry_next_free_slot + 1,
                MAX_PATCHING_CACHE_ENTRIES - 1,
            );
            self.entry_valid_count += 1; // ✅ 增加有效 entry 计数
        } else {
            log::error!("entry_descriptor_tbl[{}] is not None", descriptor_idx);
            anyhow::bail!("Descriptor slot not available");
        }

        // 拷贝 entry 数据到数据区
        unsafe {
            let dst = self
                .entry_data_ptr()
                .offset(self.entry_current_data_size as isize);
            ptr::copy_nonoverlapping(&entry as *const _ as *const u8, dst as *mut u8, entry_size);
        }

        self.entry_current_data_size += entry_size;
        self.invalidate = true;

        // ✅ 当空间利用率过高时，触发 consolidate
        if self.space_left() < self.total_space() / 10 {
            log::debug!("Space utilization high, triggering consolidate");
            unsafe {
                let _ = self.consolidate();
            }
        }

        Ok(descriptor_idx)
    }

    /// 为指定 entry 添加一个 operation
    pub fn push_op(&mut self, entry_idx: usize, mut op: PatchingOperation) -> anyhow::Result<()> {
        let mut entry_desc = self.entry_descriptor_tbl[entry_idx]
            .ok_or_else(|| anyhow::anyhow!("Entry not found"))?;

        if self.op_next_free_slot >= MAX_PATCHING_OPERATIONS {
            anyhow::bail!("Operation descriptor table full, maybe consolidate() first");
        }

        let op_size = mem::size_of::<PatchingOperation>();
        if op_size > self.space_left() {
            anyhow::bail!("Not enough space for operation");
        }

        let op_idx = self.op_next_free_slot;
        let op_descriptor = Some(OpDescriptor {
            start_offset: self.op_current_data_size,
        });

        self.op_descriptor_tbl[op_idx] = op_descriptor;
        self.op_next_free_slot = min(self.op_next_free_slot + 1, MAX_PATCHING_OPERATIONS - 1);
        self.op_valid_count += 1; // ✅ 增加有效 operation 计数

        // ✅ 确保 PatchingOperation 数据的 next_idx 初始化为 None
        op.next_idx = None;

        // 拷贝 operation 数据
        unsafe {
            let dst = self
                .op_data_ptr()
                .offset(self.op_current_data_size as isize);
            ptr::copy_nonoverlapping(&op as *const _ as *const u8, dst as *mut u8, op_size);
        }

        self.op_current_data_size += op_size;

        // 链接到 entry 的操作链表
        if entry_desc.op_head_idx.is_none() {
            // 第一个操作
            entry_desc.op_head_idx = Some(op_idx);
            self.entry_descriptor_tbl[entry_idx] = Some(entry_desc);
        } else {
            // 追加到链表末尾，找到最后一个 operation
            let mut current_idx = entry_desc.op_head_idx.unwrap();
            loop {
                // ✅ 通过读取 PatchingOperation 数据中的 next_idx 来遍历链表
                let current_op = self.op_ref(current_idx);
                if current_op.next_idx.is_none() {
                    // ✅ 只需更新实际 PatchingOperation 数据中的 next_idx（供 stub 读取）
                    let prev_op = self.op_mut(current_idx);
                    prev_op.next_idx = Some(op_idx);
                    break;
                }
                current_idx = current_op.next_idx.unwrap();
            }
        }

        self.invalidate = true;
        Ok(())
    }

    /// 批量添加 operations
    pub fn push_op_batch(
        &mut self,
        entry_idx: usize,
        ops: &[PatchingOperation],
    ) -> anyhow::Result<()> {
        for op in ops {
            self.push_op(entry_idx, *op)?;
        }
        Ok(())
    }

    /// 清除 entry 的所有 operations
    pub fn clear_entry_ops(&mut self, entry_idx: usize) -> anyhow::Result<()> {
        let mut entry_desc = self.entry_descriptor_tbl[entry_idx]
            .ok_or_else(|| anyhow::anyhow!("Entry not found"))?;

        if let Some(op_head_idx) = entry_desc.op_head_idx {
            self.remove_op_chain(op_head_idx);
            entry_desc.op_head_idx = None;
            self.entry_descriptor_tbl[entry_idx] = Some(entry_desc);
        }

        Ok(())
    }

    /// 获取 entry 的所有 operations
    pub fn ops(&self, entry_idx: usize) -> Vec<PatchingOperation> {
        if let Some(desc) = self.entry_descriptor_tbl[entry_idx].as_ref() {
            self.get_entry_ops(desc)
        } else {
            Vec::new()
        }
    }

    fn get_entry_ops(&self, desc: &EntryDescriptor) -> Vec<PatchingOperation> {
        let mut ops = Vec::new();
        let mut current_idx = desc.op_head_idx;

        while let Some(idx) = current_idx {
            if let Some(_op_desc) = self.op_descriptor_tbl[idx] {
                // ✅ 通过 op_ref 读取 operation，其中包含 next_idx
                let op = self.op_ref(idx);
                ops.push(*op);
                current_idx = op.next_idx;
            } else {
                break;
            }
        }

        ops
    }

    /// 根据索引删除 entry（O(1) 复杂度，无需查找）
    pub fn remove_by_idx(&mut self, entry_idx: usize) -> anyhow::Result<()> {
        let entry_desc = self.entry_descriptor_tbl[entry_idx]
            .ok_or_else(|| anyhow::anyhow!("Entry descriptor not found at index {}", entry_idx))?;

        // 删除关联的所有 operations
        if let Some(op_head_idx) = entry_desc.op_head_idx {
            self.remove_op_chain(op_head_idx);
        }

        // 标记 entry descriptor 为 None（懒删除）
        self.entry_descriptor_tbl[entry_idx] = None;
        self.entry_pending_deletions += 1;
        self.entry_valid_count = self.entry_valid_count.saturating_sub(1); // ✅ 减少有效 entry 计数
        self.invalidate = true;

        // 达到阈值时压缩
        if self.entry_pending_deletions > PENDING_DELETIONS_LIMIT {
            unsafe {
                let _ = self.consolidate()?;
            }
        }

        Ok(())
    }

    /// 根据 ID 删除 entry（需要 O(n) 线性查找）
    pub fn remove(&mut self, id: PatchPointID) -> anyhow::Result<()> {
        // 查找 entry 索引
        let entry_idx = self
            .find_entry(id)
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow::anyhow!("Entry with id {:?} not found", id))?;

        // 使用索引删除
        self.remove_by_idx(entry_idx)
    }

    /// 删除 operation 链表
    fn remove_op_chain(&mut self, head_idx: usize) {
        let mut current_idx = Some(head_idx);

        while let Some(idx) = current_idx {
            if let Some(_op_desc) = self.op_descriptor_tbl[idx] {
                // ✅ 先读取 next_idx，然后删除当前 descriptor
                let next = self.op_ref(idx).next_idx;
                self.op_descriptor_tbl[idx] = None;
                self.op_pending_deletions += 1;
                self.op_valid_count = self.op_valid_count.saturating_sub(1); // ✅ 减少有效 operation 计数
                current_idx = next;
            } else {
                break;
            }
        }

        if self.op_pending_deletions > PENDING_DELETIONS_LIMIT {
            unsafe {
                let _ = self.consolidate();
            }
        }
    }

    /// 压缩内存，移除空洞
    pub unsafe fn consolidate(&mut self) -> anyhow::Result<()> {
        self.invalidate = true;

        // 备份所有有效的 entries 和 operations
        let entries_with_ops: Vec<(PatchingCacheEntry, Vec<PatchingOperation>)> = self
            .entry_descriptor_tbl
            .iter()
            .filter_map(|desc| desc.as_ref())
            .map(|desc| {
                let entry = unsafe { self.entry_ref_by_offset(desc.start_offset).clone() };
                let ops = self.get_entry_ops(desc);
                (entry, ops)
            })
            .collect();

        // 清空
        self.clear();

        // 重新添加（紧凑排列）
        for (entry, ops) in entries_with_ops {
            let entry_idx = self.push(entry)?;
            if !ops.is_empty() {
                self.push_op_batch(entry_idx, &ops)?;
            }
        }

        Ok(())
    }

    /// 保留满足条件的 entries
    /// ✅ 闭包接收可变引用，允许在判断的同时修改 entry
    pub fn retain<F>(&mut self, mut f: F) -> usize
    where
        F: FnMut(&mut PatchingCacheEntry) -> bool,
    {
        let mut removed_count = 0;
        let mut checked = 0;
        let initial_count = self.entry_valid_count;

        // ✅ 使用计数器提前退出
        for idx in 0..MAX_PATCHING_CACHE_ENTRIES {
            if checked >= initial_count {
                break;
            }

            if let Some(desc) = self.entry_descriptor_tbl[idx] {
                let entry = unsafe { self.entry_mut_by_offset(desc.start_offset) };
                checked += 1;

                if !f(entry) {
                    // 删除这个 entry
                    if let Some(op_head_idx) = desc.op_head_idx {
                        self.remove_op_chain(op_head_idx);
                    }
                    self.entry_descriptor_tbl[idx] = None;
                    self.entry_pending_deletions += 1;
                    self.entry_valid_count = self.entry_valid_count.saturating_sub(1); // ✅ 减少有效 entry 计数
                    removed_count += 1;
                }
            }
        }

        if self.entry_pending_deletions > PENDING_DELETIONS_LIMIT {
            unsafe {
                let _ = self.consolidate();
            }
        }

        self.invalidate = true;
        removed_count
    }

    /// 获取 entry 数量（O(1) 复杂度）
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_valid_count
    }

    /// 获取 operation 数量（O(1) 复杂度）
    #[inline]
    pub fn op_count(&self) -> usize {
        self.op_valid_count
    }

    /// 查找 entry
    pub fn find_entry(&self, id: PatchPointID) -> Option<(usize, &PatchingCacheEntry)> {
        let mut found = 0;

        // ✅ 使用计数器提前退出
        for (idx, desc) in self.entry_descriptor_tbl.iter().enumerate() {
            if found >= self.entry_valid_count {
                break;
            }
            if let Some(entry_desc) = desc {
                let entry = unsafe { self.entry_ref_by_offset(entry_desc.start_offset) };
                found += 1;
                if entry.id() == id {
                    return Some((idx, entry));
                }
            }
        }
        None
    }

    /// 迭代所有有效的 entry 索引（只读）
    pub fn iter(&self) -> EntryIndexIter {
        EntryIndexIter {
            descriptor_tbl: &self.entry_descriptor_tbl,
            current: 0,
            remaining: self.entry_valid_count, // ✅ 记录剩余数量
        }
    }

    /// 迭代所有有效的 entry 索引（可变）
    pub fn iter_mut(&mut self) -> EntryIndexIterMut {
        EntryIndexIterMut {
            descriptor_tbl: &mut self.entry_descriptor_tbl,
            current: 0,
            remaining: self.entry_valid_count, // ✅ 记录剩余数量
        }
    }
}

/// Entry 索引迭代器（只读）
/// 直接遍历 descriptor table，利用计数器优化提前退出
pub struct EntryIndexIter {
    descriptor_tbl: *const [Option<EntryDescriptor>; MAX_PATCHING_CACHE_ENTRIES],
    current: usize,
    remaining: usize, // ✅ 剩余有效 entry 数量
}

impl Iterator for EntryIndexIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // ✅ 使用计数器提前退出
        if self.remaining == 0 {
            return None;
        }

        unsafe {
            while self.current < MAX_PATCHING_CACHE_ENTRIES {
                let idx = self.current;
                self.current += 1;
                if (*self.descriptor_tbl)[idx].is_some() {
                    self.remaining -= 1;
                    return Some(idx);
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for EntryIndexIter {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Entry 索引迭代器（可变）
/// 直接遍历 descriptor table，利用计数器优化提前退出
pub struct EntryIndexIterMut {
    descriptor_tbl: *mut [Option<EntryDescriptor>; MAX_PATCHING_CACHE_ENTRIES],
    current: usize,
    remaining: usize, // ✅ 剩余有效 entry 数量
}

impl Iterator for EntryIndexIterMut {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // ✅ 使用计数器提前退出
        if self.remaining == 0 {
            return None;
        }

        unsafe {
            while self.current < MAX_PATCHING_CACHE_ENTRIES {
                let idx = self.current;
                self.current += 1;
                if (*self.descriptor_tbl)[idx].is_some() {
                    self.remaining -= 1;
                    return Some(idx);
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for EntryIndexIterMut {
    fn len(&self) -> usize {
        self.remaining
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util;

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

    fn dummy_op(operand: u64) -> PatchingOperation {
        PatchingOperation::new(crate::patching_cache_entry::PatchingOperator::Add, operand)
    }

    #[test]
    fn test_push_remove() {
        // 使用 estimate_memory_occupied 计算正确的内存大小
        let size = PatchingCacheContent::estimate_memory_occupied(1000, 1000);
        let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        content.init(1000, 1000, true);

        let init_space_left = content.space_left();

        // Test if empty
        assert_eq!(content.entries().len(), 0);

        // Test push e0
        let e0 = dummy_entry(0);
        let idx0 = content.push(e0).unwrap();
        assert_eq!(content.entries().len(), 1);
        assert_eq!(idx0, 0);

        // Test push e1
        let e1 = dummy_entry(1);
        let idx1 = content.push(e1).unwrap();
        assert_eq!(content.entries().len(), 2);
        assert_eq!(idx1, 1);

        // Test remove
        content.remove(0u32.into()).unwrap();
        assert_eq!(content.entries().len(), 1);

        content.remove(1u32.into()).unwrap();
        assert_eq!(content.entries().len(), 0);

        unsafe {
            content.consolidate().unwrap();
        }
        assert_eq!(content.space_left(), init_space_left);
    }

    #[test]
    fn test_push_ops() {
        // ✅ 使用 estimate_memory_occupied 计算正确的大小
        let size = PatchingCacheContent::estimate_memory_occupied(1000, 1000);
        let mut content: Box<PatchingCacheContent> = util::alloc_box_aligned_zeroed(size);
        content.init(1000, 1000, true);

        let e0 = dummy_entry(0);
        let idx0 = content.push(e0).unwrap();

        // Push operations
        let ops = vec![dummy_op(1), dummy_op(2), dummy_op(3)];
        content.push_op_batch(idx0, &ops).unwrap();

        // Verify operations
        let retrieved_ops = content.ops(idx0);
        assert_eq!(retrieved_ops.len(), 3);
        assert_eq!(retrieved_ops[0].operand, 1);
        assert_eq!(retrieved_ops[1].operand, 2);
        assert_eq!(retrieved_ops[2].operand, 3);

        // Clear operations
        content.clear_entry_ops(idx0).unwrap();
        let retrieved_ops = content.ops(idx0);
        assert_eq!(retrieved_ops.len(), 0);
    }
}
