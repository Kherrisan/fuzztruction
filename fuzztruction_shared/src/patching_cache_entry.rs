use num_enum::IntoPrimitive;
use serde::{Deserialize, Serialize};
use strum_macros::EnumString;

use crate::{
    patching_cache::{PATCHING_CACHE_ENTRY_FLAGS, PatchingCacheEntryFlags},
    patchpoint::PatchPoint,
    types::PatchPointID,
    util,
};
use std::alloc;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatchingOperator {
    Nop = 0,
    Add = 1,
    Sub = 2,
    Shl = 3,
    Shr = 4,
    And = 5,
    Or = 6,
    Xor = 7,
    Not = 8,
    Set = 9,
    Clear = 10,
    Jmp = 11,
}

pub const PATCHING_OPERATORS: [PatchingOperator; 12] = [
    PatchingOperator::Nop,
    PatchingOperator::Add,
    PatchingOperator::Sub,
    PatchingOperator::Shl,
    PatchingOperator::Shr,
    PatchingOperator::And,
    PatchingOperator::Or,
    PatchingOperator::Xor,
    PatchingOperator::Not,
    PatchingOperator::Set,
    PatchingOperator::Clear,
    PatchingOperator::Jmp,
];

#[repr(C)]
#[derive(Clone, Copy, Serialize, Deserialize, Hash)]
pub struct PatchingOperation {
    pub op: PatchingOperator,
    pub operand: u64,
    pub next_idx: Option<usize>,
}

impl std::fmt::Debug for PatchingOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PatchingOperation {{ op: {:?}, operand: 0x{:x} }}",
            self.op, self.operand
        )
    }
}

impl Eq for PatchingOperation {}

impl PartialEq for PatchingOperation {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.operand == other.operand
    }
}

impl PatchingOperation {
    pub fn new(op: PatchingOperator, operand: u64) -> Self {
        Self {
            op,
            operand,
            next_idx: None,
        }
    }
}

#[repr(u8)]
#[derive(
    Clone,
    Serialize,
    Deserialize,
    Hash,
    PartialEq,
    Eq,
    strum_macros::Display,
    EnumString,
    Debug,
    IntoPrimitive,
    Copy,
)]
pub enum PatchingCacheEntryDirty {
    Nop = 0,
    Dirty = 1,
    Clear = 2,
    Enable = 3,
}

pub fn flags_to_str(flags: &[PatchingCacheEntryDirty; 4]) -> String {
    [
        PatchingCacheEntryFlags::Tracing,
        PatchingCacheEntryFlags::TracingVal,
        PatchingCacheEntryFlags::Patching,
        PatchingCacheEntryFlags::Jumping,
    ]
    .into_iter()
    .map(|f| format!("{} = {}", f.to_string(), flags[f as usize].to_string()))
    .collect::<Vec<_>>()
    .join(" | ")
}

#[repr(C)]
#[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PatchingCacheEntry {
    pub id: PatchPointID,

    pub dirty: [PatchingCacheEntryDirty; 4],

    pub ctx: Option<usize>,

    pub op_idx: Option<usize>,

    pub op_head_idx: Option<usize>,
    pub op_tail_idx: Option<usize>,
}

impl std::fmt::Debug for PatchingCacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchingCacheEntry")
            .field("id", &self.id)
            .field("flags", &flags_to_str(&self.dirty))
            .field("ctx", &self.ctx)
            .field("op_idx", &self.op_idx)
            .field("op_head_idx", &self.op_head_idx)
            .field("op_tail_idx", &self.op_tail_idx)
            .finish()
    }
}

impl PatchingCacheEntry {
    pub fn new(id: PatchPointID) -> PatchingCacheEntry {
        PatchingCacheEntry {
            id,
            dirty: [PatchingCacheEntryDirty::Nop; 4],
            ctx: None,
            op_idx: None,
            op_head_idx: None,
            op_tail_idx: None,
        }
    }

    pub fn layout() -> alloc::Layout {
        alloc::Layout::new::<PatchingCacheEntry>()
    }

    pub fn clone_into_box(self: &PatchingCacheEntry) -> Box<PatchingCacheEntry> {
        let size = self.size();
        let mut entry: Box<PatchingCacheEntry> = util::alloc_box_aligned_zeroed(size);

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.as_ptr() as *const u8,
                entry.as_mut_ptr() as *mut u8,
                size,
            );
        }

        entry
    }

    pub fn clone_with_new_msk(
        self: &PatchingCacheEntry,
        new_msk_len: u32,
    ) -> Box<PatchingCacheEntry> {
        unimplemented!()
        // assert!(
        //     new_msk_len <= MAX_MASK_LEN as u32 && new_msk_len > 0,
        //     "new_msk_len={}",
        //     new_msk_len
        // );

        // let mut new_size = std::mem::size_of_val(self) + new_msk_len as usize;

        // // Padding for read overlow support.
        // if self.loc_size() > 0 {
        //     new_size += self.loc_size() as usize;
        // }

        // // Zeroed memory
        // let mut entry: Box<PatchingCacheEntry> = util::alloc_box_aligned_zeroed(new_size);

        // // Copy the metadata of the old entry into the new one.

        // let mut bytes_to_copy = self.size_wo_overflow_padding();
        // if self.msk_len() > new_msk_len {
        //     // If we are shrinking the msk, do not copy all data from the old entry.
        //     bytes_to_copy -= (self.msk_len() - new_msk_len) as usize;
        // }

        // unsafe {
        //     std::ptr::copy_nonoverlapping(
        //         self.as_ptr() as *const u8,
        //         entry.as_mut_ptr() as *mut u8,
        //         bytes_to_copy,
        //     );
        // }

        // // Adapt metadata to changed values.
        // entry.metadata.msk_len = new_msk_len;
        // entry
    }

    pub fn ctx(&self) -> Option<usize> {
        self.ctx
    }

    pub fn id(&self) -> PatchPointID {
        self.id
    }

    pub fn op_idx(&self) -> Option<usize> {
        self.op_idx
    }

    pub fn reset_dirty_flags(&mut self) -> &mut Self {
        self.dirty = [PatchingCacheEntryDirty::Nop; 4];
        self
    }

    pub fn flag(&self, flag: PatchingCacheEntryFlags) -> PatchingCacheEntryDirty {
        self.dirty[flag as usize]
    }

    pub fn flag_mut(&mut self, flag: PatchingCacheEntryFlags) -> &mut PatchingCacheEntryDirty {
        &mut self.dirty[flag as usize]
    }

    pub fn set_dirty_flag(
        &mut self,
        flag: PatchingCacheEntryFlags,
        dirty: PatchingCacheEntryDirty,
    ) -> &mut Self {
        self.dirty[flag as usize] = dirty;
        self
    }

    pub fn dirty_flags(&self) -> &[PatchingCacheEntryDirty; 4] {
        &self.dirty
    }

    pub fn dirty_flags_mut(&mut self) -> &mut [PatchingCacheEntryDirty; 4] {
        &mut self.dirty
    }

    pub fn set_dirty_flags(&mut self, dirty: [PatchingCacheEntryDirty; 4]) {
        self.dirty = dirty;
    }

    pub fn is_flag_set(
        &self,
        flag: PatchingCacheEntryFlags,
        dirty: PatchingCacheEntryDirty,
    ) -> bool {
        self.dirty[flag as usize] == dirty
    }

    /// The size in bytes of the whole entry. Cloning a MutationCacheEntry requires
    /// to copy .size() bytes from a pointer of type MutationCacheEntry.
    pub fn size(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    pub fn as_ptr(&self) -> *const PatchingCacheEntry {
        self as *const PatchingCacheEntry
    }

    pub fn as_mut_ptr(&mut self) -> *mut PatchingCacheEntry {
        self as *mut PatchingCacheEntry
    }

    #[allow(invalid_reference_casting)]
    pub unsafe fn alias_mut(&self) -> &mut PatchingCacheEntry {
        let ptr = self as *const PatchingCacheEntry as *mut PatchingCacheEntry;
        &mut *ptr
    }

    pub fn set_dirty(&mut self, flag: PatchingCacheEntryFlags) {
        self.dirty[flag as usize] = PatchingCacheEntryDirty::Dirty;
    }

    pub fn set_dirty_to_enable(&mut self) {
        PATCHING_CACHE_ENTRY_FLAGS.iter().for_each(|f| {
            if self.dirty[*f as usize] == PatchingCacheEntryDirty::Dirty {
                self.dirty[*f as usize] = PatchingCacheEntryDirty::Enable;
            }
        });
    }

    pub fn set_by(&mut self, cond: PatchingCacheEntryDirty, dirty: PatchingCacheEntryDirty) {
        PATCHING_CACHE_ENTRY_FLAGS.iter().for_each(|f| {
            if self.dirty[*f as usize] == cond {
                self.dirty[*f as usize] = dirty;
            }
        });
    }

    pub fn is_any(&self, dirty: PatchingCacheEntryDirty) -> bool {
        self.dirty.iter().any(|d| *d == dirty)
    }

    pub fn is_all(&self, dirty: PatchingCacheEntryDirty) -> bool {
        self.dirty.iter().all(|d| *d == dirty)
    }

    pub fn is_dirty(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.dirty[flag as usize] == PatchingCacheEntryDirty::Dirty
    }

    pub fn is_enable(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.dirty[flag as usize] == PatchingCacheEntryDirty::Enable
    }

    pub fn is_nop(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.dirty[flag as usize] == PatchingCacheEntryDirty::Nop
    }

    pub fn set_clear(&mut self, flag: PatchingCacheEntryFlags) {
        self.dirty[flag as usize] = PatchingCacheEntryDirty::Clear;
    }

    pub fn is_clear(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.dirty[flag as usize] == PatchingCacheEntryDirty::Clear
    }
}

impl From<&PatchPoint> for PatchingCacheEntry {
    fn from(pp: &PatchPoint) -> Self {
        PatchingCacheEntry::new(pp.id())
    }
}
