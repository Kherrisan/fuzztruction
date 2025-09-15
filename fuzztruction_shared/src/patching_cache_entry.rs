use llvm_stackmap::LocationType;
use memoffset::offset_of;
use num_enum::IntoPrimitive;
use serde::{Deserialize, Serialize};
use strum_macros::{EnumString};

use crate::{
    dwarf::{self, DwarfReg},
    patching_cache::{PATCHING_CACHE_ENTRY_FLAGS, PatchingCacheEntryFlags},
    patchpoint::PatchPoint,
    types::PatchPointID,
    util,
};
use std::alloc;

const MAX_OP_COUNT: usize = 1024 * 1024 * 64;

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

#[repr(C)]
#[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PatchingCacheEntryMetadata {
    /// A unique ID used to map mutation entries onto PatchPoint instances.
    /// We need this field, since the `vma` might differ between multiple
    /// fuzzer instances.
    id: PatchPointID,

    vma: u64,

    dirty: [PatchingCacheEntryDirty; 4],

    pub loc_type: llvm_stackmap::LocationType,
    pub loc_size: u16,
    pub dwarf_regnum: dwarf::DwarfReg,
    pub offset_or_constant: i32,

    pub target_value_size_bits: u32,

    pub ctx: Option<usize>,
    pub op_idx: Option<usize>,
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

impl std::fmt::Debug for PatchingCacheEntryMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchingCacheEntryMetadata")
            .field("id", &self.id)
            .field("vma", &format!("0x{:x}", self.vma))
            .field("flags", &flags_to_str(&self.dirty))
            .field("loc_type", &self.loc_type)
            .field("loc_size", &self.loc_size)
            .field("dwarf_regnum", &self.dwarf_regnum)
            .field("offset_or_constant", &self.offset_or_constant)
            .field("op_idx", &self.op_idx)
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PatchingCacheEntry {
    pub metadata: PatchingCacheEntryMetadata,
    /// The mask that is applied in chunks of size `loc_size` each time the mutated
    /// location is accessed. If `loc_size` > 0, then the mask is msk_len + loc_size bytes
    /// long, else it is msk_len bytes in size.
    pub op_head_idx: Option<usize>,
    pub op_tail_idx: Option<usize>,
}

impl std::fmt::Debug for PatchingCacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchingCacheEntry")
            .field("metadata", &self.metadata)
            .field("op_head_idx", &self.op_head_idx)
            .field("op_tail_idx", &self.op_tail_idx)
            .finish()
    }
}

impl PatchingCacheEntry {
    pub fn new(
        id: PatchPointID,
        vma: u64,
        loc_type: llvm_stackmap::LocationType,
        loc_size: u16,
        dwarf_regnum: dwarf::DwarfReg,
        offset_or_constant: i32,
        target_value_size_bits: u32,
    ) -> PatchingCacheEntry {
        let metadata = PatchingCacheEntryMetadata {
            id,
            vma,
            dirty: [PatchingCacheEntryDirty::Nop; 4],
            loc_type,
            loc_size,
            dwarf_regnum,
            offset_or_constant,
            target_value_size_bits,
            op_idx: None,
            ctx: None,
        };

        PatchingCacheEntry {
            metadata,
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

    pub fn offsetof_op_slot() -> usize {
        offset_of!(PatchingCacheEntryMetadata, op_idx)
    }

    pub fn ctx(&self) -> Option<usize> {
        self.metadata.ctx
    }

    pub fn id(&self) -> PatchPointID {
        self.metadata.id
    }

    pub fn vma(&self) -> u64 {
        self.metadata.vma
    }

    pub fn loc_type(&self) -> LocationType {
        self.metadata.loc_type
    }

    pub fn loc_size(&self) -> u16 {
        self.metadata.loc_size
    }

    pub fn dwarf_regnum(&self) -> DwarfReg {
        self.metadata.dwarf_regnum
    }

    pub fn op_idx(&self) -> Option<usize> {
        self.metadata.op_idx
    }

    pub fn reset_dirty_flags(&mut self) -> &mut Self {
        self.metadata.dirty = [PatchingCacheEntryDirty::Nop; 4];
        self
    }

    pub fn flag(&self, flag: PatchingCacheEntryFlags) -> PatchingCacheEntryDirty {
        self.metadata.dirty[flag as usize]
    }

    pub fn flag_mut(&mut self, flag: PatchingCacheEntryFlags) -> &mut PatchingCacheEntryDirty {
        &mut self.metadata.dirty[flag as usize]
    }

    pub fn set_dirty_flag(
        &mut self,
        flag: PatchingCacheEntryFlags,
        dirty: PatchingCacheEntryDirty,
    ) -> &mut Self {
        self.metadata.dirty[flag as usize] = dirty;
        self
    }

    pub fn dirty_flags(&self) -> &[PatchingCacheEntryDirty; 4] {
        &self.metadata.dirty
    }

    pub fn dirty_flags_mut(&mut self) -> &mut [PatchingCacheEntryDirty; 4] {
        &mut self.metadata.dirty
    }

    pub fn set_dirty_flags(&mut self, dirty: [PatchingCacheEntryDirty; 4]) {
        self.metadata.dirty = dirty;
    }

    pub fn is_flag_set(
        &self,
        flag: PatchingCacheEntryFlags,
        dirty: PatchingCacheEntryDirty,
    ) -> bool {
        self.metadata.dirty[flag as usize] == dirty
    }

    /// The size in bytes of the whole entry. Cloning a MutationCacheEntry requires
    /// to copy .size() bytes from a pointer of type MutationCacheEntry.
    pub fn size(&self) -> usize {
        std::mem::size_of::<PatchingCacheEntryMetadata>()
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
        self.metadata.dirty[flag as usize] = PatchingCacheEntryDirty::Dirty;
    }

    pub fn set_dirty_to_enable(&mut self) {
        PATCHING_CACHE_ENTRY_FLAGS.iter().for_each(|f| {
            if self.metadata.dirty[*f as usize] == PatchingCacheEntryDirty::Dirty {
                self.metadata.dirty[*f as usize] = PatchingCacheEntryDirty::Enable;
            }
        });
    }

    pub fn set_by(&mut self, cond: PatchingCacheEntryDirty, dirty: PatchingCacheEntryDirty) {
        PATCHING_CACHE_ENTRY_FLAGS.iter().for_each(|f| {
            if self.metadata.dirty[*f as usize] == cond {
                self.metadata.dirty[*f as usize] = dirty;
            }
        });
    }

    pub fn is_any(&self, dirty: PatchingCacheEntryDirty) -> bool {
        self.metadata.dirty.iter().any(|d| *d == dirty)
    }

    pub fn is_all(&self, dirty: PatchingCacheEntryDirty) -> bool {
        self.metadata.dirty.iter().all(|d| *d == dirty)
    }

    pub fn is_dirty(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.metadata.dirty[flag as usize] == PatchingCacheEntryDirty::Dirty
    }

    pub fn is_enable(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.metadata.dirty[flag as usize] == PatchingCacheEntryDirty::Enable
    }

    pub fn is_nop(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.metadata.dirty[flag as usize] == PatchingCacheEntryDirty::Nop
    }

    pub fn set_clear(&mut self, flag: PatchingCacheEntryFlags) {
        self.metadata.dirty[flag as usize] = PatchingCacheEntryDirty::Clear;
    }

    pub fn is_clear(&self, flag: PatchingCacheEntryFlags) -> bool {
        self.metadata.dirty[flag as usize] == PatchingCacheEntryDirty::Clear
    }
}

impl From<&PatchPoint> for PatchingCacheEntry {
    fn from(pp: &PatchPoint) -> Self {
        let loc = pp.location().as_ref().unwrap();
        PatchingCacheEntry::new(
            pp.id(),
            pp.vma(),
            loc.loc_type,
            loc.loc_size,
            DwarfReg::try_from(loc.dwarf_regnum).unwrap(),
            loc.offset_or_constant,
            pp.target_value_size_in_bit(),
        )
    }
}
