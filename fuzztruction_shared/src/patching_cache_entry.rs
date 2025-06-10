use llvm_stackmap::LocationType;
use memoffset::offset_of;
use serde::{Deserialize, Serialize};

use crate::{
    dwarf::{self, DwarfReg},
    patching_cache::PatchingCacheEntryFlags,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatchingOperation {
    pub op: PatchingOperator,
    pub operand: u64,
    pub next_idx: Option<usize>,
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

#[repr(C)]
#[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PatchingCacheEntryMetadata {
    /// A unique ID used to map mutation entries onto PatchPoint instances.
    /// We need this field, since the `vma` might differ between multiple
    /// fuzzer instances.
    id: PatchPointID,

    vma: u64,
    flags: u8,

    pub loc_type: llvm_stackmap::LocationType,
    pub loc_size: u16,
    pub dwarf_regnum: dwarf::DwarfReg,
    pub offset_or_constant: i32,

    pub target_value_size_bits: u32,

    pub ctx: Option<usize>,
    pub op_idx: Option<usize>,
}

fn flags_to_str(flags: u8) -> String {
    let mut s = vec![];
    if flags & PatchingCacheEntryFlags::Tracing as u8 > 0 {
        s.push("Tracing");
    }
    if flags & PatchingCacheEntryFlags::TracingWithVal as u8 > 0 {
        s.push("TracingWithVal");
    }
    if flags & PatchingCacheEntryFlags::Patching as u8 > 0 {
        s.push("Patching");
    }
    if flags & PatchingCacheEntryFlags::Jumping as u8 > 0 {
        s.push("Jumping");
    }
    s.join(" | ")
}

impl std::fmt::Debug for PatchingCacheEntryMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationCacheEntryMetadata")
            .field("id", &self.id)
            .field("vma", &format!("0x{:x}", self.vma))
            .field("flags", &flags_to_str(self.flags))
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
        flags: u8,
        loc_type: llvm_stackmap::LocationType,
        loc_size: u16,
        dwarf_regnum: dwarf::DwarfReg,
        offset_or_constant: i32,
        target_value_size_bits: u32,
    ) -> PatchingCacheEntry {
        let metadata = PatchingCacheEntryMetadata {
            id,
            vma,
            flags,
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

    pub fn reset_flags(&mut self) -> &mut Self {
        self.metadata.flags = 0;
        self
    }

    pub fn enable_tracing_with_val(&mut self) -> &mut Self {
        self.set_flag(PatchingCacheEntryFlags::TracingWithVal)
    }

    pub fn disable_tracing_with_val(&mut self) -> &mut Self {
        self.unset_flag(PatchingCacheEntryFlags::TracingWithVal)
    }

    pub fn enable_tracing(&mut self) -> &mut Self {
        self.set_flag(PatchingCacheEntryFlags::Tracing)
    }

    pub fn disable_tracing(&mut self) -> &mut Self {
        self.unset_flag(PatchingCacheEntryFlags::Tracing)
    }

    pub fn enable_mutation(&mut self) -> &mut Self {
        self.unset_flag(PatchingCacheEntryFlags::Patching)
    }

    pub fn disable_mutation(&mut self) -> &mut Self {
        self.set_flag(PatchingCacheEntryFlags::Patching)
    }

    pub fn set_flag(&mut self, flag: PatchingCacheEntryFlags) -> &mut Self {
        self.metadata.flags |= flag as u8;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.flags == 0
    }

    pub fn flags(&self) -> u8 {
        self.metadata.flags
    }

    pub fn set_flags(&mut self, val: u8) {
        self.metadata.flags = val;
    }

    pub fn unset_flag(&mut self, flag: PatchingCacheEntryFlags) -> &mut Self {
        self.metadata.flags &= !(flag as u8);
        self
    }

    pub fn is_flag_set(&self, flag: PatchingCacheEntryFlags) -> bool {
        (self.metadata.flags & flag as u8) > 0
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

    pub fn is_nop(&self) -> bool {
        self.op_head_idx.is_none()
    }
}

impl From<&PatchPoint> for PatchingCacheEntry {
    fn from(pp: &PatchPoint) -> Self {
        let loc = pp.location().as_ref().unwrap();
        PatchingCacheEntry::new(
            pp.id(),
            pp.vma(),
            0,
            loc.loc_type,
            loc.loc_size,
            DwarfReg::try_from(loc.dwarf_regnum).unwrap(),
            loc.offset_or_constant,
            pp.target_value_size_in_bit(),
        )
    }
}
