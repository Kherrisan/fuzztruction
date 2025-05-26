use llvm_stackmap::LocationType;
use memoffset::offset_of;

use crate::{
    dwarf::{self, DwarfReg},
    patching_cache::PatchingCacheEntryFlags,
    types::PatchPointID,
    util,
};
use std::alloc;

const MAX_MASK_LEN: usize = 1024 * 1024 * 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatchingOperator {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Mod = 4,
    Shl = 5,
    Shr = 6,
    And = 7,
    Or = 8,
    Xor = 9,
    Not = 10,
    Neg = 11,
    Abs = 12,
    Set = 13,
    Clear = 14,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchingOperation {
    pub op: PatchingOperator,
    pub operand: u128
}

#[repr(C)]
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

    pub read_pos: u32,
    /// The length of the mask stored at MutationCacheEntry.msk. If `loc_size` is > 0,
    /// the mask contains `loc_size` additional bytes that can be used in case the
    /// mutation stub read ptr overflows and reads more then msk_len bytes (see agent.rs).
    msk_len: u32,
}

impl std::fmt::Debug for PatchingCacheEntryMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationCacheEntryMetadata")
            .field("id", &self.id)
            .field("vma", &format!("0x{:x}", self.vma))
            .field("flags", &self.flags)
            .field("loc_type", &self.loc_type)
            .field("loc_size", &self.loc_size)
            .field("dwarf_regnum", &self.dwarf_regnum)
            .field("offset_or_constant", &self.offset_or_constant)
            .field("read_pos", &self.read_pos)
            .field("msk_len", &self.msk_len)
            .finish()
    }
}

#[repr(C)]
pub struct PatchingCacheEntry {
    pub metadata: PatchingCacheEntryMetadata,
    /// The mask that is applied in chunks of size `loc_size` each time the mutated
    /// location is accessed. If `loc_size` > 0, then the mask is msk_len + loc_size bytes
    /// long, else it is msk_len bytes in size.
    pub msk: [u8; 0],
}

impl std::fmt::Debug for PatchingCacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutationCacheEntry")
            .field("metadata", &self.metadata)
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
        msk_len: u32,
    ) -> Box<PatchingCacheEntry> {
        assert!(msk_len < MAX_MASK_LEN as u32);

        // In case loc_size > 0, we pad the msk_len by loc_size bytes to allow
        // read overflows during mask application.
        let mut real_msk_len = msk_len as usize;

        let mut size = std::mem::size_of::<PatchingCacheEntry>() + msk_len as usize;

        // One element padding in case read_pos overflows msk_len (see jit gen_mutation_gpr).
        if loc_size > 0 {
            size += loc_size as usize;
            real_msk_len += loc_size as usize;
        }

        let mut entry = util::alloc_box_aligned_zeroed::<PatchingCacheEntry>(size);

        entry.metadata = PatchingCacheEntryMetadata {
            id,
            vma,
            flags,
            loc_type,
            loc_size,
            dwarf_regnum,
            offset_or_constant,
            read_pos: 0,
            target_value_size_bits,
            msk_len,
        };

        // Initialize the msk to 0x00
        unsafe {
            std::ptr::write_bytes(
                entry.get_msk_as_ptr::<u8>(),
                0x00,
                real_msk_len, // Also zero the padding.
            );
        }

        entry
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
        assert!(
            new_msk_len <= MAX_MASK_LEN as u32 && new_msk_len > 0,
            "new_msk_len={}",
            new_msk_len
        );

        let mut new_size = std::mem::size_of_val(self) + new_msk_len as usize;

        // Padding for read overlow support.
        if self.loc_size() > 0 {
            new_size += self.loc_size() as usize;
        }

        // Zeroed memory
        let mut entry: Box<PatchingCacheEntry> = util::alloc_box_aligned_zeroed(new_size);

        // Copy the metadata of the old entry into the new one.

        let mut bytes_to_copy = self.size_wo_overflow_padding();
        if self.msk_len() > new_msk_len {
            // If we are shrinking the msk, do not copy all data from the old entry.
            bytes_to_copy -= (self.msk_len() - new_msk_len) as usize;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.as_ptr() as *const u8,
                entry.as_mut_ptr() as *mut u8,
                bytes_to_copy,
            );
        }

        // Adapt metadata to changed values.
        entry.metadata.msk_len = new_msk_len;
        entry
    }

    /// Get the offset off the msk_len field.
    /// We do not want to make the msk_len field public, thus we need this method.
    pub fn offsetof_msk_len() -> usize {
        offset_of!(PatchingCacheEntryMetadata, msk_len)
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

    pub fn msk_len(&self) -> u32 {
        self.metadata.msk_len
    }

    pub fn reset_flags(&mut self) -> &mut Self {
        self.metadata.flags = PatchingCacheEntryFlags::Empty as u8;
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
        self.unset_flag(PatchingCacheEntryFlags::Mutation)
    }

    pub fn disable_mutation(&mut self) -> &mut Self {
        self.set_flag(PatchingCacheEntryFlags::Mutation)
    }

    pub fn set_flag(&mut self, flag: PatchingCacheEntryFlags) -> &mut Self {
        self.metadata.flags |= flag as u8;
        self
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
        let mut ret = self.size_wo_overflow_padding();
        if self.msk_len() > 0 {
            // The msk is padded with an additional element which is used in case
            // read_pos overflows.
            ret += self.loc_size() as usize;
        }
        ret
    }

    fn size_wo_overflow_padding(&self) -> usize {
        std::mem::size_of::<PatchingCacheEntryMetadata>() + self.msk_len() as usize
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

    pub fn get_msk_as_ptr<T>(&self) -> *mut T {
        self.msk.as_ptr() as *mut T
    }

    pub fn get_msk_as_slice(&self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.get_msk_as_ptr(), self.metadata.msk_len as usize)
        }
    }

    pub fn is_nop(&self) -> bool {
        let msk = self.get_msk_as_slice();
        if msk.is_empty() {
            return true;
        }
        msk.iter().all(|v| *v == 0)
    }
}
