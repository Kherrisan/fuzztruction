use anyhow::Result;
use std::{
    assert_matches::assert_matches, collections::HashMap, fs::OpenOptions, ops::Range, path::Path,
};

use crate::{
    constants::PATCH_POINT_SIZE,
    mutation_cache::MutationCacheEntryFlags,
    mutation_cache_entry::MutationCacheEntry,
    types::PatchPointID,
    var::{VarDeclRefID, VarType},
};
use llvm_stackmap::{LLVMInstruction, LiveOut, Location, LocationType, StackMap};
use proc_maps::MapRange;
use serde::{Deserialize, Serialize};

/// Static information about an llvm patch point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PatchPointIR {
    pub id: u64,
    pub module_name: String,
    pub file_name: String,
    pub line: u32,
    pub col: u32,
    pub func_name: String,
    pub ins: LLVMInstruction,
    pub var_name: String,
    pub is_func_entry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PatchPoint {
    /// A unique ID that identifies this PatchPoint independent of the address space
    /// it belongs to.
    id: PatchPointID,
    /// The patch point ID that was assigned during compilation.
    /// May contains duplicates as parallel compilation units may have the same
    /// patch point ID, or one patchpoint in the loop may be unrolled.
    llvm_id: u64,
    /// The VMA base if this patch point belongs to binary that is position independent.
    base: u64,
    /// The VMA of this patch point. If this belongs to a PIC binary, `address`
    /// is only an offset relative to `base`.
    address: u64,
    /// The IR information of this patch point.
    ir: Option<PatchPointIR>,
    /// The variable ID of this patch point.
    var_id: Option<VarDeclRefID>,
    /// The variable type information of this patch point.
    var_type: Option<VarType>,
    /// The live value that where recorded by this patch point.
    /// One patchpoint only contains at most one location, as we only insert one
    /// patchpoint intrinsic for one SSA value at a time.
    /// Patchpoint with no location means only its id is useful, like the AFL trampoline.
    location: Option<Location>,
    /// The memory mapping this patch point belongs to.
    mapping: MapRange,
    /// The VMA of the function that contains this PatchPoint.
    function_address: u64,
    /// The location of the value that was spilled into the `spill_slot`.
    /// This is used to determine the values size, because the spill slot is
    /// located on the stack and therefore has a size that is a multiple of 8 (on 64bit).
    target_value_size_in_bit: u32,
    /// Liveout registers
    live_outs: Vec<LiveOut>,
}

impl PatchPoint {
    pub fn new(
        base: u64,
        address: u64,
        llvm_id: u64,
        location: Option<Location>,
        target_value_size_in_bit: u32,
        mapping: MapRange,
        function_address: u64,
        live_outs: Vec<LiveOut>,
    ) -> Result<Self> {
        assert!(address + base > 0);

        PatchPointID::get(address as usize, mapping.inode, mapping.offset).map(|id| Self {
            id,
            llvm_id,
            base,
            address,
            ir: None,
            var_id: None,
            var_type: None,
            location,
            mapping,
            function_address,
            target_value_size_in_bit,
            live_outs,
        })
    }

    pub fn var_type(&self) -> &Option<VarType> {
        &self.var_type
    }

    pub fn var_type_mut(&mut self) -> &mut Option<VarType> {
        &mut self.var_type
    }

    pub fn var_id(&self) -> &Option<VarDeclRefID> {
        &self.var_id
    }

    pub fn var_id_mut(&mut self) -> &mut Option<VarDeclRefID> {
        &mut self.var_id
    }

    pub fn ir(&self) -> &Option<PatchPointIR> {
        &self.ir
    }

    pub fn ir_mut(&mut self) -> &mut Option<PatchPointIR> {
        &mut self.ir
    }

    pub fn target_value_size_in_bit(&self) -> u32 {
        self.target_value_size_in_bit
    }

    pub fn llvm_id(&self) -> u64 {
        self.llvm_id
    }

    pub fn id(&self) -> PatchPointID {
        self.id
    }

    pub fn mapping(&self) -> &MapRange {
        &self.mapping
    }

    pub fn function_address(&self) -> u64 {
        self.function_address
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn vma(&self) -> u64 {
        self.base + self.address
    }

    pub fn vma_range(&self) -> Range<u64> {
        self.vma()..(self.vma() + PATCH_POINT_SIZE as u64)
    }

    pub fn location(&self) -> &Option<Location> {
        &self.location
    }

    pub fn into_mutation_cache_entry(&self) -> Box<MutationCacheEntry> {
        self.into()
    }

    pub fn load(path: &Path) -> Vec<PatchPoint> {
        let file = OpenOptions::new().read(true).open(path).unwrap();
        serde_json::from_reader(file).unwrap()
    }

    pub fn dump(path: &Path, patch_points: &[PatchPoint]) {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(file, patch_points).unwrap();
    }

    pub fn live_outs(&self) -> &Vec<LiveOut> {
        &self.live_outs
    }

    pub fn from_stackmap(
        map: &StackMap,
        mapping: &MapRange,
        elf_file: &elf::ElfBytes<elf::endian::AnyEndian>,
    ) -> Vec<PatchPoint> {
        let mut idx: usize = 0;
        let mut patch_points = Vec::new();

        // If it is PIC, the base is the start address of the mapping.
        // If not, the addresses in the stackmap are absolute.
        assert!(matches!(
            elf_file.ehdr.e_type,
            elf::abi::ET_DYN | elf::abi::ET_EXEC
        ));
        let is_pic = elf_file.ehdr.e_type == elf::abi::ET_DYN;
        let base = is_pic.then(|| mapping.start()).unwrap_or(0) as u64;

        for function in &map.stk_size_records {
            assert!(function.function_address > 0);
            let records = &map.stk_map_records[idx..(idx + function.record_count as usize)];
            records.iter().for_each(|record| {
                if record.locations.is_empty() {
                    log::warn!("StkMapRecord without recorded locations");
                }
                let locations = &record.locations;
                assert_eq!(locations.len(), 2);

                let spill_slot_location = &locations[0];
                assert_matches!(
                    spill_slot_location.loc_type,
                    LocationType::Register | LocationType::Direct
                );

                assert!(locations[1].loc_type == LocationType::Constant);
                let target_value_size = locations[1].offset_or_constant;
                // The size of the recorded value must be positive.
                assert!(target_value_size > 0);

                let mut vma = (function.function_address as usize
                    + record.instruction_offset as usize) as u64;
                // Rebased function address
                let mut function_address = base + function.function_address;
                if is_pic {
                    vma -= mapping.offset as u64;
                    function_address -= mapping.offset as u64;

                    // Sanity check
                    let absolute_vma = vma as u64 + mapping.start() as u64;
                    assert!(
                        (mapping.start() as u64 + mapping.size() as u64) > absolute_vma,
                        "vma 0x{:x} is too big for mapping {:#?}! record={:#?}",
                        absolute_vma,
                        mapping,
                        record
                    );
                }

                if record.patch_point_id == 134260 {
                    println!("adfad");
                }

                if let Ok(pp) = PatchPoint::new(
                    base,
                    vma,
                    record.patch_point_id,
                    Some(spill_slot_location.clone()),
                    target_value_size as u32,
                    mapping.clone(),
                    function_address,
                    record.live_outs.clone(),
                ) {
                    patch_points.push(pp);
                } else {
                    log::error!("Duplicated PatchPointID: {}", record.patch_point_id);
                    panic!("Duplicated PatchPointID: {}", record.patch_point_id)
                }
            });
            idx += function.record_count as usize;
        }

        patch_points
    }
}

impl From<&PatchPoint> for Box<MutationCacheEntry> {
    fn from(pp: &PatchPoint) -> Self {
        let (loc_type, loc_size, dwarf_regnum, offset_or_constant) = if let Some(l) = pp.location()
        {
            (l.loc_type, l.loc_size, l.dwarf_regnum, l.offset_or_constant)
        } else {
            (llvm_stackmap::LocationType::Invalid, 0, 0, 0)
        };

        MutationCacheEntry::new(
            pp.id(),
            pp.vma(),
            MutationCacheEntryFlags::Empty as u8,
            loc_type,
            loc_size,
            dwarf_regnum.try_into().unwrap(),
            offset_or_constant,
            pp.target_value_size_in_bit(),
            0,
        )
    }
}

pub fn dump_patchpoints_bin(path: &Path, patch_points: &HashMap<PatchPointID, PatchPoint>) {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    // Print first 5 patch points for debugging
    patch_points.iter().take(5).for_each(|(id, pp)| {
        log::info!("PatchPoint: id={:?}, pp={:?}", id, pp);
    });

    // Serialize all patch points
    let patchpoints = patch_points.values().cloned().collect::<Vec<_>>();
    bincode::serialize_into(&mut file, &patchpoints).unwrap();

    let metadata = file.metadata().unwrap();
    log::info!("Dumped {} bytes to {:?}", metadata.len(), path);
}

pub fn load_patchpoints_bin(path: &Path) -> Vec<PatchPoint> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .expect(format!("Failed to open patchpoints at {path:?}").as_str());
    bincode::deserialize_from(&file).expect("Failed to deserialize patchpoints")
}
