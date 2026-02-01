use anyhow::Result;

use std::{
    assert_matches::assert_matches, collections::HashMap, fmt::Display, fs::OpenOptions,
    hash::Hasher, ops::Range, path::Path,
};

use crate::{
    constants::PATCH_POINT_SIZE,
    func::FunctionId,
    types::PatchPointID,
    var::{VarDeclRef, VarType},
};
use llvm_stackmap::{LLVMInstruction, LiveOut, Location, LocationType, StackMap};
use proc_maps::MapRange;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InstrumentMethod {
    StackSpill = 0,
    Direct = 1,
    NOP = 2,
}

impl Into<u32> for &InstrumentMethod {
    fn into(self) -> u32 {
        match self {
            InstrumentMethod::StackSpill => 0,
            InstrumentMethod::Direct => 1,
            InstrumentMethod::NOP => 2,
        }
    }
}

impl TryFrom<u32> for InstrumentMethod {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => InstrumentMethod::StackSpill,
            1 => InstrumentMethod::Direct,
            2 => InstrumentMethod::NOP,
            _ => return Err(format!("Invalid instrument method: {}", value)),
        })
    }
}

impl Serialize for InstrumentMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(self.into())
    }
}

impl<'de> Deserialize<'de> for InstrumentMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(InstrumentMethod::try_from(value).unwrap())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateVarType {
    #[serde(rename = "arg")]
    Arg,

    #[serde(rename = "ret")]
    Ret,

    #[serde(rename = "asan")]
    Asan,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatchpointType {
    ENTRY,
    RET,
    VALUE,
    TRAMPOLINE,
}

/// Static information about an llvm patch point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PatchPointIR {
    pub id: u32,
    pub svfg_id: Option<u32>,
    pub pag_id: Option<u32>,
    pub icfg_id: Option<u32>,
    pub is_state_var: Option<StateVarType>,
    pub module: String,
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub function: String,
    pub func_idx: u32,
    pub ins: LLVMInstruction,
    pub pp_type: PatchpointType,
    pub var: Option<VarDeclRef>,
    pub method: InstrumentMethod,
    pub detail: String,
    pub bb_name: String,
}

impl PatchPointIR {
    pub fn is_load(&self) -> bool {
        self.ins == LLVMInstruction::Load
    }

    pub fn is_store(&self) -> bool {
        self.ins == LLVMInstruction::Store
    }

    pub fn is_branch(&self) -> bool {
        self.ins == LLVMInstruction::Br || self.ins == LLVMInstruction::Switch
    }

    pub fn is_offset_variable(&self) -> bool {
        self.detail.contains("offset")
    }

    pub fn var_with_loc(&self) -> String {
        format!(
            "{}({}:{}:{})",
            self.var.as_ref().unwrap(),
            self.file,
            self.line,
            self.col
        )
    }

    // For Load and Store instructions, we patch the pointer of the variable.
    // For example, the pointer operand of the Load and Store instructions.
    // Thus the variable type infomation is embedded in a pointer, unconditionally.
    pub fn is_var_ptr_patched(&self) -> bool {
        self.ins == LLVMInstruction::Load || self.ins == LLVMInstruction::Store
    }

    pub fn function_id(&self) -> FunctionId {
        (&self.function).into()
    }

    pub fn is_func_entry(&self) -> bool {
        self.pp_type == PatchpointType::ENTRY
    }

    pub fn is_func_exit(&self) -> bool {
        self.pp_type == PatchpointType::RET
    }

    pub fn is_cmp(&self) -> bool {
        self.ins == LLVMInstruction::ICmp
            || (self.ins == LLVMInstruction::Call && self.detail.contains("memcmp"))
    }

    pub fn is_trampoline(&self) -> bool {
        self.pp_type == PatchpointType::TRAMPOLINE
    }

    pub fn display_graph_node(&self, escape: bool) -> String {
        if let Some(var) = self.var.as_ref() {
            let name = if self.is_cmp() {
                "@icmp".to_string()
            } else {
                var.as_string()
            };
            format!(
                "{}(ir#{}){} {}:{}:{}",
                name,
                self.id,
                if escape { "\\n" } else { "\n" },
                self.file,
                self.line,
                self.col
            )
        } else {
            format!("#{}", self.id)
        }
    }
}

impl Display for PatchPointIR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(var) = self.var.as_ref() {
            write!(
                f,
                "{}(ir#{},{}:{}:{})",
                var.as_string(),
                self.id,
                self.file,
                self.line,
                self.col
            )
        } else {
            write!(
                f,
                "#{},{}:{}:{}",
                self.id, self.file, self.line, self.col
            )
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchPoint {
    /// A unique ID that identifies this PatchPoint independent of the address space
    /// it belongs to.
    id: PatchPointID,
    /// The patch point ID that was assigned during compilation.
    /// May contains duplicates as parallel compilation units may have the same
    /// patch point ID, or one patchpoint in the loop may be unrolled.
    llvm_id: u32,
    /// The VMA base if this patch point belongs to binary that is position independent.
    base: u64,
    /// The VMA of this patch point. If this belongs to a PIC binary, `address`
    /// is only an offset relative to `base`.
    address: u64,
    /// The IR information of this patch point.
    ir: Option<PatchPointIR>,
    /// The live value that where recorded by this patch point.
    /// One patchpoint only contains at most one location, as we only insert one
    /// patchpoint intrinsic for one SSA value at a time.
    /// Patchpoint with no location means only its id is useful, like the AFL trampoline.
    location: Option<Location>,
    /// The memory mapping this patch point belongs to.
    mapping: MapRange,
    /// The VMA of the function that contains this PatchPoint.
    function_address: u64,
    /// The instruction offset to the function it belongs to.
    instruction_offset: u32,
    /// The location of the value that was spilled into the `spill_slot`.
    /// This is used to determine the values size, because the spill slot is
    /// located on the stack and therefore has a size that is a multiple of 8 (on 64bit).
    target_value_size_in_bit: u32,
    /// Liveout registers
    live_outs: Vec<LiveOut>,
}

impl PartialEq for PatchPoint {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PatchPoint {}

impl std::hash::Hash for PatchPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PatchPoint {
    pub fn new(
        base: u64,
        address: u64,
        llvm_id: u32,
        location: Option<Location>,
        target_value_size_in_bit: u32,
        mapping: MapRange,
        function_address: u64,
        instruction_offset: u32,
        live_outs: Vec<LiveOut>,
    ) -> Result<Self> {
        assert!(address + base > 0);

        PatchPointID::get(address as usize, mapping.inode, mapping.offset).map(|id| Self {
            id,
            llvm_id,
            base,
            address,
            ir: None,
            location,
            mapping,
            function_address,
            instruction_offset,
            target_value_size_in_bit,
            live_outs,
        })
    }

    pub fn loc_str(&self) -> String {
        let ir = self.ir.as_ref().unwrap();
        format!("{}:{}:{}", ir.file, ir.line, ir.col)
    }

    pub fn var_with_loc(&self) -> String {
        let ir = self.ir.as_ref().unwrap();
        format!(
            "{}({}:{}:{})",
            self.var().unwrap(),
            ir.file,
            ir.line,
            ir.col
        )
    }

    pub fn is_cmp(&self) -> bool {
        self.ir.as_ref().unwrap().ins == LLVMInstruction::ICmp
            || (self.ir.as_ref().unwrap().ins == LLVMInstruction::Call
                && self.ir.as_ref().unwrap().detail.contains("memcmp"))
    }

    pub fn var(&self) -> Option<&VarDeclRef> {
        self.ir.as_ref().unwrap().var.as_ref()
    }

    pub fn var_type(&self) -> Option<&VarType> {
        self.var().map(|v| v.type_enum())
    }

    pub fn ir(&self) -> &PatchPointIR {
        self.ir.as_ref().unwrap()
    }

    pub fn has_ir(&self) -> bool {
        self.ir.is_some()
    }

    pub fn ir_mut(&mut self) -> &mut Option<PatchPointIR> {
        &mut self.ir
    }

    pub fn target_value_size_in_bit(&self) -> u32 {
        self.target_value_size_in_bit
    }

    pub fn llvm_id(&self) -> u32 {
        self.llvm_id
    }

    pub fn func_idx(&self) -> u32 {
        self.ir.as_ref().unwrap().func_idx
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
                if spill_slot_location.loc_type == LocationType::ConstIndex {
                    println!("patchpoint ir id: {}", record.patch_point_id);
                }
                assert_matches!(
                    spill_slot_location.loc_type,
                    LocationType::Register | LocationType::Direct | LocationType::Constant
                );

                // assert!(locations[1].loc_type == LocationType::Constant);
                let target_value_size = locations[1].offset_or_constant;
                // // The size of the recorded value must be positive.
                // assert!(target_value_size > 0);

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

                if let Ok(pp) = PatchPoint::new(
                    base,
                    vma,
                    record.patch_point_id as u32,
                    Some(spill_slot_location.clone()),
                    target_value_size as u32,
                    mapping.clone(),
                    function_address,
                    record.instruction_offset as u32,
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

pub fn dump_patchpoints_bin(path: &Path, patch_points: &HashMap<PatchPointID, PatchPoint>) {
    log::info!("Dumping patchpoints to {:?}", path);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    // Print first 5 patch points for debugging
    // patch_points.iter().take(5).for_each(|(id, pp)| {
    //     log::info!("PatchPoint: id={:?}, pp={:?}", id, pp);
    // });

    // Serialize all patch points
    let patchpoints = patch_points.values().cloned().collect::<Vec<_>>();
    // serde_json::to_writer(&mut file, &patchpoints).unwrap();
    rmp_serde::encode::write(&mut file, &patchpoints).unwrap();

    let metadata = file.metadata().unwrap();
    log::info!("Dumped {} bytes to {:?}", metadata.len(), path);
}

pub fn load_patchpoints_bin(path: &Path) -> Vec<PatchPoint> {
    log::info!("Loading patchpoints from {:?}", path);
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| {
            log::error!("Failed to open file {:?}: {}", path, e);
            e
        })
        .unwrap();
    // let patchpoints: Vec<PatchPoint> = serde_json::from_reader(&file).expect("Failed to deserialize patchpoints");
    let patchpoints: Vec<PatchPoint> =
        rmp_serde::decode::from_read(&file).expect("Failed to deserialize patchpoints");
    log::info!("Loaded {} patchpoints", patchpoints.len());
    patchpoints
}
