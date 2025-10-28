use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockAddressTableEntry {
    pub func_idx: u32,
    pub block_idx: u32,
    pub block_addr: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct FunctionBlockEntry {
    pub func_name: String,
    pub block_name: String,
    pub func_idx: u32,
    pub block_idx: u32,
}

pub struct FunctionBlockTable {
    pub table: HashMap<String, (u32, HashMap<String, u32>)>,
}

impl FunctionBlockTable {
    pub fn load(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let entries: Vec<FunctionBlockEntry> = serde_json::from_reader(reader).unwrap();
        let mut table = HashMap::new();
        entries.into_iter().for_each(
            |FunctionBlockEntry {
                 func_name,
                 block_name,
                 func_idx,
                 block_idx,
             }| {
                table
                    .entry(func_name)
                    .or_insert_with(|| (func_idx, HashMap::new()))
                    .1
                    .insert(block_name, block_idx);
            },
        );
        Self { table }
    }

    pub fn get(&self, func: &String, block: &String) -> Option<(usize, usize)> {
        self.table.get(func).and_then(|(func_idx, block_table)| {
            block_table
                .get(block)
                .map(|block_idx| (*func_idx as usize, *block_idx as usize))
        })
    }
}
