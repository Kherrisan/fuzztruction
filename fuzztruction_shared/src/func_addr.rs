use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionTableEntry {
    pub func_name: String,
    pub func_idx: u32,
}

pub struct FunctionTable {
    table: HashMap<String, u32>,
}

impl FunctionTable {
    pub fn load(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let entries: Vec<FunctionTableEntry> = serde_json::from_reader(reader).unwrap();
        let mut table = HashMap::new();
        entries.into_iter().for_each(|entry| {
            table.insert(entry.func_name, entry.func_idx);
        });
        FunctionTable { table }
    }

    pub fn get(&self, func_name: &str) -> Option<u32> {
        self.table.get(func_name).cloned()
    }
}
