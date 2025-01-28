use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fmt::Display;

#[derive(Clone, Hash, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VarDeclRefID {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub name: String,
}

impl VarDeclRefID {
    pub fn same_line_with(&self, file: &str, line: u32) -> bool {
        self.file == file && self.line == line
    }

    pub fn name(&self) -> String {
        format!("{}({}:{}:{})", self.name, self.file, self.line, self.col)
    }

    pub fn serialized_name(&self) -> String {
        format!("{}:{}:{}:{}", self.file, self.line, self.col, self.name)
    }
}

impl From<String> for VarDeclRefID {
    fn from(s: String) -> Self {
        let mut iter = s.split(':');
        let file = iter.next().unwrap();
        let line = iter.next().unwrap();
        let col = iter.next().unwrap();
        let name = iter.next().unwrap();
        VarDeclRefID {
            file: file.to_string(),
            line: line.parse().unwrap(),
            col: col.parse().unwrap(),
            name: name.to_string(),
        }
    }
}

impl Display for VarDeclRefID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Debug for VarDeclRefID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VarType {
    Bitfield {
        offset: u16,
        width: u16,
    },
    Int {
        bits: u16,
    },
    Float {
        bits: u16,
    },
    Pointer {
        pointee_type: Box<VarType>,
    },
    Array {
        elem_type: Box<VarType>,
        length: Option<usize>,
    },
    Other {
        name: String,
    },
}

impl VarType {
    pub fn bytes(&self) -> u64 {
        match self {
            VarType::Int { bits } => (*bits as u64 + 7) / 8,
            VarType::Float { bits } => (*bits as u64 + 7) / 8,
            VarType::Bitfield { width, .. } => (*width as u64 + 7) / 8,
            VarType::Pointer { pointee_type } => pointee_type.bytes(),
            VarType::Array { elem_type, length } => elem_type.bytes() * length.unwrap_or(1) as u64,
            VarType::Other { .. } => 8,
        }
    }

    pub fn tracable(&self) -> bool {
        match self {
            VarType::Int { .. } => true,
            VarType::Float { .. } => true,
            VarType::Bitfield { .. } => true,
            VarType::Pointer { pointee_type } => pointee_type.tracable(),
            VarType::Array { elem_type, .. } => elem_type.tracable(),
            _ => false,
        }
    }
}
