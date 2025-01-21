use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fmt::Display;

#[derive(Clone, Hash, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VarDeclRefID(pub String, pub u32, pub u32);

impl VarDeclRefID {
    pub fn same_line_with(&self, other: &VarDeclRefID) -> bool {
        self.0 == other.0 && self.1 == other.1
    }

    pub fn name(&self) -> String {
        format!("{}:{}:{}", self.0, self.1, self.2)
    }
}

impl From<String> for VarDeclRefID {
    fn from(s: String) -> Self {
        let mut iter = s.split(':');
        let file = iter.next().unwrap();
        let line = iter.next().unwrap();
        let col = iter.next().unwrap();
        VarDeclRefID(
            file.to_string(),
            line.parse().unwrap(),
            col.parse().unwrap(),
        )
    }
}

impl Display for VarDeclRefID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.0, self.1, self.2)
    }
}

impl Debug for VarDeclRefID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.0, self.1, self.2)
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
