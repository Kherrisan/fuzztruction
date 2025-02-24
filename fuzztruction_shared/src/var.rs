use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub enum LRValue {
    #[default]
    LValue,
    RValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub struct VarDeclRef {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: VarType,
    pub is_local: bool,
    pub is_param: bool,
    pub is_global: bool,
    pub parent: Box<Option<VarDeclRef>>,
}

impl VarDeclRef {
    fn as_string(&self) -> String {
        if let Some(parent) = self.parent.as_ref() {
            format!("{}->{}", parent.as_string(), self.name)
        } else {
            format!("{}", self.name)
        }
    }
}

impl TryInto<VarType> for VarDeclRef {
    type Error = anyhow::Error;

    fn try_into(self) -> std::result::Result<VarType, Self::Error> {
        Ok(self.ty)
    }
}

impl Display for VarDeclRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_string())
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

impl Default for VarType {
    fn default() -> Self {
        VarType::Other {
            name: "".to_string(),
        }
    }
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

    pub fn interpret(&self, value: &[u8]) -> Result<String> {
        match self {
            VarType::Int { bits } => {
                let val = if *bits <= 8 {
                    value[0] as u64
                } else if *bits <= 16 {
                    u16::from_le_bytes(value[..2].try_into()?) as u64
                } else if *bits <= 32 {
                    u32::from_le_bytes(value[..4].try_into()?) as u64
                } else {
                    u64::from_le_bytes(value[..8].try_into()?)
                };
                Ok(format!("{}", val))
            }
            VarType::Float { bits } => {
                let val: f64 = if *bits <= 32 {
                    f64::from_le_bytes(value[..4].try_into()?)
                } else {
                    f64::from_le_bytes(value[..8].try_into()?)
                };
                Ok(format!("{}", val))
            }
            VarType::Bitfield { offset, width } => {
                let total_bits = *width + *offset;
                let ty_bits = total_bits.next_power_of_two();
                let mask = ((1 << *width) - 1) << *offset;
                let val = if ty_bits <= 8 {
                    (value[0] as u64 & mask) >> *offset
                } else if ty_bits <= 16 {
                    ((u16::from_le_bytes(value[..2].try_into()?) as u64 & mask) >> *offset) as u64
                } else if ty_bits <= 32 {
                    ((u32::from_le_bytes(value[..4].try_into()?) as u64 & mask) >> *offset) as u64
                } else {
                    ((u64::from_le_bytes(value[..8].try_into()?) & mask) >> *offset) as u64
                };
                Ok(format!("{}", val))
            }
            VarType::Pointer { pointee_type } => {
                let val = pointee_type.interpret(value)?;
                Ok(format!("{}", val))
            }
            VarType::Array { elem_type, length } => {
                let elem_size = elem_type.bytes();
                let mut result = String::new();
                result.push('[');

                for i in 0..length.unwrap_or(1) {
                    let start = i * elem_size as usize;
                    let end = start + elem_size as usize;
                    if i > 0 {
                        result.push_str(", ");
                    }
                    let elem_val = elem_type.interpret(&value[start..end])?;
                    result.push_str(&elem_val);
                }

                result.push(']');
                Ok(result)
            }
            VarType::Other { name } => Ok(format!("{}", name)),
        }
    }
}
