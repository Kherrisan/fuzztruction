use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub enum LRValue {
    #[default]
    LValue,
    RValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub struct VarDeclRef {
    pub name: String,
    ty: VarTypeWithTypedef,
    pub is_local: bool,
    pub is_param: bool,
    pub is_global: bool,
    pub parent: Option<Box<VarDeclRef>>,
}

impl VarDeclRef {
    pub fn type_tracable(&self) -> bool {
        self.type_enum().val_tracable()
    }

    pub fn type_enum(&self) -> &VarType {
        &self.ty.ty
    }

    pub fn typedef_name(&self) -> Option<String> {
        self.ty.typedef_name.clone()
    }

    pub fn as_string(&self) -> String {
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
        Ok(self.ty.ty)
    }
}

impl Display for VarDeclRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, derive_more::Display)]
pub enum VarType {
    #[display("bitfield({:b})", ((1 as u128) << width - 1) << offset)]
    Bitfield { offset: u16, width: u16 },
    #[display("u{bits}")]
    Int { bits: u16 },
    #[display("f{bits}")]
    Float { bits: u16 },
    #[display("{pointee}*")]
    Pointer { pointee: Box<VarTypeWithTypedef> },
    #[display("[{elem_type}; {length:?}]")]
    Array {
        elem_type: Box<VarTypeWithTypedef>,
        length: Option<usize>,
    },
    #[display("{name}")]
    Struct { name: String },
    #[display("{name}")]
    Other { name: String },
}

impl VarType {
    pub fn dummy_ptr() -> Self {
        VarType::Pointer {
            pointee: Box::new(VarTypeWithTypedef {
                ty: VarType::Other {
                    name: "".to_string(),
                },
                typedef_name: None,
            }),
        }
    }

    fn compatible(&self, other: &VarType) -> bool {
        let derefed_self = self.dereference();
        let derefed_other = other.dereference();
        if !derefed_self.is_some() || !derefed_other.is_some() {
            return false;
        }
        let derefed_self = derefed_self.unwrap();
        let derefed_other = derefed_other.unwrap();
        if derefed_self == derefed_other {
            return true;
        }
        // For example:
        // *u8, **u8 and *(u8[])

        return true;
    }
}

impl Default for VarType {
    fn default() -> Self {
        VarType::Other {
            name: "".to_string(),
        }
    }
}

#[derive(Error, Debug)]
pub enum InterpretError {
    #[error("Bitfield overflow: offset {0}, width {1}")]
    BitfieldOverflow(u16, u16),

    #[error("Address poisoned, type: {0}")]
    AddressPoisoned(VarType),
}

impl VarType {
    pub fn bytes(&self) -> u64 {
        match self {
            VarType::Int { bits } => (*bits as u64 + 7) / 8,
            VarType::Float { bits } => (*bits as u64 + 7) / 8,
            VarType::Bitfield { width, offset } => match *width + *offset {
                1..=8 => 1,
                9..=16 => 2,
                17..=32 => 4,
                33..=64 => 8,
                _ => {
                    16
                }
            },
            VarType::Pointer {
                pointee: pointee_type,
            } => pointee_type.ty.bytes(),
            VarType::Array { elem_type, length } => {
                elem_type.ty.bytes() * length.unwrap_or(1) as u64
            }
            VarType::Other { .. } => 8,
            VarType::Struct { .. } => 8,
        }
    }

    pub fn dereference(&self) -> Option<&VarTypeWithTypedef> {
        match self {
            VarType::Pointer { pointee } => Some(pointee),
            VarType::Array { elem_type, .. } => Some(elem_type),
            _ => None,
        }
    }

    pub fn val_tracable(&self) -> bool {
        if let Some(derefed_type) = self.dereference() {
            match &derefed_type.ty {
                VarType::Int { .. } => true,
                VarType::Float { .. } => true,
                VarType::Bitfield { .. } => self.bytes() <= 8,
                VarType::Pointer { pointee } => match pointee.ty {
                    VarType::Int { .. } => true,
                    VarType::Float { .. } => true,
                    VarType::Bitfield { .. } => self.bytes() <= 8,
                    _ => false,
                },
                VarType::Array { elem_type, .. } => match elem_type.ty {
                    VarType::Int { .. } => true,
                    VarType::Float { .. } => true,
                    VarType::Bitfield { .. } => self.bytes() <= 8,
                    _ => false,
                },
                _ => false,
            }
        } else {
            false
        }
    }

    pub fn interpret(&self, value: &[u8]) -> Result<String> {
        if value.len() == 0 {
            return Err(InterpretError::AddressPoisoned(self.clone()).into());
        }
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
                if total_bits > 128 {
                    return Err(InterpretError::BitfieldOverflow(*width, *offset).into());
                }
                let ty_bits = total_bits.next_power_of_two();
                let mask = (((1 as u128) << *width) - 1) << *offset;
                let val = if ty_bits <= 8 {
                    (value[0] as u128 & mask) >> *offset
                } else if ty_bits <= 16 {
                    ((u16::from_le_bytes(value[..2].try_into()?) as u128 & mask) >> *offset) as u128
                } else if ty_bits <= 32 {
                    ((u32::from_le_bytes(value[..4].try_into()?) as u128 & mask) >> *offset) as u128
                } else {
                    ((u64::from_le_bytes(value[..8].try_into()?) as u128 & mask) >> *offset) as u128
                };
                Ok(format!("{}", val))
            }
            VarType::Pointer {
                pointee: pointee_type,
            } => {
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
            VarType::Struct { name } => Ok(format!("{}", name)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub struct VarTypeWithTypedef {
    #[serde(flatten)]
    pub ty: VarType,
    pub typedef_name: Option<String>,
}

impl VarTypeWithTypedef {
    pub fn bytes(&self) -> u64 {
        self.ty.bytes()
    }

    pub fn interpret(&self, value: &[u8]) -> Result<String> {
        self.ty.interpret(value)
    }
}

impl From<&VarType> for VarTypeWithTypedef {
    fn from(ty: &VarType) -> Self {
        VarTypeWithTypedef {
            ty: ty.clone(),
            typedef_name: None,
        }
    }
}

impl From<VarType> for VarTypeWithTypedef {
    fn from(ty: VarType) -> Self {
        VarTypeWithTypedef {
            ty,
            typedef_name: None,
        }
    }
}

impl Display for VarTypeWithTypedef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty)
    }
}
