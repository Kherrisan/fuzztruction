use anyhow::Context;
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
    ty: VarType,
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
        &self.ty
    }

    pub fn typedef_name(&self) -> Option<String> {
        match &self.ty {
            VarType::Void => None,
            VarType::Bitfield { typedef, .. } => typedef.clone(),
            VarType::Int { typedef, .. } => typedef.clone(),
            VarType::Float { typedef, .. } => typedef.clone(),
            VarType::Pointer { typedef, .. } => typedef.clone(),
            VarType::Array { typedef, .. } => typedef.clone(),
            VarType::Struct { typedef, .. } => typedef.clone(),
            VarType::Enum { typedef, .. } => typedef.clone(),
            VarType::Union { typedef, .. } => typedef.clone(),
            VarType::Other { typedef, .. } => typedef.clone(),
        }
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
        Ok(self.ty)
    }
}

impl Display for VarDeclRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, derive_more::Display)]
#[serde(tag = "type")]
pub enum VarType {
    #[display("bitfield({:b})", ((1 as u128) << width - 1) << offset)]
    Bitfield {
        offset: u16,
        width: u16,
        typedef: Option<String>,
    },
    #[display("u{width}")]
    Int { width: u16, typedef: Option<String> },
    #[display("f{width}")]
    Float { width: u16, typedef: Option<String> },
    #[display("{pointee:?}*")]
    Pointer {
        pointee: Option<Box<VarType>>,
        typedef: Option<String>,
    },
    #[display("[{element}; {size:?}]")]
    Array {
        element: Box<VarType>,
        size: Option<usize>,
        typedef: Option<String>,
    },
    #[display("{name}")]
    Struct {
        name: String,
        typedef: Option<String>,
    },
    #[display("{name}")]
    Enum {
        name: String,
        typedef: Option<String>,
    },
    #[display("{name}")]
    Union {
        name: String,
        typedef: Option<String>,
    },
    #[display("{name}")]
    Other {
        name: String,
        typedef: Option<String>,
    },
    #[display("void")]
    Void,
}

impl VarType {
    pub fn dummy_ptr() -> Self {
        VarType::Pointer {
            pointee: Some(Box::new(VarType::Other {
                name: "void".to_string(),
                typedef: None,
            })),
            typedef: None,
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

impl PartialEq for VarType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VarType::Void, VarType::Void) => true,
            (
                VarType::Bitfield {
                    offset: o1,
                    width: w1,
                    ..
                },
                VarType::Bitfield {
                    offset: o2,
                    width: w2,
                    ..
                },
            ) => o1 == o2 && w1 == w2,
            (VarType::Int { width: w1, .. }, VarType::Int { width: w2, .. }) => w1 == w2,
            (VarType::Float { width: w1, .. }, VarType::Float { width: w2, .. }) => w1 == w2,
            (VarType::Pointer { pointee: p1, .. }, VarType::Pointer { pointee: p2, .. }) => {
                p1 == p2
            }
            (
                VarType::Array {
                    element: e1,
                    size: s1,
                    ..
                },
                VarType::Array {
                    element: e2,
                    size: s2,
                    ..
                },
            ) => e1 == e2 && s1 == s2,
            (VarType::Struct { name: n1, .. }, VarType::Struct { name: n2, .. }) => n1 == n2,
            (VarType::Enum { name: n1, .. }, VarType::Enum { name: n2, .. }) => n1 == n2,
            (VarType::Union { name: n1, .. }, VarType::Union { name: n2, .. }) => n1 == n2,
            (VarType::Other { name: n1, .. }, VarType::Other { name: n2, .. }) => n1 == n2,
            _ => false,
        }
    }
}

impl Eq for VarType {}

impl Default for VarType {
    fn default() -> Self {
        VarType::Other {
            name: "".to_string(),
            typedef: None,
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
    pub fn num_bytes(&self) -> u64 {
        match self {
            VarType::Void => 0,
            VarType::Int { width, .. } => (*width as u64 + 7) / 8,
            VarType::Float { width, .. } => (*width as u64 + 7) / 8,
            VarType::Bitfield { width, offset, .. } => match *width + *offset {
                1..=8 => 1,
                9..=16 => 2,
                17..=32 => 4,
                33..=64 => 8,
                _ => 16,
            },
            VarType::Pointer { pointee: None, .. } => 8,
            VarType::Pointer {
                pointee: Some(pointee),
                ..
            } => pointee.num_bytes(),
            VarType::Array { size, .. } => (size.unwrap_or(1) as u64 + 7) / 8,
            VarType::Other { .. } => 8,
            VarType::Struct { .. } => 8,
            VarType::Enum { .. } => 4,
            VarType::Union { .. } => 8,
        }
    }

    pub fn dereference(&self) -> Option<&VarType> {
        match self {
            VarType::Pointer { pointee: None, .. } => None,
            VarType::Pointer {
                pointee: Some(pointee),
                ..
            } => Some(pointee),
            VarType::Array { element, .. } => Some(element),
            _ => None,
        }
    }

    pub fn is_pointer(&self) -> bool {
        match self {
            VarType::Pointer { .. } => true,
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            VarType::Int { .. } => true,
            VarType::Float { .. } => true,
            VarType::Bitfield { .. } => true,
            VarType::Enum { .. } => true,
            _ => false,
        }
    }

    pub fn is_composite(&self) -> bool {
        match self {
            VarType::Pointer { .. } => true,
            VarType::Array { .. } => true,
            VarType::Struct { .. } => true,
            VarType::Union { .. } => true,
            _ => false,
        }
    }

    pub fn val_tracable(&self) -> bool {
        if let Some(derefed_type) = self.dereference() {
            match &derefed_type {
                VarType::Int { .. } => true,
                VarType::Float { .. } => true,
                VarType::Bitfield { .. } => self.num_bytes() <= 8,
                VarType::Pointer { pointee: None, .. } => true,
                VarType::Pointer {
                    pointee: Some(pointee),
                    ..
                } => match pointee.as_ref() {
                    VarType::Int { .. } => true,
                    VarType::Float { .. } => true,
                    VarType::Bitfield { .. } => self.num_bytes() <= 8,
                    _ => false,
                },
                VarType::Array { element, .. } => match element.as_ref() {
                    VarType::Int { .. } => true,
                    VarType::Float { .. } => true,
                    VarType::Bitfield { .. } => self.num_bytes() <= 8,
                    _ => false,
                },
                _ => false,
            }
        } else {
            match self {
                VarType::Int { .. }
                | VarType::Float { .. }
                | VarType::Pointer { .. }
                | VarType::Enum { .. }
                | VarType::Union { .. }
                | VarType::Bitfield { .. } => self.num_bytes() <= 8,
                _ => false,
            }
        }
    }

    pub fn interpret(&self, value: &[u8]) -> Result<String> {
        if value.len() == 0 {
            return Ok("null".to_string());
        }
        match self {
            VarType::Void => Ok("void".to_string()),
            VarType::Int { width, .. } => {
                let val = if *width <= 8 {
                    value[0] as u64
                } else if *width <= 16 {
                    u16::from_le_bytes(value[..2].try_into()?) as u64
                } else if *width <= 32 {
                    u32::from_le_bytes(value[..4].try_into()?) as u64
                } else {
                    u64::from_le_bytes(value[..8].try_into()?)
                };
                Ok(format!("{}", val))
            }
            VarType::Float { width, .. } => {
                let val: f64 = if *width <= 32 {
                    f64::from_le_bytes(value[..4].try_into()?)
                } else {
                    f64::from_le_bytes(value[..8].try_into()?)
                };
                Ok(format!("{}", val))
            }
            VarType::Bitfield { offset, width, .. } => {
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
            VarType::Pointer { pointee: None, .. } => {
                let address = u64::from_le_bytes(value[..8].try_into()?) as usize;
                Ok(format!("0x{:x}", address))
            }
            VarType::Pointer {
                pointee: Some(pointee_type),
                ..
            } => {
                let val = pointee_type
                    .interpret(value)
                    .context(format!("Failed to interpret {}", self))?;
                Ok(format!("{}", val))
            }
            VarType::Array { element, size, .. } => {
                let elem_bytes = element.num_bytes() as usize;
                let bytes = if let Some(size) = size {
                    size / 8
                } else {
                    elem_bytes
                };
                let len = bytes / elem_bytes;
                let mut result = String::new();
                result.push('[');

                for i in 0..len {
                    let start = i * elem_bytes;
                    let end = start + elem_bytes;
                    if i > 0 {
                        result.push_str(", ");
                    }
                    let elem_val = value
                        .get(start..end)
                        .context(format!("Failed to get {}..{} from {}", start, end, self))?;
                    let elem_val = element.interpret(elem_val).context(format!(
                        "Failed to interpret {}-th element with type {}",
                        i, element
                    ))?;
                    result.push_str(&elem_val);
                }

                result.push(']');
                Ok(result)
            }
            VarType::Enum { name, .. } => Ok(format!("{}", name)),
            VarType::Union { name, .. } => Ok(format!("{}", name)),
            VarType::Other { name, .. } => Ok(format!("{}", name)),
            VarType::Struct { name, .. } => Ok(format!("{}", name)),
        }
    }

    pub fn word(&self) -> String {
        if self.num_bytes() <= 8 {
            "byte".to_string()
        } else if self.num_bytes() <= 16 {
            "word".to_string()
        } else if self.num_bytes() <= 32 {
            "dword".to_string()
        } else if self.num_bytes() <= 64 {
            "qword".to_string()
        } else {
            panic!("Unsupported operand width: {}", self.num_bytes());
        }
    }
}
