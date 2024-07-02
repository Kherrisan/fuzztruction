use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct PatchPointID(pub u32);

static PATCH_POINT_ID_INVALID: u32 = u32::MAX;
lazy_static! {
    static ref PATCH_POINT_ID_MAP: Mutex<HashMap<(usize, usize, usize), PatchPointID>> =
        Mutex::new(HashMap::new());
}

impl PatchPointID {
    // Record a reverse index map: (vma, inode, offset) -> PatchPointID
    pub fn get(
        id: u32,
        base_offset: usize,
        inode: usize,
        section_file_offset: usize,
    ) -> PatchPointID {
        let mut map = PATCH_POINT_ID_MAP.lock().unwrap();
        let key = (base_offset, inode, section_file_offset);
        let pp = PatchPointID(id);
        let had_pp = map.insert(key, pp.clone());
        assert!(
            had_pp.is_none(),
            "There was already an entry for the given key!"
        );
        pp
    }

    pub fn invalid() -> PatchPointID {
        PatchPointID(PATCH_POINT_ID_INVALID)
    }
}

impl ToString for PatchPointID {
    fn to_string(&self) -> String {
        format!("PatchPointID({})", self.0)
    }
}

impl From<PatchPointID> for u32 {
    fn from(pp: PatchPointID) -> Self {
        pp.0
    }
}

impl From<u32> for PatchPointID {
    fn from(v: u32) -> Self {
        PatchPointID(v)
    }
}

impl From<PatchPointID> for u64 {
    fn from(pp: PatchPointID) -> Self {
        pp.0 as u64
    }
}

impl From<u64> for PatchPointID {
    fn from(v: u64) -> Self {
        PatchPointID(v as u32)
    }
}

impl From<usize> for PatchPointID {
    fn from(v: usize) -> Self {
        PatchPointID(v as u32)
    }
}

impl From<&PatchPointID> for usize {
    fn from(pp: &PatchPointID) -> Self {
        pp.0 as usize
    }
}

impl From<PatchPointID> for usize {
    fn from(pp: PatchPointID) -> Self {
        pp.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VAddr(pub u64);

macro_rules! implement_from_for_multiple {
    ($t:ty) => {
        impl From<$t> for VAddr {
            fn from(v: $t) -> Self {
                VAddr(v as u64)
            }
        }
    };
    ($t:ty, $($tt:ty),+) => {
        impl From<$t> for VAddr {
            fn from(v: $t) -> Self {
                VAddr(v as u64)
            }
        }
        implement_from_for_multiple!($($tt),+);
    };
}

implement_from_for_multiple!(u8, u16, u32, u64, usize);
