use derive_more::Display;
use serde::{Deserialize, Serialize};

use std::fmt::Debug;

use crate::{patchpoint::PatchPointIR};

#[derive(Clone, Debug, Display, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct FunctionId(pub String);

impl Into<FunctionId> for &PatchPointIR {
    fn into(self) -> FunctionId {
        FunctionId(self.function.clone())
    }
}

impl Into<FunctionId> for &String {
    fn into(self) -> FunctionId {
        FunctionId(self.clone())
    }
}
