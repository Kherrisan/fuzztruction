// use std::{path::PathBuf, process};

use std::{path::PathBuf, process};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rerun-if-changed={}/../pass/fuzztruction-preprocessing-pass.cc", manifest_dir);
    println!("cargo:rerun-if-changed={}/../pass/fuzztruction-source-clang-fast.c", manifest_dir);
    println!("cargo:rerun-if-changed={}/../pass/fuzztruction-source-llvm-pass.cc", manifest_dir);

    println!("Building source llvm pass...");
    
    let mut cmd = process::Command::new("make");
    let cwd = PathBuf::from(manifest_dir).join("../pass");
    cmd.current_dir(cwd);
    let mut ret = cmd.spawn().unwrap();
    ret.wait().unwrap();
}
