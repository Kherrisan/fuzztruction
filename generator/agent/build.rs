// use std::{path::PathBuf, process};

use std::{path::PathBuf, process};

fn main() {
    println!("cargo:rerun-if-changed=pass/fuzztruction-preprocessing-pass.cc");
    println!("cargo:rerun-if-changed=pass/fuzztruction-source-clang-fast.c");
    println!("cargo:rerun-if-changed=pass/fuzztruction-source-llvm-pass.cc");

    println!("Building source llvm pass...");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut cmd = process::Command::new("make");
    let cwd = PathBuf::from(manifest_dir).join("../pass");
    cmd.current_dir(cwd);
    cmd.spawn().unwrap();
}
