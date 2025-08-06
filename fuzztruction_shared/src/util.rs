use chrono::{DateTime, Local};
use libafl_bolts::rands::Rand;
use log::log_enabled;
use nix::sys::signal::Signal;
use serde::Serialize;
use std::{
    alloc,
    convert::TryInto,
    fs::{self, read_link, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

pub trait ExpectNone {
    /// Whether this value is None.
    fn is_none(&self) -> bool;

    /// Panics with the given message if the value is not None.
    #[track_caller]
    fn expect_none(&self, msg: &str) {
        if !self.is_none() {
            panic!("Expected None: {}", msg);
        }
    }

    /// Panic if the value is not None.
    #[track_caller]
    fn unwrap_none(&self) {
        if !self.is_none() {
            panic!("Expected to unwrap a None, but got Some");
        }
    }
}

impl<E> ExpectNone for Option<E> {
    fn is_none(&self) -> bool {
        self.is_none()
    }
}

pub fn get_layout<T>(size: usize) -> alloc::Layout {
    let layout = alloc::Layout::new::<T>();
    let alignment = layout.align();
    alloc::Layout::from_size_align(size, alignment).unwrap()
}

/// Alloc a Box with the given `size` and with the correct alignment for T.
/// If `size` is smaller than the size if T, this function panics.
pub fn alloc_box_aligned<T>(size: usize) -> Box<T> {
    unsafe {
        let layout = get_layout::<T>(size);
        assert!(size >= layout.size());
        let buf = alloc::alloc(layout);
        Box::from_raw(buf as *mut T)
    }
}

/// Alloc a Box with the given `size` and with the correct alignment for T.
/// Furthermore, the returned memory is initialized to zero.
/// If `size` is smaller than the size if T, this function panics.
pub fn alloc_box_aligned_zeroed<T>(size: usize) -> Box<T> {
    unsafe {
        let layout = get_layout::<T>(size);
        assert!(size >= layout.size());
        let buf = alloc::alloc_zeroed(layout);
        Box::from_raw(buf as *mut T)
    }
}

pub fn current_log_level() -> log::Level {
    if log_enabled!(log::Level::Trace) {
        log::Level::Trace
    } else if log_enabled!(log::Level::Debug) {
        log::Level::Debug
    } else if log_enabled!(log::Level::Info) {
        log::Level::Info
    } else if log_enabled!(log::Level::Warn) {
        log::Level::Warn
    } else if log_enabled!(log::Level::Error) {
        log::Level::Error
    } else {
        unreachable!();
    }
}

pub fn try_get_child_exit_reason(pid: i32) -> Option<(Option<i32>, Option<Signal>)> {
    let status: libc::c_int = 0;
    let ret = unsafe {
        let pid = pid;
        libc::waitpid(pid, status as *mut libc::c_int, libc::WNOHANG)
    };
    log::info!("waitpid={}, status={}", ret, status);
    if ret > 0 {
        // Child exited
        let mut exit_code = None;
        if libc::WIFEXITED(status) {
            exit_code = Some(libc::WEXITSTATUS(status));
        }
        let mut signal = None;
        if libc::WIFSIGNALED(status) {
            signal = Some(libc::WTERMSIG(status).try_into().unwrap());
        }
        return Some((exit_code, signal));
    }
    None
}

pub fn interruptable_sleep(duration: Duration, interrupt_signal: &Arc<AtomicBool>) -> bool {
    let second = Duration::from_secs(1);
    assert!(duration > second);

    let mut duration_left = duration;
    while duration_left > second {
        if interrupt_signal.load(Ordering::Relaxed) {
            return true;
        }
        thread::sleep(second);
        duration_left = duration_left.saturating_sub(second);
    }
    thread::sleep(duration_left);
    false
}

pub fn print_fds() {
    let fd_dir = PathBuf::from("/proc/self/fd");
    let entries = fs::read_dir(&fd_dir).expect("Unable to read /proc/self/fd");

    // Iterate over each entry
    for entry in entries {
        let entry = entry.expect("Unable to read entry");
        let fd_path = entry.path();

        // Read the symbolic link to get the file it points to
        match read_link(&fd_path) {
            Ok(target_path) => {
                println!(
                    "FD {}: {}",
                    fd_path.file_name().unwrap().to_str().unwrap(),
                    target_path.display()
                );
            }
            Err(e) => {
                eprintln!("Error reading link for {:?}: {}", fd_path, e);
            }
        }
    }
}

pub fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let res = serde_json::from_slice(&buf)?;
    Ok(res)
}

pub fn dump_json<T: serde::Serialize>(path: &Path, data: &T) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let buf = serde_json::to_vec(data)?;
    file.write_all(&buf)?;
    Ok(())
}

pub fn load_bin<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let mut file = OpenOptions::new().read(true).open(path)?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let start = Instant::now();
    let res = bincode::deserialize(&buf)?;
    let duration = start.elapsed();

    log::info!(
        "Loaded {} bytes from {:?} in {:?}",
        buf.len(),
        path,
        duration
    );

    Ok(res)
}

pub fn dump_bin<T: Serialize>(path: &Path, data: &T) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let start = Instant::now();
    let buf = bincode::serialize(data)?;
    let duration = start.elapsed();

    file.write_all(&buf)?;

    log::info!("Dumped {} bytes to {:?} in {:?}", buf.len(), path, duration);

    Ok(())
}

pub fn get_file_version(path: &PathBuf) -> anyhow::Result<DateTime<Local>> {
    let metadata = fs::metadata(path)?;
    let modified_time = metadata.modified()?;
    Ok(DateTime::<Local>::from(modified_time))
}

pub fn shuffle<T, R: Rand>(slice: &mut [T], rng: &mut R) {
    let len = slice.len();
    for i in (1..len).rev() {
        let j = rng.below_or_zero(i + 1);
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::slice;

    #[test]
    fn test_alloc_box_aligned() {
        let val: u64 = 5;
        let s = std::mem::size_of_val(&val);
        for _ in 0..256 {
            let b = alloc_box_aligned::<u64>(s);
            drop(b);
        }
    }

    #[test]
    fn test_alloc_box_aligned_zeroed() {
        let size = 4096 * 3 + 5;
        let mut e: Box<u8> = alloc_box_aligned_zeroed(size);

        let s = unsafe { slice::from_raw_parts_mut(e.as_mut() as *mut u8, size) };
        s.fill(0xff);
    }
}
