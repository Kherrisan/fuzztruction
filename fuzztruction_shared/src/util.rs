use anyhow::{Result, anyhow};
use chrono::{DateTime, Local};
use libafl_bolts::rands::Rand;
use log::log_enabled;
use nix::sys::signal::Signal;
use serde::Serialize;
use std::{
    alloc,
    convert::TryInto,
    fs::{self, OpenOptions, read_link},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

pub fn shell_execute(command: &str, args: Vec<&str>) -> Result<String> {
    let mut cmd = Command::new(command);
    cmd.args(args);
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(anyhow!("Failed to run command: {}", command));
    }
    Ok(String::from_utf8(output.stdout)?)
}

pub fn print_pstree() {
    let current_pid = unsafe { libc::getpid() };
    let output = shell_execute(
        "pstree",
        vec!["-p", "-a", "-h", "-l", &current_pid.to_string()],
    )
    .unwrap();
    println!("pstree: {}", output);
}

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
    let file = OpenOptions::new().read(true).open(path)?;
    let res = serde_json::from_reader(&file)?;
    Ok(res)
}

pub fn dump_json<T: serde::Serialize>(path: &Path, data: &T) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    serde_json::to_writer(&mut file, data)?;
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

pub fn check_uid_gid() {
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };
    println!("UID: {}, GID: {}", uid, gid);
    let euid = unsafe { libc::geteuid() };
    let egid = unsafe { libc::getegid() };
    println!("EUID: {}, EGID: {}", euid, egid);
}

pub fn check_rlimit_core_unlimited() {
    unsafe {
        let mut rlim: libc::rlimit = std::mem::zeroed();
        let get_ret = libc::getrlimit(libc::RLIMIT_CORE, &mut rlim as *mut libc::rlimit);
        assert_eq!(get_ret, 0, "Failed to get core limit");
        println!("Core limit: cur={}, max={}", rlim.rlim_cur, rlim.rlim_max);
        assert_eq!(rlim.rlim_cur, libc::RLIM_INFINITY);
        assert_eq!(rlim.rlim_max, libc::RLIM_INFINITY);
    }
}

pub fn set_rlimit_core_unlimited() -> Result<()> {
    unsafe {
        let mut rlim: libc::rlimit = std::mem::zeroed();
        let get_ret = libc::getrlimit(libc::RLIMIT_CORE, &mut rlim as *mut libc::rlimit);
        if get_ret != 0 {
            return Err(anyhow!("Failed to get core limit"));
        }
        println!(
            "Core limit before setting: cur={}, max={}",
            rlim.rlim_cur, rlim.rlim_max
        );

        let mut rlim: libc::rlimit = std::mem::zeroed();
        rlim.rlim_cur = libc::RLIM_INFINITY;
        rlim.rlim_max = libc::RLIM_INFINITY;
        let set_ret = libc::setrlimit(libc::RLIMIT_CORE, &rlim as *const libc::rlimit);
        if set_ret != 0 {
            return Err(anyhow!("Failed to set core limit"));
        }

        // recheck the core limit
        let mut rlim: libc::rlimit = std::mem::zeroed();
        let get_ret = libc::getrlimit(libc::RLIMIT_CORE, &mut rlim as *mut libc::rlimit);
        if get_ret != 0 {
            return Err(anyhow!("Failed to get core limit"));
        }
        if rlim.rlim_cur != libc::RLIM_INFINITY {
            return Err(anyhow!(
                "After setting core limit to unlimited, it is still not unlimited"
            ));
        } else {
            log::info!("Core rlim_cur after setting: {}", rlim.rlim_cur);
        }
        if rlim.rlim_max != libc::RLIM_INFINITY {
            return Err(anyhow!(
                "After setting core limit to unlimited, it is still not unlimited"
            ));
        } else {
            log::info!("Core rlim_max after setting: {}", rlim.rlim_max);
        }
    }
    Ok(())
}

pub fn check_core_dumpable() -> Result<()> {
    unsafe {
        let ret = libc::prctl(libc::PR_GET_DUMPABLE);
        if ret != 1 {
            return Err(anyhow!("Core dumpable is not enabled"));
        }
        log::info!("Core dumpable: {}", ret);
        Ok(())
    }
}

pub fn set_core_dumpable() -> Result<()> {
    unsafe {
        let ret = libc::prctl(libc::PR_SET_DUMPABLE, 1);
        if ret != 0 {
            return Err(anyhow!("Failed to set core dumpable"));
        }
        let ret = libc::prctl(libc::PR_GET_DUMPABLE);
        if ret != 1 {
            return Err(anyhow!(
                "After setting core dumpable, it is still not dumpable"
            ));
        }
        log::info!("Core dumpable: {}", ret);
        Ok(())
    }
}

// 在 util.rs 中添加
pub fn print_detailed_memory_stats() {
    let pid = unsafe { libc::getpid() };
    let status_path = format!("/proc/{}/status", pid);
    
    if let Ok(content) = std::fs::read_to_string(&status_path) {
        println!("=== Detailed Memory Statistics ===");
        
        let keys = [
            "VmSize",   // 虚拟内存大小
            "VmRSS",    // 实际物理内存
            "VmData",   // 数据段
            "VmStk",    // 栈
            "VmExe",    // 可执行代码
            "VmLib",    // 共享库
            "VmPTE",    // 页表
            "VmSwap",   // 交换空间
            "RssAnon",  // 匿名页面（堆、栈）
            "RssFile",  // 文件映射页面
            "RssShmem", // 共享内存页面
        ];
        
        for line in content.lines() {
            for key in &keys {
                if line.starts_with(key) {
                    println!("{}", line);
                    break;
                }
            }
        }
        
        println!("===================================");
    }
    
    // 计算页面数量
    if let Ok(content) = std::fs::read_to_string(&status_path) {
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        let pages = kb / 4;  // 假设 4KB 页面
                        println!("Approximate page count: {} pages", pages);
                    }
                }
            }
        }
    }
}

pub fn print_memory_maps_count() {
    let pid = unsafe { libc::getpid() };
    let maps_path = format!("/proc/{}/maps", pid);
    
    if let Ok(content) = std::fs::read_to_string(&maps_path) {
        let count = content.lines().count();
        println!("Memory mappings count: {}", count);
        
        // 按类型分组统计
        let mut heap_count = 0;
        let mut anon_count = 0;
        let mut file_count = 0;
        
        for line in content.lines() {
            if line.contains("[heap]") {
                heap_count += 1;
            } else if line.contains("[anon") {
                anon_count += 1;
            } else if !line.contains("[") {
                file_count += 1;
            }
        }
        
        println!("  - Heap mappings: {}", heap_count);
        println!("  - Anonymous mappings: {}", anon_count);
        println!("  - File mappings: {}", file_count);
    }
}

pub fn print_process_memory_usage() {
    use std::fs;
    use std::io::Read;
    
    let pid = unsafe { libc::getpid() };
    let status_path = format!("/proc/{}/status", pid);
    
    // 读取 /proc/pid/status 文件
    let mut content = String::new();
    if let Ok(mut file) = fs::File::open(&status_path) {
        if file.read_to_string(&mut content).is_err() {
            eprintln!("Failed to read {}", status_path);
            return;
        }
    } else {
        eprintln!("Failed to open {}", status_path);
        return;
    }
    
    println!("=== Process Memory Usage (PID: {}) ===", pid);
    
    // 查找并打印相关的内存信息
    let keys = ["VmSize", "VmRSS", "VmPTE", "VmData", "VmStk", "VmExe", "VmLib"];
    
    for line in content.lines() {
        for key in &keys {
            if line.starts_with(key) {
                // 解析行格式: "VmSize:    123456 kB"
                if let Some(parts) = line.split_once(':') {
                    let value = parts.1.trim();
                    println!("  {:12} {}", format!("{}:", parts.0), value);
                }
                break;
            }
        }
    }
    
    println!("=====================================");
}

/**
 * Read random data from /dev/random
 */
pub fn read_random_dev(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);

    let mut file = std::fs::File::open("/dev/random").expect("Failed to open /dev/random");
    file.read(&mut data)
        .expect("Failed to read from /dev/random");

    data
}

/// 宏：测量代码块执行时间并累加到指定变量中
/// 用法：measure_sum!( 累加变量, { 代码块 } )
#[macro_export]
macro_rules! measure_total_time {
    ($acc:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block; // 执行代码块并获取返回值
        let elapsed = start.elapsed();
        $acc += chrono::Duration::from_std(elapsed).unwrap_or_default(); // 累加时间
        result // 返回结果，保证逻辑连贯性
    }};
}

#[cfg(test)]
mod test {
    use super::*;
    use std::slice;

    #[test]
    fn test_print_process_memory_usage() {
        print_process_memory_usage();
    }

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

    #[test]
    fn test_process_tree() {
        print_pstree();
    }
}
