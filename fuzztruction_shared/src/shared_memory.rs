use std::{env, ffi::CString, ptr, slice};

use libafl_bolts::{
    rands::{Rand, RandomSeed, StdRand},
    AsMutSlice, AsSlice,
};
use libc::{c_int, munmap, shm_unlink};

use anyhow::{anyhow, Result};

// This is macOS's limit
// https://stackoverflow.com/questions/38049068/osx-shm-open-returns-enametoolong
// #[cfg(target_vendor = "apple")]
// const MAX_MMAP_FILENAME_LEN: usize = 31;

// #[cfg(not(target_vendor = "apple"))]
// const MAX_MMAP_FILENAME_LEN: usize = 256;

/// Mmap-based The sharedmap impl for unix using [`shm_open`] and [`mmap`].
/// Default on `MacOS` and `iOS`, where we need a central point to unmap
/// shared mem segments for dubious Mach kernel reasons.
#[derive(Clone, Debug)]
pub struct MmapShMem {
    /// The path of this shared memory segment.
    /// None in case we didn't [`shm_open`] this ourselves, but someone sent us the FD.
    path: String,
    /// The size of this map
    map_size: usize,
    /// The map ptr
    map: *mut u8,
    /// The file descriptor of the shmem
    shm_fd: c_int,
}

impl MmapShMem {
    /// Create a new [`MmapShMem`]
    pub fn new(path: String, size: usize, create: bool) -> Result<Self> {
        let fd;
        let mut size = size;

        log::trace!(
            "shm_open(name={:#?}, create={create}, size={size:x} ({size_kb}))",
            path,
            size_kb = size / 1024
        );

        unsafe {
            let mut flags = libc::O_RDWR;
            if create {
                flags |= libc::O_CREAT | libc::O_EXCL;
            }

            let c_path = CString::new(path.clone())?;

            fd = libc::shm_open(c_path.as_ptr() as *const i8, flags, 0o777);
            log::trace!("shm {path} fd: {fd}");
            if fd < 0 {
                let err = format!("Failed to open shm {:#?}", path);
                log::error!("{}", err);
                return Err(anyhow!(err));
            }

            if create {
                log::trace!("Truncating fd {fd} to {bytes} bytes", fd = fd, bytes = size);
                let ret = libc::ftruncate(fd, size as i64);
                if ret != 0 {
                    return Err(anyhow!("Failed to ftruncate shm {:#?}", path));
                }
            } else {
                let mut stat_buffer: libc::stat = std::mem::zeroed();
                let ret = libc::fstat(fd, &mut stat_buffer);
                if ret != 0 {
                    return Err(anyhow!("Failed to get shm size {:#?}", path));
                }
                size = stat_buffer.st_size as usize;
            }

            let mapping = libc::mmap(
                0 as *mut libc::c_void,
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if mapping == libc::MAP_FAILED {
                return Err(anyhow!("Failed to map shm into addresspace: {:#?}", path));
            }

            Ok(MmapShMem {
                path,
                map_size: size,
                map: mapping as *mut u8,
                shm_fd: fd,
            })
        }
    }

    pub fn write_to_env(&self, name: &str) -> Result<()> {
        env::set_var(format!("ENV_PINGU_SHM_{}_PATH", name), self.path.to_owned());
        env::set_var(
            format!("ENV_PINGU_SHM_{}_SIZE", name),
            self.map_size.to_string(),
        );

        Ok(())
    }
}

/// A [`ShMemProvider`] which uses `shmget`/`shmat`/`shmctl` to provide shared memory mappings.
#[cfg(unix)]
#[derive(Clone, Debug)]
pub struct MmapShMemProvider {
    rand: StdRand,
}

unsafe impl Send for MmapShMemProvider {}

#[cfg(unix)]
impl Default for MmapShMemProvider {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(unix)]
impl MmapShMemProvider {
    pub fn new() -> Result<Self> {
        Ok(Self {
            rand: StdRand::new(),
        })
    }

    pub fn new_shmem(&mut self, map_size: usize, label: &str) -> Result<MmapShMem> {
        let id = self.rand.next() as u32;
        let path = format!("/pingu_{label}_{id}");
        Ok(MmapShMem::new(path, map_size, true)?)
    }

    pub fn shmem_from_env(name: &str) -> Result<MmapShMem> {
        let path = env::var(format!("ENV_PINGU_SHM_{}_PATH", name))?;
        let size = env::var(format!("ENV_PINGU_SHM_{}_SIZE", name))?;

        Ok(MmapShMem::new(path, size.parse()?, false)?)
    }
}

impl MmapShMem {
    pub fn size(&self) -> usize {
        self.map_size
    }

    pub unsafe fn as_object<T: Sized + 'static>(&self) -> &T {
        assert!(self.size() >= core::mem::size_of::<T>());
        (self.as_slice().as_ptr() as *const () as *const T)
            .as_ref()
            .unwrap()
    }

    pub unsafe fn as_object_mut<T: Sized + 'static>(&mut self) -> &mut T {
        assert!(self.size() >= core::mem::size_of::<T>());
        (self.as_mut_slice().as_mut_ptr() as *mut () as *mut T)
            .as_mut()
            .unwrap()
    }
}

impl AsSlice for MmapShMem {
    type Entry = u8;
    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.map, self.map_size) }
    }
}

impl AsMutSlice for MmapShMem {
    type Entry = u8;
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.map, self.map_size) }
    }
}

impl Drop for MmapShMem {
    fn drop(&mut self) {
        unsafe {
            assert!(
                !self.map.is_null(),
                "Map should never be null for MmapShMem (on Drop)"
            );

            munmap(self.map as *mut _, self.map_size);
            self.map = ptr::null_mut();

            assert!(
                self.shm_fd != -1,
                "FD should never be -1 for MmapShMem (on Drop)"
            );

            let c_path = CString::new(self.path.clone()).unwrap();
            shm_unlink(c_path.as_ptr() as *const _);
        }
    }
}
