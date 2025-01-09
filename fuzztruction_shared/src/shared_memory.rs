use std::{env, ffi::CString, ptr, slice};

use libafl_bolts::{
    rands::{Rand, RandomSeed, StdRand},
    AsMutSlice, AsSlice,
};
use libc::{c_int, munmap, shm_unlink};

use anyhow::{anyhow, Context, Result};
use rand::Rng;

/// Mmap-based The sharedmap impl for unix using [`shm_open`] and [`mmap`].
/// Default on `MacOS` and `iOS`, where we need a central point to unmap
/// shared mem segments for dubious Mach kernel reasons.
///
/// While in libafl_bolts, MmapShMem id is derived from shm_fd,
/// MmapShMem.write_to_env will write shm_fd to env, sharing with the child process.
/// However, the child process will not be able to open through only the shm_fd.
/// Because the shm_open, which is used to create and open the shm file, is setted with
/// FD_CLOEXEC, which means the file descriptor will be closed when the process is replaced.
/// So if the child process wants to open the shm file, it should shm_open the path of the shm file again.
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

unsafe impl Send for MmapShMem {}

impl MmapShMem {
    /// Create a new [`MmapShMem`]
    pub fn new(path: String, size: usize, create: bool) -> Result<Self> {
        let fd;
        let mut size = size;

        log::debug!(
            "shm_ (name={:#?}, create={create}, size={size:x} ({size_kb}))",
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
                let err = format!("Failed to open shm file {:#?}", path);
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
        env::set_var(format!("PINGU_SHM_{}_PATH", name), self.path.to_owned());
        env::set_var(
            format!("PINGU_SHM_{}_SIZE", name),
            self.map_size.to_string(),
        );

        log::debug!(
            "shm PINGU_SHM_{name}_PATH:{path} PINGU_SHM_{name}_SIZE:{size} fd: {fd}",
            name = name,
            path = self.path,
            size = self.map_size,
            fd = self.shm_fd
        );

        Ok(())
    }

    pub fn new_shmem(map_size: usize, label: &str) -> Result<MmapShMem> {
        let rnd_suffix: u32 = rand::thread_rng().gen();
        let path = format!("/pingu_{label}_{rnd_suffix}");
        Ok(MmapShMem::new(path, map_size, true)?)
    }

    pub fn shmem_from_env(name: &str) -> Result<MmapShMem> {
        let path = env::var(format!("PINGU_SHM_{}_PATH", name))
            .context(format!("PINGU_SHM_{}_PATH not found", name))?;
        let size = env::var(format!("PINGU_SHM_{}_SIZE", name))
            .context(format!("PINGU_SHM_{}_SIZE", name))?;

        Ok(MmapShMem::new(
            path.clone(),
            size.parse().context(format!(
                "Error in parsing PINGU_SHM_{}_SIZE: {}",
                name, size
            ))?,
            false,
        )
        .context(format!("When getting shm {} from path: {}", name, path))?)
    }
}

impl MmapShMem {
    pub fn map(&self) -> *mut u8 {
        self.map
    }

    pub fn path(&self) -> &str {
        &self.path
    }

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
