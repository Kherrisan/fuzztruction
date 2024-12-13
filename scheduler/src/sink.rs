use anyhow::{anyhow, Context, Result};

use byte_unit::n_mib_bytes;
use fuzztruction_shared::aux_messages::AuxStreamMessage;
use fuzztruction_shared::aux_messages::AuxStreamType;
use fuzztruction_shared::aux_stream::AuxStreamAssembler;
use fuzztruction_shared::constants::ENV_LOG_LEVEL;
use fuzztruction_shared::messages::{Message, MessageType, MsgHeader};
use fuzztruction_shared::shared_memory::MmapShMem;
use fuzztruction_shared::util::current_log_level;
use fuzztruction_shared::util::try_get_child_exit_reason;
use libafl::executors::ExitKind;
use log::error;
use pingu_generator::agent::HANDSHAKE_TIMEOUT;
use pingu_generator::messages::HelloMessage;
use posixmq::PosixMq;
use proc_maps::MapRange;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

use std::env;
use std::env::set_current_dir;
use std::os::unix::prelude::AsRawFd;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;
use std::{
    convert::TryFrom,
    ffi::CString,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    ops::*,
    path::PathBuf,
};
use std::{fs, io};
use thiserror::Error;

use libc::SIGKILL;
use mktemp::Temp;
use nix::sys::signal::Signal;

use crate::config::Config;
use crate::io_channels::InputChannel;
use crate::sink_bitmap::{Bitmap, BITMAP_DEFAULT_MAP_SIZE};
use crate::source::Source;

use filedescriptor;

const FS_OPT_MAPSIZE: u32 = 0x40000000;

// FDs used by the forkserver to communicate with us.
// Hardcoded in AFLs config.h.
const FORKSRV_FD: i32 = 198;
const AFL_READ_FROM_PARENT_FD: i32 = FORKSRV_FD;
const AFL_WRITE_TO_PARENT_FD: i32 = FORKSRV_FD + 1;

const AFL_SHM_ENV_VAR_NAME: &str = "__AFL_SHM_ID";
#[cfg(not(feature = "never-timeout"))]
const AFL_DEFAULT_TIMEOUT: Duration = Duration::from_millis(100);
#[cfg(feature = "never-timeout")]
const AFL_DEFAULT_TIMEOUT: Duration = Duration::from_secs(3600 * 24);
#[cfg(not(feature = "never-timeout"))]
const DEFAULT_SINK_RECEIVE_TIMEOUT: Duration = Duration::from_secs(1);
#[cfg(feature = "never-timeout")]
const DEFAULT_SINK_RECEIVE_TIMEOUT: Duration = Duration::from_secs(3600 * 24);

fn repeat_on_interrupt<F, R>(f: F) -> R
where
    F: Fn() -> R,
    R: TryInto<libc::c_int> + Clone,
{
    loop {
        let ret = f();
        if ret.clone().try_into().unwrap_or(0) != -libc::EINTR {
            return ret;
        } else {
            log::trace!("Repeating call because of EINTR");
        }
    }
}

/// Type used to represent error conditions of the source.
#[derive(Error, Debug)]
pub enum SinkError {
    #[error("The workdir '{0}' already exists.")]
    WorkdirExists(String),
    #[error("Fatal error occurred: {0}")]
    FatalError(String),
    #[error("Exceeded timeout while waiting for data: {0}")]
    CommunicationTimeoutError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RunResult {
    Terminated(i32),
    Signalled(Signal),
    TimedOut,
}

impl Into<ExitKind> for RunResult {
    fn into(self) -> ExitKind {
        match self {
            RunResult::Terminated(_) => ExitKind::Ok,
            RunResult::Signalled(_) => ExitKind::Crash,
            RunResult::TimedOut => ExitKind::Timeout,
        }
    }
}

unsafe impl Send for Sink {}

#[derive(Debug)]
pub struct Sink {
    /// That file system path to the target binary.
    path: PathBuf,
    /// The arguments passed to the binary.
    args: Vec<String>,
    /// Workdir
    #[allow(unused)]
    workdir: PathBuf,
    state_dir: PathBuf,
    /// Description of how the target binary consumes fuzzing input.
    input_channel: InputChannel,
    /// The file that is used to pass input to the target.
    input_file: (File, PathBuf),
    /// The session id of the forkserver we are communicating with.
    forkserver_pid: Option<i32>,
    /// The bitmap used to compute coverage.
    afl_map_shm: Option<MmapShMem>,
    // /// The fd used to send data to the forkserver.
    // send_fd: Option<i32>,
    // /// Non blocking fd used to receive data from the forkserver.
    // receive_fd: Option<i32>,
    // The mq used to receive message from the sink forkserver agent.
    mq_recv: Option<PosixMq>,
    mq_send: Option<PosixMq>,
    mq_recv_name: String,
    mq_send_name: String,
    #[allow(unused)]
    stdout_file: Option<(File, PathBuf)>,
    #[allow(unused)]
    stderr_file: Option<(File, PathBuf)>,
    // The pid of the sink forked child process.
    pub child_pid: Option<i32>,
    mem_file: Option<File>,
    /// The memory mappings of the target application.
    /// Available after calling `.start()`.
    mappings: Option<Vec<MapRange>>,
    stop_signal: Arc<RwLock<bool>>,
    /// Whether to log the output written to stdout. If false, the output is discarded.
    log_stdout: bool,
    /// Whether to log the output written to stderr. If false, the output is discarded.
    log_stderr: bool,
    pub aux_stream_assembler: AuxStreamAssembler,
    config: Option<Config>,
    bitmap_was_resize: bool,
    workdir_file_whitelist: Vec<PathBuf>,
}

impl Sink {
    pub fn new(
        path: PathBuf,
        mut args: Vec<String>,
        mut workdir: PathBuf,
        input_channel: InputChannel,
        config: Option<&Config>,
        log_stdout: bool,
        log_stderr: bool,
    ) -> Result<Sink> {
        let mut workdir_file_whitelist = vec![];

        workdir.push("sink");

        let mut state_dir = workdir.clone();
        state_dir.push("state");
        log::debug!("Creating state dir {:?}", &state_dir);
        fs::create_dir_all(&state_dir)?;

        workdir.push("workdir");
        // Create the file into we write inputdata before execution.
        log::debug!("Creating workdir {:?}", &workdir);
        fs::create_dir_all(&workdir)?;

        let tmpfile_path = Temp::new_file_in(&workdir).unwrap().to_path_buf();
        let mut input_file_path = String::from(tmpfile_path.to_str().unwrap());
        input_file_path.push_str("_input");
        let input_file_path = PathBuf::from(input_file_path);

        let input_file = match input_channel {
            InputChannel::File => {
                if let Some(elem) = args.iter_mut().find(|e| **e == "@@") {
                    *elem = input_file_path.to_str().unwrap().to_owned();
                } else {
                    return Err(anyhow!(format!("No @@ marker in args, even though the input channel is defined as file. args: {:#?}", args)));
                }
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&input_file_path)?
            }
            _ => OpenOptions::new().read(true).open("/dev/null")?,
        };

        let mut rand_suffix: String = thread_rng()
            .sample_iter(&Alphanumeric)
            .take(6)
            .map(char::from)
            .collect();
        rand_suffix += &format!("_tid_{}", unsafe { libc::gettid().to_string() });

        let mq_recv_name: String = "/mq_sink_recv_".to_owned() + &rand_suffix;
        let mq_send_name: String = "/mq_sink_send_".to_owned() + &rand_suffix;

        let mut stdout_file = None;
        if log_stdout {
            // Setup file for stdout logging.
            let mut path = workdir.clone();
            path.push("stdout");
            workdir_file_whitelist.push(path.to_owned());
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .unwrap();
            stdout_file = Some((file, path));
        }

        let mut stderr_file = None;
        if log_stderr {
            // Setup file for stdout logging.
            let mut path = workdir.clone();
            path.push("stderr");
            workdir_file_whitelist.push(path.to_owned());
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .unwrap();
            stderr_file = Some((file, path));
        }

        Ok(Sink {
            path,
            args,
            workdir,
            state_dir,
            input_channel,
            input_file: (input_file, input_file_path),
            forkserver_pid: None,
            afl_map_shm: None,
            mq_recv: None,
            mq_send: None,
            mq_recv_name,
            mq_send_name,
            log_stdout,
            log_stderr,
            child_pid: None,
            stop_signal: Arc::new(RwLock::new(false)),
            stdout_file,
            stderr_file,
            aux_stream_assembler: AuxStreamAssembler::new(),
            config: config.cloned(),
            bitmap_was_resize: false,
            workdir_file_whitelist,
            mem_file: None,
            mappings: None,
        })
    }

    pub fn build(config: &Config, id: Option<usize>) -> Result<Sink> {
        let sink = Sink::from_config(config, id)?;

        Ok(sink)
    }

    pub fn from_config(config: &Config, id: Option<usize>) -> Result<Sink> {
        let config_new = config.clone();
        let mut workdir = config_new.general.work_dir.clone();
        workdir.push(
            id.map(|id| id.to_string())
                .unwrap_or_else(|| "0".to_owned()),
        );

        let sink = Sink::new(
            config_new.sink.bin_path,
            config_new.sink.arguments,
            workdir,
            config_new.sink.input_type,
            Some(config),
            config.sink.log_stdout,
            config.sink.log_stderr,
        )?;
        Ok(sink)
    }

    pub fn check_link_time_deps(path: &Path, config: Option<&Config>) -> Result<()> {
        // Check if path points to an executable and whether it is linked against our runtime agent.
        // FIXME: Handle static binaries?
        let mut cmd = Command::new("ldd");
        cmd.args([&path]);

        if let Some(config) = config {
            // Apply environment variables such as LD_LIBRARY_PATH
            for (key, val) in &config.source.env {
                cmd.env(key, val);
            }
        }

        let output = cmd
            .output()
            .unwrap_or_else(|_| panic!("Failed to call ldd on {:#?}", path))
            .stdout;
        let output = String::from_utf8(output).expect("Failed to convert stdout to UTF8.");

        if output.contains("libpingu_generator.so => not found") {
            Err(SinkError::FatalError(
                "Target failed to find some libraries/library!".to_owned(),
            ))
            .context(output)?;
        }
        Ok(())
    }

    /// Wait for the given duration for the forkserver read fd to become ready.
    /// Returns Ok(true) if data becomes ready during the given `timeout`, else
    /// Ok(false).
    ///
    /// # Error
    ///
    /// Returns an Error if an unexpected error occurs.
    // fn wait_for_data(&self, timeout: Duration) -> Result<()> {
    //     let pollfd = filedescriptor::pollfd {
    //         fd: self.receive_fd.unwrap(),
    //         events: filedescriptor::POLLIN,
    //         revents: 0,
    //     };
    //     let mut pollfds = [pollfd];

    //     let nready = filedescriptor::poll(&mut pollfds, Some(timeout));
    //     match nready {
    //         Ok(1) => Ok(()),
    //         Ok(0) => Err(SinkError::CommunicationTimeoutError(format!(
    //             "Did not received data after {:?}",
    //             timeout
    //         ))
    //         .into()),
    //         Ok(n) => {
    //             unreachable!("Unexpected return value: {}", n);
    //         }
    //         Err(ref err) => {
    //             if let filedescriptor::Error::Poll(err) = err {
    //                 if err.kind() == io::ErrorKind::Interrupted {
    //                     return self.wait_for_data(timeout);
    //                 }
    //             }
    //             Err(SinkError::FatalError(format!("Failed to poll fd: {:#?}", err)).into())
    //         }
    //     }
    // }

    /// Wait for `timeout` long for a message of type T.
    /// Messages of different type received in between are ignored.
    pub fn wait_for_message<T: Message>(&mut self, timeout: Duration) -> Result<T> {
        let mq_recv = self.mq_recv.as_ref().unwrap();
        let mut buf: Vec<u8> = vec![0; mq_recv.attributes().max_msg_len];

        log::trace!("Waiting for message of type {:?}", T::message_type());

        let start_ts = Instant::now();
        let mut timeout_left = timeout;
        loop {
            self.receive_message(timeout_left, &mut buf)
                .context(format!(
                    "Failed to receive message of type {:?}",
                    T::message_type()
                ))
                .context(format!(
                    "Error while waiting for message {:?}",
                    T::message_type()
                ))?;
            timeout_left = timeout.saturating_sub(start_ts.elapsed());

            let header = MsgHeader::try_from_bytes(&buf)?;
            if header.id == T::message_type() {
                let ret = T::try_from_bytes(&buf)?;
                return Ok(ret);
            } else {
                log::warn!(
                    "Skipping message of type {:?} while waiting for {:?} message.",
                    header.id,
                    T::message_type()
                );
            }
        }
    }

    /// Receive a message from the agent. In case it is a AuxStreamMessage message, it is directly processed.
    /// If not, the function returns and places the received message into `receive_buf`.
    pub fn receive_message(&mut self, timeout: Duration, receive_buf: &mut [u8]) -> Result<()> {
        loop {
            self.mq_recv
                .as_ref()
                .unwrap()
                .receive_timeout(receive_buf, timeout)?;
            let header = MsgHeader::try_from_bytes(receive_buf)?;
            if header.id == MessageType::AuxStreamMessage {
                Source::process_aux_message(
                    &mut self.aux_stream_assembler,
                    AuxStreamMessage::try_from_bytes(receive_buf)?,
                );
                continue;
            }
            break;
        }
        Ok(())
    }

    #[allow(unreachable_code)]
    pub fn start(&mut self) -> Result<()> {
        log::debug!("Starting sink");

        log::debug!("Creating POSIX queues");
        log::debug!("Creating MQ recv: {}", self.mq_recv_name);
        let mq = posixmq::OpenOptions::readwrite()
            .create_new()
            .open(&self.mq_recv_name)
            .context("Failed to create sink recv MQ")?;
        self.mq_recv = Some(mq);

        log::debug!("Creating MQ send: {}", self.mq_send_name);
        let mq = posixmq::OpenOptions::readwrite()
            .create_new()
            .open(&self.mq_send_name)
            .context("Failed to create sink send MQ")?;
        self.mq_send = Some(mq);

        // // send_pipe[1](we) -> send_pipe[0](forkserver).
        // let send_pipe = [0i32; 2];
        // // receive_pipe[1](forkserver) -> receive_pipe[0](we).
        // let receive_pipe = [0i32; 2];

        // // Create pipe for communicating with the forkserver.
        // unsafe {
        //     let ret = libc::pipe(send_pipe.as_ptr() as *mut i32);
        //     assert_eq!(ret, 0);
        //     let ret = libc::pipe(receive_pipe.as_ptr() as *mut i32);
        //     assert_eq!(ret, 0);
        // }

        // self.send_fd = Some(send_pipe[1]);
        // let child_receive_fd = send_pipe[0];

        // self.receive_fd = Some(receive_pipe[0]);
        // let child_send_fd = receive_pipe[1];

        log::debug!("Forking sink child");
        let forkserver_pid = unsafe { libc::fork() };
        match forkserver_pid {
            -1 => return Err(anyhow!("Fork failed!")),
            0 => {
                /*
                Child
                Be aware that we are forking a potentially multithreaded application
                here. Since fork() only copies the calling thread, the environment
                might be left in a dirty state because of, e.g., mutexs that where
                locked at the time fork was called.
                Because of this it is only save to call async-signal-safe functions
                (https://man7.org/linux/man-pages/man7/signal-safety.7.html).
                Note that loggin function (debug!...) often internally use mutexes
                to lock the output buffer, thus using logging here is forbidden
                and likely causes deadlocks.
                */
                set_current_dir(&self.config.as_ref().unwrap().sink.cwd)
                    .expect("Failed to set workdir");

                unsafe {
                    let ret = libc::setsid();
                    assert!(ret >= 0);
                }

                // Setup args
                let path =
                    self.path.to_str().map(|s| s.to_owned()).ok_or_else(|| {
                        SinkError::Other(anyhow!("Invalid UTF-8 character in path"))
                    })?;
                let mut args = self.args.clone();
                args.insert(0, path.clone());

                let argv: Vec<CString> = args
                    .iter()
                    .map(|arg| CString::new(arg.as_bytes()).unwrap())
                    .collect();

                // Setup environment
                let mut envp: Vec<CString> = Vec::new();

                let env_mq_recv = CString::new(
                    format!("FT_MQ_SINK_SEND={}", self.mq_recv_name.as_str()).as_bytes(),
                )
                .expect("Failed to format FT_MQ_SEND");
                envp.push(env_mq_recv);

                let env_mq_send = CString::new(
                    format!("FT_MQ_SINK_RECV={}", self.mq_send_name.as_str()).as_bytes(),
                )
                .expect("Failed to format FT_MQ_RECV");
                envp.push(env_mq_send);

                let env_log_level =
                    CString::new(format!("{}={}", ENV_LOG_LEVEL, current_log_level()).as_bytes())
                        .expect("Failed to format FT_LOG_LEVEL");
                envp.push(env_log_level);

                let mut env_from_config = Vec::new();
                if let Some(cfg) = self.config.as_ref() {
                    cfg.sink.env.iter().for_each(|var| {
                        env_from_config
                            .push(CString::new(format!("{}={}", var.0, var.1).as_bytes()).unwrap())
                    })
                }
                env_from_config.iter().for_each(|e| {
                    envp.push(e.to_owned());
                });

                // Resolve symbols at the start, thus we do not have to do it
                // after each fork.
                let ld_bind_now = CString::new("LD_BIND_NOW=1".as_bytes())
                    .expect("Failed to create LD_BIND_NOW string");
                envp.push(ld_bind_now);

                let dev_null_fd = unsafe {
                    let path = CString::new("/dev/null".as_bytes()).unwrap();
                    libc::open(path.as_ptr(), libc::O_RDONLY)
                };
                if dev_null_fd < 0 {
                    panic!("Failed to open /dev/null");
                }

                log::debug!("Redirecting stdout to {:?}", self.stdout_file);
                if self.log_stdout {
                    unsafe {
                        let fd = self.stdout_file.as_ref().unwrap().0.as_raw_fd();
                        libc::dup2(fd, libc::STDOUT_FILENO);
                        libc::close(fd);
                    }
                } else {
                    unsafe {
                        libc::dup2(dev_null_fd, libc::STDOUT_FILENO);
                    }
                }

                log::debug!("Redirecting stderr to {:?}", self.stderr_file);
                if self.log_stderr {
                    unsafe {
                        let fd = self.stderr_file.as_ref().unwrap().0.as_raw_fd();
                        libc::dup2(fd, libc::STDERR_FILENO);
                        libc::close(fd);
                    }
                } else {
                    unsafe {
                        libc::dup2(dev_null_fd, libc::STDERR_FILENO);
                    }
                }

                unsafe {
                    libc::close(dev_null_fd);
                }

                // unsafe {
                //     // Close the pipe ends used by our parent.
                //     // libc::close(self.receive_fd.unwrap());
                //     // libc::close(self.send_fd.unwrap());

                //     // Remap fds to the ones used by the forkserver.
                //     // The fds might have by chance the correct value, in this case
                //     // dup2 & close would actually cause us to close the fd we intended to pass.
                //     if child_receive_fd != AFL_READ_FROM_PARENT_FD {
                //         let ret = libc::dup2(child_receive_fd, AFL_READ_FROM_PARENT_FD);
                //         assert!(ret >= 0);
                //         libc::close(child_receive_fd);
                //     }

                //     if child_send_fd != AFL_WRITE_TO_PARENT_FD {
                //         let ret = libc::dup2(child_send_fd, AFL_WRITE_TO_PARENT_FD);
                //         assert!(ret >= 0);
                //         libc::close(child_send_fd);
                //     }
                // }

                unsafe {
                    if !self.log_stdout && !self.log_stderr {
                        // if we log stderr or stdout, the limit will cause our
                        // fuzzer to fail after some time.
                        let mut rlim: libc::rlimit = std::mem::zeroed();
                        rlim.rlim_cur = n_mib_bytes!(512).try_into().unwrap();
                        rlim.rlim_max = n_mib_bytes!(512).try_into().unwrap();
                        let ret = libc::setrlimit(libc::RLIMIT_FSIZE, &rlim as *const libc::rlimit);
                        assert_eq!(ret, 0);
                    }

                    // Limit maximum virtual memory size.
                    let mut rlim: libc::rlimit = std::mem::zeroed();
                    rlim.rlim_cur = n_gib_bytes!(8).try_into().unwrap();
                    rlim.rlim_max = n_gib_bytes!(8).try_into().unwrap();
                    let ret = libc::setrlimit(libc::RLIMIT_AS, &rlim as *const libc::rlimit);
                    assert_eq!(ret, 0);

                    // Disable core dumps
                    let limit_val: libc::rlimit = std::mem::zeroed();
                    let ret = libc::setrlimit(libc::RLIMIT_CORE, &limit_val);
                    assert_eq!(ret, 0);

                    // Max AS size.
                    // let mut rlim: libc::rlimit = std::mem::zeroed();
                    // rlim.rlim_cur = n_mib_bytes!(512).try_into().unwrap();
                    // rlim.rlim_max = n_mib_bytes!(512).try_into().unwrap();
                    // let ret = libc::setrlimit(libc::RLIMIT_AS, &rlim as *const libc::rlimit);
                    // assert_eq!(ret, 0);

                    // Disable ASLR of sink, may be useless.
                    let _ = libc::personality(libc::ADDR_NO_RANDOMIZE as u64);
                    let check_personality = libc::personality(0xffffffff);
                    if check_personality & libc::ADDR_NO_RANDOMIZE == 0 {
                        let err = std::io::Error::last_os_error();
                        let errno = err.raw_os_error().unwrap_or(0);
                        // 根据错误码进行判断
                        match errno {
                            libc::EINVAL => {
                                log::error!("Invalid argument");
                            }
                            libc::EPERM => {
                                log::error!("Operation not permitted");
                            }
                            // 其他错误处理...
                            _ => {
                                log::error!("Unknown error: {}", errno);
                            }
                        }
                        panic!("Failed to disable ASLR");
                    }
                }

                if let Err(err) = self.drop_privileges() {
                    log::error!("Failed to drop privileges: {:#?}", err);
                    panic!();
                }

                // Make sure that UID == EUID, since if this is not the case,
                // ld will ignore LD_PRELOAD which we need to use for targets
                // that normally load instrumented libraries during runtime.
                assert_eq!(nix::unistd::getuid(), nix::unistd::geteuid());
                assert_eq!(nix::unistd::getegid(), nix::unistd::getegid());

                // Copy all environment variables from current process into the new process
                env::vars()
                    .for_each(|e| envp.push(CString::new(format!("{}={}", e.0, e.1)).unwrap()));

                let prog = CString::new(path.as_bytes()).unwrap();
                nix::unistd::execve(&prog, &argv, &envp).unwrap();

                unreachable!("Failed to call execve on '{}'", path);
            }
            _ => { /* The parent */ }
        }

        // Take the file, thus their fds get dropped.
        self.stdout_file.take();
        self.stderr_file.take();

        /* The parent */
        log::info!("Forkserver has pid {}", forkserver_pid);

        // Note th sid, thus we can kill the child later.
        // This is a sid since the child calls setsid().
        self.forkserver_pid = Some(forkserver_pid);

        // Dump some info to state dir
        self.dump_state_to_disk(forkserver_pid)?;

        // // Close the pipe ends used by the child.
        // unsafe {
        //     libc::close(child_receive_fd);
        //     libc::close(child_send_fd);
        // }

        // unsafe {
        //     libc::fcntl(self.send_fd.unwrap(), libc::F_SETFD, libc::FD_CLOEXEC);
        //     libc::fcntl(self.receive_fd.unwrap(), libc::F_SETFD, libc::FD_CLOEXEC);
        // }

        // Wait for the handshake response.
        log::debug!("Waiting for handshake message.");
        let msg = self
            .wait_for_message::<HelloMessage>(HANDSHAKE_TIMEOUT)
            .context("Timeout while waiting for forkserver to come up.")?;
        log::debug!("Got HelloMessage. Agents TID is {:?}", msg.senders_tid);

        // Read the available data.
        // let buffer = [0u8; 4];
        // unsafe {
        //     let ret = libc::read(
        //         self.receive_fd.unwrap(),
        //         buffer.as_ptr() as *mut libc::c_void,
        //         4,
        //     );
        //     if ret != 4 {
        //         return Err(anyhow!(format!(
        //             "Failed to do handshake with forkserver. ret={}",
        //             ret
        //         )));
        //     }
        // }

        // if self.stdout_file.is_some() {
        //     // Take the the stdout file thus its fd gets dropped.
        //     self.stdout_file.take();
        // }
        // if self.stderr_file.is_some() {
        //     // Take the the stderr file thus its fd gets dropped.
        //     self.stderr_file.take();
        // }

        self.mem_file = Some(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(format!("/proc/{}/mem", forkserver_pid))
                .context("Failed to open /proc/<x>/mem")?,
        );

        // Get the mapping of the target memory space.
        // NOTE: This might not include lazily loaded libraries.
        let mut mappings =
            get_process_maps(self.forkserver_pid.unwrap()).context("Failed to get process maps")?;
        for map in mappings.iter_mut() {
            let pathname = map
                .pathname
                .as_ref()
                .map(|e| self.resolve_path_from_child(&e));
            map.pathname = pathname.map(|v| v.to_str().unwrap().to_owned());
        }
        self.mappings = Some(mappings);

        // Save mapping in state_dir for later use.
        let mut path = self.state_dir.clone();
        path.push("source_maps");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.write_all(format!("{:#?}", &self.mappings).as_bytes())
            .unwrap();

        self.start_message_loop()?;

        // We are ready to fuzz!
        Ok(())
    }

    /// Dump some information to the sink state directory.
    fn dump_state_to_disk(&self, forkserver_pid: i32) -> Result<()> {
        let mut path = self.state_dir.clone();

        path.push("forkserver_pid");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.write_all(format!("{}", forkserver_pid).as_bytes())?;

        let own_pid = unsafe { libc::getpid() };
        let mut path = self.state_dir.clone();
        path.push("own_pid");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.write_all(format!("{}", own_pid).as_bytes())?;

        let own_tid = unsafe { libc::gettid() };
        let mut path = self.state_dir.clone();
        path.push("own_tid");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.write_all(format!("{}", own_tid).as_bytes())?;

        Ok(())
    }

    fn start_message_loop(&mut self) -> Result<()> {
        let stop_signal = self.stop_signal.clone();
        let mq_recv = self.mq_recv.take().unwrap();
        let _: JoinHandle<Result<()>> = thread::spawn(move || {
            let mut assembler = AuxStreamAssembler::new();
            let mut buf: Vec<u8> = vec![0; mq_recv.attributes().max_msg_len];
            loop {
                if let Ok(timeout) = stop_signal.read() {
                    if *timeout {
                        break;
                    }
                }
                match mq_recv.receive_timeout(&mut buf, DEFAULT_SINK_RECEIVE_TIMEOUT) {
                    Ok(_) => {
                        log::trace!("Received message from the sink");
                        let header = MsgHeader::try_from_bytes(&buf)?;
                        if header.id != MessageType::AuxStreamMessage {
                            continue;
                        }
                        let msg = AuxStreamMessage::try_from_bytes(&buf)?;
                        match assembler.process_str_msg(msg) {
                            Ok(Some((ty, s))) => match ty {
                                AuxStreamType::LogRecord => {
                                    Source::process_log_record_message(s);
                                }
                                _ => log::error!("Received message on unsupported channel."),
                            },
                            Ok(None) => {
                                log::trace!("Received incomplete message, continue to receive");
                            }
                            Err(err) => log::error!("Error while decoding aux stream: {}", err),
                        }
                    }
                    Err(e) => match e.kind() {
                        io::ErrorKind::TimedOut => {
                            log::trace!("Sink message loop timed out");
                            continue;
                        }
                        _ => {
                            log::error!("{}", e.to_string());
                        }
                    },
                }
            }
            Ok(())
        });

        Ok(())
    }

    fn drop_privileges(&mut self) -> Result<()> {
        let uid_gid = self
            .config
            .as_ref()
            .map(|config| config.general.jail_uid_gid())
            .unwrap_or(None);
        if uid_gid.is_some() {
            jail::acquire_privileges()?;
        }
        if let Some((uid, gid)) = uid_gid {
            jail::drop_privileges(uid, gid, true)?;
        }
        Ok(())
    }

    /// Stops the forksever. Must be called before calling start() again.
    /// It is save to call this function multiple times.
    pub fn stop(&mut self) {
        if let Ok(mut timeout) = self.stop_signal.write() {
            *timeout = true;
        }
        if let Some(sid) = self.forkserver_pid.take() {
            unsafe {
                libc::close(self.send_fd.unwrap());
                libc::close(self.receive_fd.unwrap());

                let ret = libc::killpg(sid, SIGKILL);
                assert!(ret == 0);
                // reap it
                libc::waitpid(sid, std::ptr::null_mut() as *mut libc::c_int, 0);
            }
        }
    }

    /// Write the given bytes into the sinks input channel. This function
    /// is only allowed to be called on sinks with InputChannel::Stdin or InputChannel::File
    /// input channel.
    pub fn write(&mut self, data: &[u8]) {
        debug_assert!(
            self.input_channel == InputChannel::Stdin || self.input_channel == InputChannel::File
        );

        self.input_file.0.seek(SeekFrom::Start(0)).unwrap();
        self.input_file.0.set_len(0).unwrap();
        self.input_file.0.write_all(data).unwrap();
        self.input_file.0.seek(SeekFrom::Start(0)).unwrap();
        self.input_file.0.sync_all().unwrap();
    }

    pub fn run(&mut self, _timeout: Duration) -> Result<RunResult> {
        panic!("run() not implemented!");
    }

    pub fn fork(&mut self) -> Result<()> {
        self.purge_workdir();

        let buffer = [0u8; 4];
        let buf_ptr = buffer.as_ptr() as *mut libc::c_void;

        // Tell the forkserver to fork.
        log::trace!("Requesting fork");
        let ret = repeat_on_interrupt(|| unsafe { libc::write(self.send_fd.unwrap(), buf_ptr, 4) });
        if ret != 4 {
            error!("Fork request failed");
            return Err(anyhow!("Failed to write to send_fd: {}", ret));
        }

        log::trace!("Waiting for child pid");
        self.wait_for_data(AFL_DEFAULT_TIMEOUT)
            .context("Failed to retrive child pid from forkserver")?;
        let ret =
            repeat_on_interrupt(|| unsafe { libc::read(self.receive_fd.unwrap(), buf_ptr, 4) });
        if ret != 4 {
            error!("Failed to retrive child pid");
            return Err(anyhow!("Failed to read from receive_non_blocking_fd"));
        }

        let child_pid = i32::from_le_bytes(buffer);
        if child_pid <= 0 {
            log::error!("Child pid '{}' is invalid", child_pid);
            return Err(anyhow!(
                "Failed to parse child_pid. child_pid={}, bytes={:?}",
                child_pid,
                buffer
            ));
        }

        log::trace!("Got child pid {}", child_pid);
        self.child_pid = Some(child_pid);

        let mut path = self.state_dir.clone();
        path.push("child_pid");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.write_all(format!("{}", child_pid).as_bytes())?;

        Ok(())
    }

    pub fn terminate(&mut self) -> Result<RunResult> {
        let buffer = [0u8; 4];
        let buf_ptr = buffer.as_ptr() as *mut libc::c_void;

        match self.wait_for_data(Duration::ZERO) {
            Ok(_) => {
                // The child is already terminated.
                // But this may barely happened, unless the SUT is signaled.
                log::trace!("Sink child already terminated or signaled");
            }
            Err(_) => {
                // The child is still running
                log::trace!("Sink child is still running, so kill it manully");
                let kill_ret = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(self.child_pid.unwrap()),
                    nix::sys::signal::SIGTERM,
                );
                if let Err(ref err) = kill_ret {
                    // This might just be caused by the fact that the child won the race
                    // and terminated before we killed it.
                    log::trace!("Failed to kill child: {:#?}", err);
                }
                // Read the exit status of the child, after killing
                if let Err(err) = self
                    .wait_for_data(AFL_DEFAULT_TIMEOUT)
                    .context("Child did not acknowledge termination request")
                {
                    // The forkserver is not responding
                    let reason = try_get_child_exit_reason(self.forkserver_pid.unwrap());
                    log::error!(
                        "Exit reason: {:#?}, child_pid={:?}, kill_ret={:?}",
                        reason,
                        self.child_pid.unwrap(),
                        kill_ret
                    );
                    return Err(err.context(format!("child_exit_reason={:#?}", reason)));
                }
            }
        }

        log::trace!("Sink child terminated, getting exit status");
        let ret =
            repeat_on_interrupt(|| unsafe { libc::read(self.receive_fd.unwrap(), buf_ptr, 4) });
        if ret != 4 {
            error!("Failed to get exit status");
            return Err(anyhow!("Failed to read from receive_non_blocking_fd"));
        }

        let exit_status = i32::from_le_bytes(buffer);
        log::trace!("Sink child status is {}", exit_status);

        if libc::WIFEXITED(exit_status) {
            // The child exited normally, through end of main() or exit(x).
            // Since the SUT server is usually running daemonly,
            // It could rarely exited normally.
            Ok(RunResult::Terminated(libc::WEXITSTATUS(exit_status)))
        } else if libc::WIFSIGNALED(exit_status) {
            let signal = libc::WTERMSIG(exit_status);
            let signal = match Signal::try_from(signal) {
                Ok(s) => s,
                Err(e) => {
                    error!(
                        "Failed to parse signal code {}. Error: {:?}. Using dummy signal SIGUSR2",
                        signal, e
                    );
                    // Some dummy signal type.
                    Signal::SIGUSR2
                }
            };
            match signal {
                Signal::SIGKILL => Ok(RunResult::TimedOut),
                Signal::SIGTERM => Ok(RunResult::Terminated(0)),
                _ => Ok(RunResult::Signalled(signal)),
            }
        } else {
            unreachable!();
        }
    }

    pub fn bitmap(&mut self) -> &mut Bitmap {
        panic!("bitmap() not implemented! You should use MapObserver in LibAFL instead");
    }

    fn purge_workdir(&self) {
        log::trace!("Purging workdir");
        if let Err(err) = self._purge_workdir() {
            log::warn!("Failed to purge workdir: {:#?}", err);
        }
        let _ = fs::create_dir_all(&self.workdir);
    }

    fn _purge_workdir(&self) -> Result<()> {
        let mut delete_ctr = 0usize;
        let dir = fs::read_dir(&self.workdir)?;
        for entry in dir {
            let entry = entry?;
            if !self.workdir_file_whitelist.contains(&entry.path()) && entry.path() != self.workdir
            {
                if entry.path().is_file() {
                    fs::remove_file(entry.path())?;
                    delete_ctr += 1;
                } else if entry.path().is_dir() {
                    fs::remove_dir_all(entry.path())?;
                    delete_ctr += 1;
                }
            }
        }

        log::trace!("Purged {} files from workdir", delete_ctr);
        Ok(())
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        self.stop();
    }
}
