use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use std::{env, time::Duration};

use anyhow::{anyhow, Context, Result};
use ipmpsc::{Receiver, Sender, SharedRingBuffer};
use serde::{Deserialize, Serialize};

const IOSYNC_TIMEOUT: Duration = Duration::from_millis(10);
pub const IOSYNC_BUF_SIZE_DEFAULT: usize = 1024;

// #[derive(Debug, Error)]
// #[allow(unused)]
// pub enum IOSyncChannelError {
    
// }

#[derive(Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct IOSyncMessage(usize);

pub struct IOSyncChannel {
    name: String,
    tx: Sender,
    rx: Receiver,
    size: usize,
    seq: usize,
    interrupted: Arc<RwLock<bool>>,
}

impl Debug for IOSyncChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IOSyncChannel")
            .field("name", &self.name)
            .field("size", &self.size)
            .field("seq", &self.seq)
            .finish()
    }
}

impl IOSyncChannel {
    pub fn new(name: &str, shm_size: usize) -> IOSyncChannel {
        let tx_buf = SharedRingBuffer::create(format!("{name}_TX").as_str(), shm_size as u32)
            .expect("Failed to create shared ring buffer for tx");
        let rx_buf = SharedRingBuffer::create(format!("{name}_RX").as_str(), shm_size as u32)
            .expect("Failed to create shared ring buffer for rx");

        let tx = Sender::new(tx_buf);
        let rx = Receiver::new(rx_buf);

        IOSyncChannel {
            name: name.to_string(),
            tx,
            rx,
            size: shm_size,
            seq: 0,
            interrupted: Arc::new(RwLock::new(false)),
        }
    }

    pub fn reset_interruption(&mut self) -> Arc<RwLock<bool>> {
        self.interrupted = Arc::new(RwLock::new(false));
        self.interrupted.clone()
    }

    pub fn set_state(&self, paused: bool) -> Result<bool> {
        match self.interrupted.write() {
            Ok(mut interrupted) => {
                *interrupted = paused;
                Ok(paused)
            }
            Err(e) => Err(anyhow!(e.to_string())),
        }
    }

    pub fn io_yield(&self) -> Result<()> {
        let mut total_trials = 0;
        loop {
            match self.interrupted.read() {
                Ok(interrupted) => {
                    if *interrupted {
                        break;
                    }
                }
                Err(e) => {
                    return Err(anyhow!(e.to_string()));
                }
            }
            let msg = IOSyncMessage(self.seq);
            match self.tx.send_timeout(&msg, IOSYNC_TIMEOUT) {
                Ok(sent) => {
                    if sent {
                        break;
                    } else {
                        let seq = self.seq;
                        log::warn!(
                            "[{name}] Synchronizing {seq} timeout, retrying...",
                            name = self.name
                        );
                        total_trials += 1;
                    }
                }
                Err(e) => {
                    log::error!(
                        "[{name}] Failed to send message, retrying...error is {e}",
                        name = self.name
                    );
                    total_trials += 1;
                }
            }
            if total_trials >= 10 {
                log::error!(
                    "[{name}] Failed to send message, trials out",
                    name = self.name
                );
                return Err(anyhow!(format!("Failed to send message, trials out")));
            }
        }
        Ok(())
    }

    pub fn io_await(&mut self) -> Result<bool> {
        let mut total_tmout = 0_u128;
        loop {
            match self.interrupted.read() {
                Ok(interrupted) => {
                    if *interrupted {
                        break;
                    }
                }
                Err(e) => {
                    return Err(anyhow!(e.to_string()));
                }
            }
            match self.rx.recv_timeout(IOSYNC_TIMEOUT) {
                Ok(msg) => {
                    if let Some(msg) = msg {
                        let IOSyncMessage(seq) = msg;
                        if seq == self.seq {
                            log::trace!(
                                "[{name}] Synchronized {seq} successfully",
                                name = self.name
                            );
                            self.seq += 1;
                            break;
                        }
                    } else {
                        log::trace!(
                            "[{name}] Synchronizing {seq} timeout, retrying...",
                            seq = self.seq,
                            name = self.name
                        );
                    }
                }
                Err(e) => {
                    return Err(e).context(format!("Fatal error in receving message from shared ring buffer from {name}, next event seq is {seq}", name=self.name, seq=self.seq))
                }
            }
            total_tmout += IOSYNC_TIMEOUT.as_micros();
            if total_tmout >= 10_000_000 {
                log::trace!(
                    "[{name}] Synchronizing {seq} timeout for 10ms",
                    seq = self.seq,
                    name = self.name
                );
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn to_env(&self, agent: &str) {
        env::set_var(format!("IO_SYNC_{agent}"), self.name.clone());
        env::set_var(
            format!("{name}_SIZE", name = self.name),
            self.size.to_string(),
        );
    }

    pub fn from_env(agent: &str) -> IOSyncChannel {
        let name = env::var(format!("IO_SYNC_{agent}")).expect(
            format!("Failed to get iosync channel name with env key: IO_SYNC_{agent}").as_str(),
        );
        let size = env::var(format!("{name}_SIZE"))
            .expect("Failed to get IO_SYNC_SIZE from env")
            .parse::<usize>()
            .expect("Failed to parse IO_SYNC_SIZE");

        let tx_buf = SharedRingBuffer::open(format!("IO_SYNC_TX_{name}").as_str())
            .expect("Failed to open shared ring buffer for tx");
        let rx_buf = SharedRingBuffer::open(format!("IO_SYNC_RX_{name}").as_str())
            .expect("Failed to open shared ring buffer for rx");
        let tx = Sender::new(tx_buf);
        let rx = Receiver::new(rx_buf);

        IOSyncChannel {
            name: name.to_string(),
            tx,
            rx,
            size,
            seq: 0,
            interrupted: Arc::new(RwLock::new(false)),
        }
    }
}
