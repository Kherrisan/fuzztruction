use ansi_term::Colour::Red;
use anyhow::Result;
use fuzztruction_shared::{
    aux_messages::AuxStreamType, aux_stream::AuxStreamBuilder, log_utils::LogRecordWrapper,
};
use log::{kv::ToValue, Record};
use std::{panic, process};

use crate::agent;

pub fn setup_logger(level: log::Level) -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .level(level.to_level_filter())
        // .chain(std::io::stderr())
        .chain(fern::Output::call(|record| log_to_posix_mq(record)))
        .apply()?;
    Ok(())
}

/// Passes the given log Record to the coordinator via the POSIX queue.
/// This allows us to configure logging realted things at a single location.
pub fn log_to_posix_mq(record: &Record) {
    let time = chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S%.3f")
        .to_string();
    let tid = unsafe { libc::gettid() };
    let kvs = [
        ("log_source", "source".to_value()),
        ("time", time.as_str().to_value()),
        ("tid", tid.to_value()),
    ];
    let record = record.to_builder().key_values(&kvs).build();
    let record = LogRecordWrapper::from_record(&record);
    let msg = serde_json::to_string(&record).expect("Failed serialize log record.");
    let mut debug_stream = AuxStreamBuilder::new(AuxStreamType::LogRecord);
    let msgs = debug_stream.from_str(&msg).build();
    for m in msgs.into_iter() {
        let r = agent::send_message(m, 60000);
        if let Err(e) = r {
            eprintln!("Failed to log message: {}", e);
        }
    }
}

fn panic_hook(info: &panic::PanicInfo<'_>) {
    log::error!("{}", Red.paint(format!("\nAgent panic: {:#?}", info)));
    if let Some(location) = info.location() {
        let file = location.file();
        let line = location.line();
        log::error!("{}", Red.paint(format!("Location: {}:{}", file, line)));
    }
    process::abort();
}

pub fn setup_panic_logging() {
    log::info!("Panics are logged via log::error");
    panic::set_hook(Box::new(panic_hook));
}
