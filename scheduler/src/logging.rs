//! Functions related to logging.

use ansi_term::Colour::Red;
use anyhow::{Context, Result};
use fern::colors::{Color, ColoredLevelConfig};
use log::{self, kv::ToKey};
use std::{
    fs, panic,
    path::{Path, PathBuf},
    process,
    str::FromStr,
};
use yansi::Paint;

/// Setup the global logger and only log messages of level `log_level`
/// or higher.
pub fn setup_logger(log_path: &Path, log_level: &str) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    let log_file = options.create(true).truncate(true).write(true).read(true);

    fern::Dispatch::new()
        .format(|out, message, record| {
            let message = format!("{}", message);

            let level_colors = ColoredLevelConfig::new()
                .error(Color::Red)
                .warn(Color::Yellow)
                .info(Color::Blue);
            let kvs = record.key_values();
            // println!("{:#?}", kvs.);
            let time = kvs.get("time".to_key()).map_or(
                chrono::Local::now()
                    .format("%Y-%m-%d %H:%M:%S%.3f")
                    .to_string(),
                |v| v.to_borrowed_str().unwrap().to_owned(),
            );
            let tid = kvs
                .get("tid".to_key())
                .map_or(unsafe { libc::gettid() }.to_string(), |v| {
                    v.to_borrowed_str().unwrap().to_owned()
                });
            let log_source = kvs
                .get("log_source".to_key())
                .map(|s| s.to_borrowed_str().unwrap().to_owned())
                .unwrap_or("fuzzer".to_string());

            out.finish(format_args!(
                "{} [{}][{}][{}][{}:{}][{}] {}",
                time,
                match log_source.as_str() {
                    "source" => "source".green().bold(),
                    "sink" => "sink".cyan().bold(),
                    _ => log_source.as_str().magenta().bold(),
                },
                tid,
                record.target().split("::").next().unwrap_or("?"),
                record
                    .file()
                    .map(|s| PathBuf::from(s)
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned())
                    .unwrap_or_else(|| "?".to_owned()),
                record
                    .line()
                    .map(|l| format!("{}", l))
                    .unwrap_or_else(|| "?".to_owned()),
                level_colors.color(record.level()),
                message,
            ))
        })
        .level(
            log::LevelFilter::from_str(log_level)
                .context(format!("'{}' is not a valid log level", log_level))?,
        )
        .chain(std::io::stdout())
        .chain(log_file.open(log_path)?)
        .apply()?;
    Ok(())
}

fn panic_hook(info: &panic::PanicInfo<'_>) {
    log::error!("{}", Red.paint(format!("\nPanic: {:#?}", info)));
    if let Some(location) = info.location() {
        let file = location.file();
        let line = location.line();
        log::error!("{}", Red.paint(format!("Location: {}:{}", file, line)));
    }
}

pub fn setup_panic_logging() {
    log::info!("Panics are logged via log::error");
    panic::set_hook(Box::new(panic_hook));
}
