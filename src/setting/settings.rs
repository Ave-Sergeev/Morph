use anyhow::Result;
use config::Config;
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Server {
    pub host: String,
    pub port: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Logging {
    pub log_level: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ModelSettings {
    pub path: String,
    pub sample_rate: usize,
    pub window_length: usize,
    pub frame_length: usize,
    pub frame_step: usize,
    pub fft_size: usize,
    pub n_mels: usize,
    pub ref_value: f32,
    pub amin: f32,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Settings {
    pub server: Server,
    pub logging: Logging,
    pub model: ModelSettings,
}

impl Settings {
    pub fn new(location: &str) -> Result<Self> {
        let mut builder = Config::builder();

        if Path::new(location).exists() {
            builder = builder.add_source(config::File::with_name(location));
        } else {
            log::warn!("Configuration file not found");
        }

        let settings = builder.build()?.try_deserialize()?;

        Ok(settings)
    }

    pub fn json_pretty(&self) -> String {
        to_string_pretty(&self).expect("Failed serialize")
    }
}
