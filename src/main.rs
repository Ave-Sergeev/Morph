use crate::controllers::embedding_controller::VoiceEmbeddingController;
use crate::pb::inference_pb::embedding_server::EmbeddingServer;
use crate::setting::settings::Settings;
use env_logger::Builder;
use log::LevelFilter;
use std::error::Error;
use std::str::FromStr;
use tonic::transport::Server;

mod controllers;
mod pb;
mod service;
mod setting;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let settings = Settings::new("config.yaml").map_err(|err| format!("Failed to load setting: {err}"))?;

    Builder::new()
        .filter_level(LevelFilter::from_str(settings.logging.log_level.as_str()).unwrap_or(LevelFilter::Info))
        .init();

    log::info!("Settings:\n{}", settings.json_pretty());

    let voice_embedding_controller = VoiceEmbeddingController::new(settings.model);

    let address = format!("{}:{}", settings.server.host, settings.server.port)
        .parse()
        .map_err(|err| format!("Invalid server address: {err}"))?;

    log::info!("Server listening on: {address}");

    let voice_embedding_server = EmbeddingServer::new(voice_embedding_controller);

    Server::builder()
        .add_service(voice_embedding_server)
        .serve(address)
        .await
        .map_err(|err| format!("GRPC server returned error: {err}"))?;

    Ok(())
}
