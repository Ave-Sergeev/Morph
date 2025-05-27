use crate::pb::inference_pb;
use crate::pb::inference_pb::{VoiceEmbeddingRequest, VoiceEmbeddingResponse};
use crate::service::voice_embedder::VoiceEmbedder;
use crate::setting::settings::ModelSettings;
use inference_pb::embedding_server;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub struct VoiceEmbeddingController {
    pub voice_embedded: Arc<VoiceEmbedder>,
}

impl VoiceEmbeddingController {
    pub fn new(model_settings: &ModelSettings) -> Self {
        let embedded = VoiceEmbedder::new(&model_settings.path).expect("Failed to initialize model");

        Self {
            voice_embedded: Arc::new(embedded),
        }
    }
}

#[tonic::async_trait]
impl embedding_server::Embedding for VoiceEmbeddingController {
    async fn embed_voice(
        &self,
        request: Request<VoiceEmbeddingRequest>,
    ) -> Result<Response<VoiceEmbeddingResponse>, Status> {
        let request = request.into_inner();

        log::info!("Received request is id: {}", request.id);

        let embeddings = self
            .voice_embedded
            .process(&request.content)
            .map_err(|err| Status::internal(err.to_string()))?;

        Ok(Response::new(VoiceEmbeddingResponse {
            id: request.id,
            embeddings,
        }))
    }
}
