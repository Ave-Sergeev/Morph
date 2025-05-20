use anyhow::{Error, Result};
use hound::{SampleFormat, WavReader};
use ndarray::{Array1, Array3};
use num_complex::Complex;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use tonic::Status;

pub struct VoiceEmbedder {
    pub session: Session,
}

impl VoiceEmbedder {
    pub fn new(model_path: &str) -> Result<Self, Error> {
        let onnx_session = Self::make_onnx_session(model_path)?;

        Ok(VoiceEmbedder { session: onnx_session })
    }

    /// Processes the given audio content and extracts embeddings using a pre-trained model.
    pub fn process(&self, content: &[u8]) -> Result<Vec<f32>, Error> {
        let audio_data = Self::transform_audio_to_i16(content)?;

        let data = audio_data
            .iter()
            .map(|item| (*item as f32) / f32::from(i16::MAX))
            .collect::<Vec<_>>();

        // Длина фрейма 0,025 шаг 0,01 сек (при sample_rate 16000Гц)
        let window_length = 400;
        let frame_length = 400;
        let frame_step = 160;
        let fft_size = 158;

        let spectrogram = Self::compute_spectrogram(&data, frame_length, frame_step, fft_size, window_length);

        let time = spectrogram[0].len();
        let frequency = spectrogram[1].len();

        let input_array = Array3::from_shape_fn((1, time, frequency), |(_, i, j)| spectrogram[i][j]);

        let inputs = ort::inputs! {
            "audio_signal" => input_array.clone(),
            "length" => Array1::<i64>::from_iter([input_array.shape()[0] as i64]),
        }?;

        let outputs = self.session.run(inputs)?;

        let embeddings = outputs["embs"]
            .try_extract_tensor::<f32>()?
            .view()
            .iter()
            .copied()
            .collect::<Vec<f32>>();

        Ok(embeddings)
    }

    /// Spectrogram calculation using FFT
    fn compute_spectrogram(
        audio: &[f32],
        frame_length: usize,
        frame_step: usize,
        fft_size: usize,
        window_length: usize,
    ) -> Vec<Vec<f32>> {
        let frames = Self::split_into_frames(audio, frame_length, frame_step);

        let hann_window: Vec<f32> = Self::create_hann_window(window_length);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut spectrogram = Vec::with_capacity(frames.len());

        for frame in frames {
            let mut windowed_frame: Vec<f32> = frame.iter().zip(hann_window.iter()).map(|(&s, &w)| s * w).collect();

            windowed_frame.resize(fft_size, 0.0);

            let mut buffer: Vec<Complex<f32>> = windowed_frame.into_iter().map(|x| Complex::new(x, 0.0)).collect();

            fft.process(&mut buffer);

            let magnitudes: Vec<f32> = buffer
                .iter()
                .take(fft_size / 2 + 1)
                .map(|complex| complex.norm_sqr())
                .collect();

            spectrogram.push(magnitudes);
        }

        spectrogram
    }

    /// Generates a Hann window of the specified length
    fn create_hann_window(window_length: usize) -> Vec<f32> {
        (0..window_length)
            .map(|i| {
                let n = i as f32;
                let window_length_f32 = window_length as f32;

                0.5 * (1.0 - (2.0 * PI * n / (window_length_f32 - 1.0)).cos())
            })
            .collect()
    }

    /// Splitting the signal into overlapping frames
    fn split_into_frames(signal: &[f32], frame_length: usize, frame_step: usize) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();
        let mut start = 0;

        while start + frame_length <= signal.len() {
            let frame = signal[start..start + frame_length].to_vec();
            frames.push(frame);
            start += frame_step;
        }

        frames
    }

    /// Creates an ONNX runtime session from a model file with optimized settings.
    fn make_onnx_session(model_path: &str) -> Result<Session, Error> {
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(session)
    }

    /// Transforms the given audio data in WAV format to a vector of 16-bit signed integers.
    fn transform_audio_to_i16(audio: &[u8]) -> Result<Vec<i16>, Status> {
        let expected_sample_rate = 16000;

        if audio.is_empty() {
            return Err(Status::invalid_argument("Audio input is empty"));
        }

        let is_valid = Self::is_mono_pcm_wav(audio, expected_sample_rate)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;

        if !is_valid {
            return Err(Status::invalid_argument(
                "Invalid WAV format: must be mono PCM with the correct sample rate",
            ));
        }

        Self::get_samples_from_wav(audio).map_err(|err| Status::invalid_argument(err.to_string()))
    }

    /// Checks if WAV data is mono PCM format with the expected sample rate.
    pub fn is_mono_pcm_wav(wav_data: &[u8], expected_sample_rate: u32) -> Result<bool> {
        let reader = WavReader::new(wav_data)?;
        let spec = reader.spec();

        let is_valid =
            spec.sample_format == SampleFormat::Int && spec.sample_rate == expected_sample_rate && spec.channels == 1;

        Ok(is_valid)
    }

    /// Extracts 16-bit signed integer samples from the given WAV audio data.
    fn get_samples_from_wav(wav: &[u8]) -> Result<Vec<i16>> {
        let mut reader = WavReader::new(wav)?;

        let samples = reader.samples().flatten().collect();

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_embedded_success() {
        let model_path = "./model/model.onnx";
        let audio_path = "./audio/test_audio.wav";

        let voice_embedded = VoiceEmbedder::new(model_path).expect("Failed to initialize VoiceEmbedder");

        let voice = std::fs::read(audio_path).expect("Failed to read audio file");

        let embeddings = voice_embedded
            .process(&voice)
            .expect("Voice embedding generation failed");

        assert!(!embeddings.is_empty(), "Embeddings don't have to be empty");

        assert_eq!(
            embeddings.len(),
            192,
            "The embedding dimension should be 192, obtained by: {}",
            embeddings.len()
        );

        for &embedding in &embeddings {
            assert!(
                embedding >= -1.0 && embedding <= 1.0,
                "The value of embedding {} goes beyond [-1, 1]",
                embedding
            );
        }
    }
}
