use crate::setting::settings::ModelSettings;
use anyhow::{Error, Result};
use hound::{SampleFormat, WavReader};
use ndarray::{Array1, Array2, Array3};
use num_complex::{Complex, ComplexFloat};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use tonic::Status;

pub struct VoiceEmbedder {
    pub session: Session,
    pub settings: ModelSettings,
}

impl VoiceEmbedder {
    pub fn new(settings: ModelSettings) -> Result<Self, Error> {
        let session = Self::make_onnx_session(settings.path.as_str())?;

        Ok(VoiceEmbedder { session, settings })
    }

    /// Processes the given audio content and extracts embeddings using a pre-trained model.
    pub fn process(&self, content: &[u8]) -> Result<Vec<f32>, Error> {
        let audio_data = Self::transform_audio_to_i16(content)?;

        let data = audio_data
            .iter()
            .map(|item| (*item as f32) / f32::from(i16::MAX))
            .collect::<Vec<_>>();

        let spectrogram = Self::compute_spectrogram(&data, &self.settings);

        let time = spectrogram.len();
        let frequency = spectrogram[0].len();

        let input_array = Array3::from_shape_fn((1, frequency, time), |(_, i, j)| spectrogram[j][i]);

        let inputs = ort::inputs! {
            "audio_signal" => input_array.view(),
            "length" => Array1::<i64>::from_iter([time as i64]),
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
    fn compute_spectrogram(audio: &[f32], settings: &ModelSettings) -> Vec<Vec<f32>> {
        let window_length = settings.window_length;
        let frame_length = settings.frame_length;
        let sample_rate = settings.sample_rate;
        let frame_step = settings.frame_step;
        let ref_value = settings.ref_value;
        let fft_size = settings.fft_size;
        let n_mels = settings.n_mels;
        let amin = settings.amin;

        let frames = Self::split_into_frames(audio, frame_length, frame_step);

        let hann_window: Vec<f32> = Self::create_hann_window(window_length);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut spectrogram = Vec::with_capacity(frames.len());

        for frame in frames {
            let mut windowed_frame: Vec<f32> = frame.iter().zip(hann_window.iter()).map(|(&s, &w)| s * w).collect();

            windowed_frame.resize(fft_size, 0.0);

            let mut buffer: Vec<Complex<f32>> =
                windowed_frame.into_iter().map(|elem| Complex::new(elem, 0.0)).collect();

            fft.process(&mut buffer);

            let power_spectrum: Vec<f32> = buffer
                .iter()
                .take(fft_size / 2 + 1)
                .map(|complex| complex.abs())
                .collect();

            let mel_filterbank = Self::create_mel_filterbank(sample_rate, fft_size, n_mels);

            let mut mel_spectrum = Vec::with_capacity(n_mels);

            for m in 0..n_mels {
                let mel_energy: f32 = mel_filterbank
                    .row(m)
                    .iter()
                    .zip(&power_spectrum)
                    .map(|(&f, &p)| f * p)
                    .sum();
                mel_spectrum.push(mel_energy);
            }

            let mel_spectrum_db = Self::power_to_db(&mel_spectrum, ref_value, amin, Some(80.0));

            spectrogram.push(mel_spectrum_db);
        }

        spectrogram
    }

    /// Converts the power spectrum to decibels
    fn power_to_db(spectrum: &[f32], ref_value: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
        let ref_value = ref_value.max(amin);

        let log_spec: Vec<f32> = spectrum
            .iter()
            .map(|&m| {
                let m_clamped = m.max(amin);
                10.0 * (m_clamped / ref_value).log10()
            })
            .collect();

        if let Some(top_db) = top_db {
            let max_db = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            log_spec.into_iter().map(|db| db.max(max_db - top_db)).collect()
        } else {
            log_spec
        }
    }

    /// Converts frequency to chalk scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Converts chalk back to frequency
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Constructs a chalk filterbank: the matrix `[n_mels, fft_bins]`
    fn create_mel_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize) -> Array2<f32> {
        let f_min = 0.0;
        let f_max = (sample_rate / 2) as f32;

        let n_fft_bins = n_fft / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((n_mels, n_fft_bins));

        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);
        let mel_points: Vec<f32> = (0..(n_mels + 2))
            .map(|i| mel_min + (i as f32) * (mel_max - mel_min) / (n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((f / sample_rate as f32) * (n_fft as f32)).floor() as usize)
            .collect();

        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            let left_to_center = (f_center - f_left).max(1) as f32;
            let center_to_right = (f_right - f_center).max(1) as f32;

            for k in f_left..f_center {
                if k < n_fft_bins {
                    filterbank[[m, k]] = (k - f_left) as f32 / left_to_center;
                }
            }

            for k in f_center..f_right {
                if k < n_fft_bins {
                    filterbank[[m, k]] = (f_right - k) as f32 / center_to_right;
                }
            }
        }

        filterbank
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
    use crate::setting::settings::Settings;

    #[test]
    fn test_voice_embedded_success() {
        let audio_path = "./audio/test_audio.wav";

        let settings = Settings::new("config.yaml").expect("Failed to load setting");

        let voice_embedded = VoiceEmbedder::new(settings.model).expect("Failed to initialize VoiceEmbedder");

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
    }
}
