use anyhow::{Error, Result};
use plotters::prelude::*;

#[allow(dead_code)]
/// Building a spectrogram image and saving it in .png file
pub fn save_spectrogram(
    spectrogram: &[Vec<f32>],
    output_path: &str,
    sample_rate: Option<usize>,
    n_fft: Option<usize>,
) -> Result<(), Error> {
    let height = spectrogram.len();
    let width = spectrogram.first().map_or(0, |row| row.len());

    let root = BitMapBackend::new(output_path, (width as u32 * 2, height as u32 * 2)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrogram", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0.0..width as f32,  // X-axis (time)
            0.0..height as f32, // Y-axis (frequency)
        )?;

    if let (Some(sr), Some(n_fft)) = (sample_rate, n_fft) {
        let time_per_frame = (n_fft as f32) / (sr as f32);
        let max_freq = sr as f32 / 2.0;

        chart
            .configure_mesh()
            .x_desc("Time (sec)")
            .y_desc("Frequency (Hz)")
            .x_labels(5)
            .y_labels(5)
            .x_label_formatter(&|x| format!("{:.2}", x * time_per_frame))
            .y_label_formatter(&|y| format!("{:.0}", y * max_freq / height as f32))
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_desc("Time (frames)")
            .y_desc("Frequency (bins)")
            .draw()?;
    }

    let max_val = spectrogram
        .iter()
        .flatten()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0);
    let min_val = spectrogram
        .iter()
        .flatten()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    for (y, row) in spectrogram.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let normalized = (value - min_val) / (max_val - min_val);
            let color = ViridisRGBA::get_color_normalized(normalized as f64, 0.0, 1.0);

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x as f32, y as f32), (x as f32 + 1.0, y as f32 + 1.0)],
                color.filled(),
            )))?;
        }
    }

    Ok(())
}
