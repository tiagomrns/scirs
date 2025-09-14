//! Visualization and reporting utilities for image processing results
//!
//! This module provides tools for creating visual representations of
//! image processing results, statistical plots, and comprehensive
//! analysis reports. Designed for scientific documentation and
//! presentation of image analysis workflows.

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use std::fmt::{Debug, Write};

use crate::analysis::{ImageQualityMetrics, TextureMetrics};
use crate::error::{NdimageError, NdimageResult};
use crate::utils::{safe_f64_to_float, safe_usize_to_float};
use statrs::statistics::Statistics;

/// Color map types for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Grayscale color map
    Gray,
    /// Jet color map (blue to red)
    Jet,
    /// Viridis perceptually uniform color map
    Viridis,
    /// Plasma color map
    Plasma,
    /// Inferno color map
    Inferno,
    /// Hot color map (black to white through red/yellow)
    Hot,
    /// Cool color map (cyan to magenta)
    Cool,
    /// Spring color map (magenta to yellow)
    Spring,
    /// Summer color map (green to yellow)
    Summer,
    /// Autumn color map (red to yellow)
    Autumn,
    /// Winter color map (blue to green)
    Winter,
}

/// Configuration for plotting operations
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the plot in pixels
    pub width: usize,
    /// Height of the plot in pixels
    pub height: usize,
    /// Title of the plot
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Color map to use
    pub colormap: ColorMap,
    /// Whether to show grid
    pub show_grid: bool,
    /// Number of bins for histograms
    pub num_bins: usize,
    /// Output format
    pub format: ReportFormat,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Image Analysis Plot".to_string(),
            xlabel: "X".to_string(),
            ylabel: "Y".to_string(),
            colormap: ColorMap::Viridis,
            show_grid: true,
            num_bins: 256,
            format: ReportFormat::Text,
        }
    }
}

/// Configuration for report generation
#[derive(Debug, Clone)]
pub struct ReportConfig {
    /// Title of the report
    pub title: String,
    /// Author information
    pub author: String,
    /// Include detailed statistics
    pub include_statistics: bool,
    /// Include quality metrics
    pub include_qualitymetrics: bool,
    /// Include texture analysis
    pub includetexture_analysis: bool,
    /// Include histograms
    pub include_histograms: bool,
    /// Include profile plots
    pub include_profiles: bool,
    /// Output format (html, markdown, text)
    pub format: ReportFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Markdown,
    Text,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "Image Analysis Report".to_string(),
            author: "SciRS2 Image Analysis".to_string(),
            include_statistics: true,
            include_qualitymetrics: true,
            includetexture_analysis: true,
            include_histograms: true,
            include_profiles: true,
            format: ReportFormat::Markdown,
        }
    }
}

/// RGB color representation
#[derive(Debug, Clone, Copy)]
pub struct RgbColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl RgbColor {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// Create a color map for visualization
#[allow(dead_code)]
pub fn create_colormap(_colormap: ColorMap, numcolors: usize) -> Vec<RgbColor> {
    let mut _colors = Vec::with_capacity(numcolors);

    for i in 0..numcolors {
        let t = i as f64 / (numcolors - 1) as f64;
        let color = match _colormap {
            ColorMap::Gray => {
                let val = (t * 255.0) as u8;
                RgbColor::new(val, val, val)
            }
            ColorMap::Jet => jet_colormap(t),
            ColorMap::Viridis => viridis_colormap(t),
            ColorMap::Plasma => plasma_colormap(t),
            ColorMap::Inferno => inferno_colormap(t),
            ColorMap::Hot => hot_colormap(t),
            ColorMap::Cool => cool_colormap(t),
            ColorMap::Spring => spring_colormap(t),
            ColorMap::Summer => summer_colormap(t),
            ColorMap::Autumn => autumn_colormap(t),
            ColorMap::Winter => winter_colormap(t),
        };
        _colors.push(color);
    }

    _colors
}

/// Generate a histogram plot representation
#[allow(dead_code)]
pub fn plot_histogram<T>(data: &ArrayView1<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if data.is_empty() {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    // Find min and max values
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All _data values are the same".into(),
        ));
    }

    // Create histogram bins
    let mut histogram = vec![0usize; config.num_bins];
    let range = max_val - min_val;
    let bin_size = range / safe_usize_to_float::<T>(config.num_bins)?;

    for &value in data.iter() {
        let normalized = (value - min_val) / bin_size;
        let bin_idx = normalized.to_usize().unwrap_or(0).min(config.num_bins - 1);
        histogram[bin_idx] += 1;
    }

    // Generate plot representation
    let max_count = *histogram.iter().max().unwrap_or(&1);
    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='histogram-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='histogram-bars'>")?;

            for (i, &count) in histogram.iter().enumerate() {
                let height_percent = (count as f64 / max_count as f64) * 100.0;
                let bin_start = min_val + safe_usize_to_float::<T>(i)? * bin_size;
                let bin_end = bin_start + bin_size;

                writeln!(
                    &mut plot,
                    "<div class='bar' style='height: {:.1}%' title='[{:.3}, {:.3}): {}'></div>",
                    height_percent,
                    bin_start.to_f64().unwrap_or(0.0),
                    bin_end.to_f64().unwrap_or(0.0),
                    count
                )?;
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='axis-labels'>")?;
            writeln!(&mut plot, "<span class='xlabel'>{}</span>", config.xlabel)?;
            writeln!(&mut plot, "<span class='ylabel'>{}</span>", config.ylabel)?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;

            for (i, &count) in histogram.iter().enumerate() {
                let bar_length = (count as f64 / max_count as f64 * 50.0) as usize;
                let bin_center = min_val
                    + (safe_usize_to_float::<T>(i)? + safe_f64_to_float::<T>(0.5)?) * bin_size;

                writeln!(
                    &mut plot,
                    "{:8.3} |{:<50} {}",
                    bin_center.to_f64().unwrap_or(0.0),
                    "*".repeat(bar_length),
                    count
                )?;
            }

            writeln!(&mut plot, "```")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "**{}** vs **{}**", config.xlabel, config.ylabel)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;

            for (i, &count) in histogram.iter().enumerate() {
                let bar_length = (count as f64 / max_count as f64 * 50.0) as usize;
                let bin_center = min_val
                    + (safe_usize_to_float::<T>(i)? + safe_f64_to_float::<T>(0.5)?) * bin_size;

                writeln!(
                    &mut plot,
                    "{:8.3} |{:<50} {}",
                    bin_center.to_f64().unwrap_or(0.0),
                    "*".repeat(bar_length),
                    count
                )?;
            }

            writeln!(&mut plot)?;
            writeln!(&mut plot, "X-axis: {}", config.xlabel)?;
            writeln!(&mut plot, "Y-axis: {}", config.ylabel)?;
        }
    }

    Ok(plot)
}

/// Generate a profile plot (line plot) representation
#[allow(dead_code)]
pub fn plot_profile<T>(
    x_data: &ArrayView1<T>,
    y_data: &ArrayView1<T>,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if x_data.len() != y_data.len() {
        return Err(NdimageError::InvalidInput(
            "X and Y _data must have the same length".into(),
        ));
    }

    if x_data.is_empty() {
        return Err(NdimageError::InvalidInput("Data arrays are empty".into()));
    }

    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='profile-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Plot _data points and lines
            let x_min = x_data.iter().cloned().fold(T::infinity(), T::min);
            let x_max = x_data.iter().cloned().fold(T::neg_infinity(), T::max);
            let y_min = y_data.iter().cloned().fold(T::infinity(), T::min);
            let y_max = y_data.iter().cloned().fold(T::neg_infinity(), T::max);

            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            if x_range > T::zero() && y_range > T::zero() {
                let mut path_data = String::new();

                for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
                    let px = ((x - x_min) / x_range * safe_usize_to_float(config.width - 100)?
                        + safe_f64_to_float::<T>(50.0)?)
                    .to_f64()
                    .unwrap_or(0.0);
                    let py = (config.height as f64 - 50.0)
                        - ((y - y_min) / y_range * safe_usize_to_float(config.height - 100)?)
                            .to_f64()
                            .unwrap_or(0.0);

                    if i == 0 {
                        write!(&mut path_data, "M {} {}", px, py)?;
                    } else {
                        write!(&mut path_data, " L {} {}", px, py)?;
                    }
                }

                writeln!(
                    &mut plot,
                    "<path d='{}' stroke='blue' stroke-width='2' fill='none'/>",
                    path_data
                )?;
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "<div class='axis-labels'>")?;
            writeln!(&mut plot, "<span class='xlabel'>{}</span>", config.xlabel)?;
            writeln!(&mut plot, "<span class='ylabel'>{}</span>", config.ylabel)?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;

            for (&x, &y) in x_data.iter().zip(y_data.iter()) {
                writeln!(
                    &mut plot,
                    "{:10.4} {:10.4}",
                    x.to_f64().unwrap_or(0.0),
                    y.to_f64().unwrap_or(0.0)
                )?;
            }

            writeln!(&mut plot, "```")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "**{}** vs **{}**", config.xlabel, config.ylabel)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "{:>10} {:>10}", config.xlabel, config.ylabel)?;
            writeln!(&mut plot, "{}", "-".repeat(22))?;

            for (&x, &y) in x_data.iter().zip(y_data.iter()) {
                writeln!(
                    &mut plot,
                    "{:10.4} {:10.4}",
                    x.to_f64().unwrap_or(0.0),
                    y.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Visualize gradient information as a vector field
#[allow(dead_code)]
pub fn visualize_gradient<T>(
    gradient_x: &ArrayView2<T>,
    gradient_y: &ArrayView2<T>,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if gradient_x.dim() != gradient_y.dim() {
        return Err(NdimageError::DimensionError(
            "Gradient components must have the same dimensions".into(),
        ));
    }

    let (height, width) = gradient_x.dim();
    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='gradient-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Sample gradient vectors at regular intervals
            let step_x = width.max(1) / (config.width / 20).max(1);
            let step_y = height.max(1) / (config.height / 20).max(1);

            for i in (0..height).step_by(step_y) {
                for j in (0..width).step_by(step_x) {
                    let gx = gradient_x[[i, j]].to_f64().unwrap_or(0.0);
                    let gy = gradient_y[[i, j]].to_f64().unwrap_or(0.0);

                    let magnitude = (gx * gx + gy * gy).sqrt();
                    if magnitude > 1e-6 {
                        let scale = 10.0 / magnitude.max(1e-6);
                        let start_x = j as f64 * config.width as f64 / width as f64;
                        let start_y = i as f64 * config.height as f64 / height as f64;
                        let end_x = start_x + gx * scale;
                        let end_y = start_y + gy * scale;

                        writeln!(
                            &mut plot,
                            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='red' stroke-width='1'/>",
                            start_x, start_y, end_x, end_y
                        )?;

                        // Add arrowhead
                        let arrow_len = 3.0;
                        let angle = gy.atan2(gx);
                        let arrow1_x = end_x - arrow_len * (angle - 0.5).cos();
                        let arrow1_y = end_y - arrow_len * (angle - 0.5).sin();
                        let arrow2_x = end_x - arrow_len * (angle + 0.5).cos();
                        let arrow2_y = end_y - arrow_len * (angle + 0.5).sin();

                        writeln!(
                            &mut plot,
                            "<polygon points='{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}' fill='red'/>",
                            end_x, end_y, arrow1_x, arrow1_y, arrow2_x, arrow2_y
                        )?;
                    }
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Gradient vector field visualization")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "- Image dimensions: {}×{}", width, height)?;

            // Compute some statistics
            let magnitude_sum: f64 = gradient_x
                .iter()
                .zip(gradient_y.iter())
                .map(|(&gx, &gy)| {
                    let gx_f = gx.to_f64().unwrap_or(0.0);
                    let gy_f = gy.to_f64().unwrap_or(0.0);
                    (gx_f * gx_f + gy_f * gy_f).sqrt()
                })
                .sum();

            let avg_magnitude = magnitude_sum / (width * height) as f64;
            writeln!(
                &mut plot,
                "- Average gradient magnitude: {:.4}",
                avg_magnitude
            )?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Gradient Vector Field")?;
            writeln!(&mut plot, "Image dimensions: {}×{}", width, height)?;

            // Show a text-based representation
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Sample gradient vectors:")?;
            writeln!(
                &mut plot,
                "{:>5} {:>5} {:>10} {:>10} {:>10}",
                "Row", "Col", "Grad_X", "Grad_Y", "Magnitude"
            )?;
            writeln!(&mut plot, "{}", "-".repeat(50))?;

            let step = height.max(width) / 10;
            for i in (0..height).step_by(step) {
                for j in (0..width).step_by(step) {
                    let gx = gradient_x[[i, j]].to_f64().unwrap_or(0.0);
                    let gy = gradient_y[[i, j]].to_f64().unwrap_or(0.0);
                    let magnitude = (gx * gx + gy * gy).sqrt();

                    writeln!(
                        &mut plot,
                        "{:5} {:5} {:10.4} {:10.4} {:10.4}",
                        i, j, gx, gy, magnitude
                    )?;
                }
            }
        }
    }

    Ok(plot)
}

/// Generate a comprehensive analysis report
#[allow(dead_code)]
pub fn generate_report<T>(
    image: &ArrayView2<T>,
    qualitymetrics: Option<&ImageQualityMetrics<T>>,
    texturemetrics: Option<&TextureMetrics<T>>,
    config: &ReportConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let mut report = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut report, "<!DOCTYPE html>")?;
            writeln!(&mut report, "<html><head><title>{}</title>", config.title)?;
            writeln!(&mut report, "<style>")?;
            writeln!(
                &mut report,
                "body {{ font-family: Arial, sans-serif; margin: 20px; }}"
            )?;
            writeln!(
                &mut report,
                "table {{ border-collapse: collapse; width: 100%; }}"
            )?;
            writeln!(
                &mut report,
                "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
            )?;
            writeln!(&mut report, "th {{ background-color: #f2f2f2; }}")?;
            writeln!(
                &mut report,
                ".metric-value {{ font-weight: bold; color: #2E86AB; }}"
            )?;
            writeln!(&mut report, "</style></head><body>")?;
            writeln!(&mut report, "<h1>{}</h1>", config.title)?;
            writeln!(
                &mut report,
                "<p><em>Generated by {}</em></p>",
                config.author
            )?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut report, "# {}", config.title)?;
            writeln!(&mut report)?;
            writeln!(&mut report, "*Generated by {}*", config.author)?;
            writeln!(&mut report)?;
        }
        ReportFormat::Text => {
            writeln!(&mut report, "{}", config.title)?;
            writeln!(&mut report, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut report)?;
            writeln!(&mut report, "Generated by: {}", config.author)?;
            writeln!(&mut report)?;
        }
    }

    // Basic image information
    let (height, width) = image.dim();
    addimage_info(&mut report, width, height, config.format)?;

    // Basic statistics
    if config.include_statistics {
        add_basic_statistics(&mut report, image, config.format)?;
    }

    // Quality metrics
    if config.include_qualitymetrics {
        if let Some(metrics) = qualitymetrics {
            add_qualitymetrics(&mut report, metrics, config.format)?;
        }
    }

    // Texture analysis
    if config.includetexture_analysis {
        if let Some(metrics) = texturemetrics {
            addtexturemetrics(&mut report, metrics, config.format)?;
        }
    }

    // Close HTML if needed
    if config.format == ReportFormat::Html {
        writeln!(&mut report, "</body></html>")?;
    }

    Ok(report)
}

#[allow(dead_code)]
fn addimage_info(
    report: &mut String,
    width: usize,
    height: usize,
    format: ReportFormat,
) -> Result<(), std::fmt::Error> {
    match format {
        ReportFormat::Html => {
            writeln!(report, "<h2>Image Information</h2>")?;
            writeln!(report, "<table>")?;
            writeln!(report, "<tr><th>Property</th><th>Value</th></tr>")?;
            writeln!(
                report,
                "<tr><td>Width</td><td class='metric-value'>{}</td></tr>",
                width
            )?;
            writeln!(
                report,
                "<tr><td>Height</td><td class='metric-value'>{}</td></tr>",
                height
            )?;
            writeln!(
                report,
                "<tr><td>Total Pixels</td><td class='metric-value'>{}</td></tr>",
                width * height
            )?;
            writeln!(
                report,
                "<tr><td>Aspect Ratio</td><td class='metric-value'>{:.3}</td></tr>",
                width as f64 / height as f64
            )?;
            writeln!(report, "</table>")?;
        }
        ReportFormat::Markdown => {
            writeln!(report, "## Image Information")?;
            writeln!(report)?;
            writeln!(report, "| Property | Value |")?;
            writeln!(report, "|----------|-------|")?;
            writeln!(report, "| Width | {} |", width)?;
            writeln!(report, "| Height | {} |", height)?;
            writeln!(report, "| Total Pixels | {} |", width * height)?;
            writeln!(
                report,
                "| Aspect Ratio | {:.3} |",
                width as f64 / height as f64
            )?;
            writeln!(report)?;
        }
        ReportFormat::Text => {
            writeln!(report, "Image Information")?;
            writeln!(report, "-----------------")?;
            writeln!(report, "Width:        {}", width)?;
            writeln!(report, "Height:       {}", height)?;
            writeln!(report, "Total Pixels: {}", width * height)?;
            writeln!(report, "Aspect Ratio: {:.3}", width as f64 / height as f64)?;
            writeln!(report)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn add_basic_statistics<T>(
    report: &mut String,
    image: &ArrayView2<T>,
    format: ReportFormat,
) -> Result<(), std::fmt::Error>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let mean = image.mean().unwrap_or(T::zero());
    let min_val = image.iter().cloned().fold(T::infinity(), T::min);
    let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

    let variance = image
        .mapv(|x| (x - mean) * (x - mean))
        .mean()
        .unwrap_or(T::zero());
    let std_dev = variance.sqrt();

    match format {
        ReportFormat::Html => {
            writeln!(report, "<h2>Basic Statistics</h2>")?;
            writeln!(report, "<table>")?;
            writeln!(report, "<tr><th>Statistic</th><th>Value</th></tr>")?;
            writeln!(
                report,
                "<tr><td>Mean</td><td class='metric-value'>{:.6}</td></tr>",
                mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Standard Deviation</td><td class='metric-value'>{:.6}</td></tr>",
                std_dev.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Variance</td><td class='metric-value'>{:.6}</td></tr>",
                variance.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Minimum</td><td class='metric-value'>{:.6}</td></tr>",
                min_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Maximum</td><td class='metric-value'>{:.6}</td></tr>",
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Range</td><td class='metric-value'>{:.6}</td></tr>",
                (max_val - min_val).to_f64().unwrap_or(0.0)
            )?;
            writeln!(report, "</table>")?;
        }
        ReportFormat::Markdown => {
            writeln!(report, "## Basic Statistics")?;
            writeln!(report)?;
            writeln!(report, "| Statistic | Value |")?;
            writeln!(report, "|-----------|-------|")?;
            writeln!(report, "| Mean | {:.6} |", mean.to_f64().unwrap_or(0.0))?;
            writeln!(
                report,
                "| Standard Deviation | {:.6} |",
                std_dev.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Variance | {:.6} |",
                variance.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Minimum | {:.6} |",
                min_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Maximum | {:.6} |",
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Range | {:.6} |",
                (max_val - min_val).to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
        ReportFormat::Text => {
            writeln!(report, "Basic Statistics")?;
            writeln!(report, "----------------")?;
            writeln!(
                report,
                "Mean:              {:.6}",
                mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Standard Deviation:{:.6}",
                std_dev.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Variance:          {:.6}",
                variance.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Minimum:           {:.6}",
                min_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Maximum:           {:.6}",
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Range:             {:.6}",
                (max_val - min_val).to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn add_qualitymetrics<T>(
    report: &mut String,
    metrics: &ImageQualityMetrics<T>,
    format: ReportFormat,
) -> Result<(), std::fmt::Error>
where
    T: Float + ToPrimitive,
{
    match format {
        ReportFormat::Html => {
            writeln!(report, "<h2>Quality Metrics</h2>")?;
            writeln!(report, "<table>")?;
            writeln!(report, "<tr><th>Metric</th><th>Value</th></tr>")?;
            writeln!(
                report,
                "<tr><td>PSNR</td><td class='metric-value'>{:.3} dB</td></tr>",
                metrics.psnr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>SSIM</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.ssim.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>MSE</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.mse.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>SNR</td><td class='metric-value'>{:.3} dB</td></tr>",
                metrics.snr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Entropy</td><td class='metric-value'>{:.3} bits</td></tr>",
                metrics.entropy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Sharpness</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.sharpness.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report, "</table>")?;
        }
        ReportFormat::Markdown => {
            writeln!(report, "## Quality Metrics")?;
            writeln!(report)?;
            writeln!(report, "| Metric | Value |")?;
            writeln!(report, "|--------|-------|")?;
            writeln!(
                report,
                "| PSNR | {:.3} dB |",
                metrics.psnr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| SSIM | {:.6} |",
                metrics.ssim.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| MSE | {:.6} |",
                metrics.mse.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| SNR | {:.3} dB |",
                metrics.snr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Entropy | {:.3} bits |",
                metrics.entropy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Sharpness | {:.6} |",
                metrics.sharpness.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
        ReportFormat::Text => {
            writeln!(report, "Quality Metrics")?;
            writeln!(report, "---------------")?;
            writeln!(
                report,
                "PSNR:      {:.3} dB",
                metrics.psnr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "SSIM:      {:.6}",
                metrics.ssim.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "MSE:       {:.6}",
                metrics.mse.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "SNR:       {:.3} dB",
                metrics.snr.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Entropy:   {:.3} bits",
                metrics.entropy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Sharpness: {:.6}",
                metrics.sharpness.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn addtexturemetrics<T>(
    report: &mut String,
    metrics: &TextureMetrics<T>,
    format: ReportFormat,
) -> Result<(), std::fmt::Error>
where
    T: Float + ToPrimitive,
{
    match format {
        ReportFormat::Html => {
            writeln!(report, "<h2>Texture Analysis</h2>")?;
            writeln!(report, "<table>")?;
            writeln!(report, "<tr><th>Metric</th><th>Value</th></tr>")?;
            writeln!(
                report,
                "<tr><td>GLCM Contrast</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.glcm_contrast.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>GLCM Homogeneity</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.glcm_homogeneity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>GLCM Energy</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.glcm_energy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>LBP Uniformity</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.lbp_uniformity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Gabor Mean</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.gabor_mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Gabor Std</td><td class='metric-value'>{:.6}</td></tr>",
                metrics.gabor_std.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "<tr><td>Fractal Dimension</td><td class='metric-value'>{:.3}</td></tr>",
                metrics.fractal_dimension.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report, "</table>")?;
        }
        ReportFormat::Markdown => {
            writeln!(report, "## Texture Analysis")?;
            writeln!(report)?;
            writeln!(report, "| Metric | Value |")?;
            writeln!(report, "|--------|-------|")?;
            writeln!(
                report,
                "| GLCM Contrast | {:.6} |",
                metrics.glcm_contrast.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| GLCM Homogeneity | {:.6} |",
                metrics.glcm_homogeneity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| GLCM Energy | {:.6} |",
                metrics.glcm_energy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| LBP Uniformity | {:.6} |",
                metrics.lbp_uniformity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Gabor Mean | {:.6} |",
                metrics.gabor_mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Gabor Std | {:.6} |",
                metrics.gabor_std.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "| Fractal Dimension | {:.3} |",
                metrics.fractal_dimension.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
        ReportFormat::Text => {
            writeln!(report, "Texture Analysis")?;
            writeln!(report, "----------------")?;
            writeln!(
                report,
                "GLCM Contrast:     {:.6}",
                metrics.glcm_contrast.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "GLCM Homogeneity:  {:.6}",
                metrics.glcm_homogeneity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "GLCM Energy:       {:.6}",
                metrics.glcm_energy.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "LBP Uniformity:    {:.6}",
                metrics.lbp_uniformity.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Gabor Mean:        {:.6}",
                metrics.gabor_mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Gabor Std:         {:.6}",
                metrics.gabor_std.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                report,
                "Fractal Dimension: {:.3}",
                metrics.fractal_dimension.to_f64().unwrap_or(0.0)
            )?;
            writeln!(report)?;
        }
    }
    Ok(())
}

// Color map implementations
#[allow(dead_code)]
fn jet_colormap(t: f64) -> RgbColor {
    let r = (1.5 - 4.0 * (t - 0.75).abs()).max(0.0).min(1.0);
    let g = (1.5 - 4.0 * (t - 0.5).abs()).max(0.0).min(1.0);
    let b = (1.5 - 4.0 * (t - 0.25).abs()).max(0.0).min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[allow(dead_code)]
fn viridis_colormap(t: f64) -> RgbColor {
    // Simplified viridis approximation
    let r = (0.267 + 0.005 * t + 2.817 * t * t - 2.088 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (-0.040 + 1.416 * t - 0.376 * t * t).max(0.0).min(1.0);
    let b = (0.329 - 0.327 * t + 2.209 * t * t - 1.211 * t * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[allow(dead_code)]
fn plasma_colormap(t: f64) -> RgbColor {
    // Simplified plasma approximation
    let r = (0.054 + 2.192 * t + 0.063 * t * t - 1.309 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (0.230 * t + 1.207 * t * t - 0.437 * t * t * t)
        .max(0.0)
        .min(1.0);
    let b = (0.847 - 0.057 * t + 0.478 * t * t - 1.268 * t * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[allow(dead_code)]
fn inferno_colormap(t: f64) -> RgbColor {
    // Simplified inferno approximation
    let r = (0.077 + 2.081 * t + 0.866 * t * t - 1.024 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (t * t * (1.842 - 0.842 * t)).max(0.0).min(1.0);
    let b = (1.777 * t * t * t * t - 1.777 * t * t * t + 0.777 * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[allow(dead_code)]
fn hot_colormap(t: f64) -> RgbColor {
    let r = (3.0 * t).min(1.0);
    let g = (3.0 * t - 1.0).max(0.0).min(1.0);
    let b = (3.0 * t - 2.0).max(0.0).min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[allow(dead_code)]
fn cool_colormap(t: f64) -> RgbColor {
    RgbColor::new(((1.0 - t) * 255.0) as u8, (t * 255.0) as u8, 255)
}

#[allow(dead_code)]
fn spring_colormap(t: f64) -> RgbColor {
    RgbColor::new(255, (t * 255.0) as u8, ((1.0 - t) * 255.0) as u8)
}

#[allow(dead_code)]
fn summer_colormap(t: f64) -> RgbColor {
    RgbColor::new(
        (t * 255.0) as u8,
        ((0.5 + 0.5 * t) * 255.0) as u8,
        (0.4 * 255.0) as u8,
    )
}

#[allow(dead_code)]
fn autumn_colormap(t: f64) -> RgbColor {
    RgbColor::new(255, (t * 255.0) as u8, 0)
}

#[allow(dead_code)]
fn winter_colormap(t: f64) -> RgbColor {
    RgbColor::new(0, (t * 255.0) as u8, ((1.0 - 0.5 * t) * 255.0) as u8)
}

/// Generate a 3D surface plot representation of a 2D array
#[allow(dead_code)]
pub fn plot_surface<T>(data: &ArrayView2<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let mut plot = String::new();

    // Find min and max values for scaling
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All _data values are the same".into(),
        ));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='surface-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='surface-container'>")?;

            // Create a simplified 3D representation using CSS transforms
            let step_x = width.max(1) / (config.width / 20).max(1);
            let step_y = height.max(1) / (config.height / 20).max(1);

            for i in (0..height).step_by(step_y) {
                for j in (0..width).step_by(step_x) {
                    let value = data[[i, j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);
                    let z_height = normalized * 100.0; // Scale to percentage

                    let x_pos = (j as f64 / width as f64) * config.width as f64;
                    let y_pos = (i as f64 / height as f64) * config.height as f64;

                    // Color based on height
                    let colormap = create_colormap(config.colormap, 256);
                    let color_idx = (normalized * 255.0) as usize;
                    let color = colormap.get(color_idx).unwrap_or(&colormap[0]);

                    writeln!(
                        &mut plot,
                        "<div class='surface-point' style='left: {:.1}px; top: {:.1}px; height: {:.1}%; background-color: {};'></div>",
                        x_pos, y_pos, z_height, color.to_hex()
                    )?;
                }
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='surface-info'>")?;
            writeln!(
                &mut plot,
                "<p>Value range: [{:.3}, {:.3}]</p>",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (3D Surface)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;
            writeln!(&mut plot, "3D Surface Plot of {}×{} _data", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // Simple ASCII art representation
            let ascii_height = 20;
            let ascii_width = 60;
            for i in 0..ascii_height {
                for j in 0..ascii_width {
                    let data_i = (i * height) / ascii_height;
                    let data_j = (j * width) / ascii_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let char = match (normalized * 10.0) as u32 {
                        0..=1 => ' ',
                        2..=3 => '.',
                        4..=5 => ':',
                        6..=7 => '+',
                        8..=9 => '*',
                        _ => '#',
                    };
                    write!(&mut plot, "{}", char)?;
                }
                writeln!(&mut plot)?;
            }

            writeln!(&mut plot, "```")?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (3D Surface)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 13))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Data dimensions: {}×{}", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;
        }
    }

    Ok(plot)
}

/// Generate a contour plot representation of a 2D array
#[allow(dead_code)]
pub fn plot_contour<T>(
    data: &ArrayView2<T>,
    num_levels: usize,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let mut plot = String::new();

    // Find min and max values for level calculation
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All data values are the same".into(),
        ));
    }

    // Calculate contour _levels
    let mut levels = Vec::new();
    for i in 0..num_levels {
        let t = i as f64 / (num_levels - 1) as f64;
        let level = min_val + (max_val - min_val) * safe_f64_to_float::<T>(t)?;
        levels.push(level);
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='contour-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Simple contour approximation by drawing level sets
            for (level_idx, &level) in levels.iter().enumerate() {
                let color_intensity = (level_idx as f64 / num_levels as f64 * 255.0) as u8;
                let color = format!(
                    "rgb({}, {}, {})",
                    color_intensity,
                    100,
                    255 - color_intensity
                );

                // Find points close to this level
                for i in 0..height - 1 {
                    for j in 0..width - 1 {
                        let val = data[[i, j]];
                        let threshold = (max_val - min_val) * safe_f64_to_float::<T>(0.02)?; // 2% tolerance

                        if (val - level).abs() < threshold {
                            let x = (j as f64 / width as f64) * config.width as f64;
                            let y = (i as f64 / height as f64) * config.height as f64;

                            writeln!(
                                &mut plot,
                                "<circle cx='{:.1}' cy='{:.1}' r='1' fill='{}' opacity='0.7'/>",
                                x, y, color
                            )?;
                        }
                    }
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "<div class='contour-legend'>")?;
            writeln!(&mut plot, "<h4>Contour Levels:</h4>")?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "<div>Level {}: {:.3}</div>",
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Contour Plot)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Contour _levels ({}):", num_levels)?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "- Level {}: {:.3}",
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
            writeln!(&mut plot)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Contour Plot)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 15))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Contour _levels ({}):", num_levels)?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "  Level {}: {:.3}",
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Generate a heatmap visualization of a 2D array
#[allow(dead_code)]
pub fn plot_heatmap<T>(data: &ArrayView2<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let mut plot = String::new();

    // Find min and max values for color scaling
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All _data values are the same".into(),
        ));
    }

    // Create colormap
    let colormap = create_colormap(config.colormap, 256);

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='heatmap-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='heatmap-container'>")?;

            let cell_width = config.width as f64 / width as f64;
            let cell_height = config.height as f64 / height as f64;

            for i in 0..height {
                for j in 0..width {
                    let value = data[[i, j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);
                    let color_idx = (normalized * 255.0) as usize;
                    let color = colormap.get(color_idx).unwrap_or(&colormap[0]);

                    let x = j as f64 * cell_width;
                    let y = i as f64 * cell_height;

                    writeln!(
                        &mut plot,
                        "<rect x='{:.1}' y='{:.1}' width='{:.1}' height='{:.1}' fill='{}' title='{:.3}'/>",
                        x, y, cell_width, cell_height, color.to_hex(), value.to_f64().unwrap_or(0.0)
                    )?;
                }
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='heatmap-colorbar'>")?;
            writeln!(
                &mut plot,
                "<div>Min: {:.3}</div>",
                min_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                &mut plot,
                "<div>Max: {:.3}</div>",
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Heatmap)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;
            writeln!(&mut plot, "Heatmap of {}×{} _data", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // ASCII heatmap representation
            let display_height = height.min(20);
            let display_width = width.min(60);

            for i in 0..display_height {
                for j in 0..display_width {
                    let data_i = (i * height) / display_height;
                    let data_j = (j * width) / display_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let char = match (normalized * 9.0) as u32 {
                        0 => ' ',
                        1 => '.',
                        2 => ':',
                        3 => '-',
                        4 => '=',
                        5 => '+',
                        6 => '*',
                        7 => '#',
                        8 => '@',
                        _ => '█',
                    };
                    write!(&mut plot, "{}", char)?;
                }
                writeln!(&mut plot)?;
            }

            writeln!(&mut plot, "```")?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Heatmap)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 10))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Data dimensions: {}×{}", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
        }
    }

    Ok(plot)
}

/// Create an image montage/grid from multiple 2D arrays
#[allow(dead_code)]
pub fn createimage_montage<T>(
    images: &[ArrayView2<T>],
    grid_cols: usize,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if images.is_empty() {
        return Err(NdimageError::InvalidInput("No images provided".into()));
    }

    if grid_cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Grid columns must be positive".into(),
        ));
    }

    let mut plot = String::new();
    let grid_rows = (images.len() + grid_cols - 1) / grid_cols;

    // Find global min/max for consistent scaling
    let mut global_min = T::infinity();
    let mut global_max = T::neg_infinity();

    for image in images {
        let min_val = image.iter().cloned().fold(T::infinity(), T::min);
        let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);
        global_min = global_min.min(min_val);
        global_max = global_max.max(max_val);
    }

    if global_max <= global_min {
        return Err(NdimageError::InvalidInput(
            "All image values are the same".into(),
        ));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='image-montage'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='montage-grid' style='display: grid; grid-template-columns: repeat({}, 1fr); gap: 10px;'>", grid_cols)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                writeln!(&mut plot, "<div class='montage-cell'>")?;
                writeln!(&mut plot, "<h4>Image {}</h4>", idx + 1)?;
                writeln!(
                    &mut plot,
                    "<div class='image-data' data-width='{}' data-height='{}'>",
                    width, height
                )?;

                // Simple representation - would need actual image rendering in practice
                writeln!(&mut plot, "<p>{}×{} array</p>", height, width)?;
                writeln!(
                    &mut plot,
                    "<p>Range: [{:.3}, {:.3}]</p>",
                    image
                        .iter()
                        .cloned()
                        .fold(T::infinity(), T::min)
                        .to_f64()
                        .unwrap_or(0.0),
                    image
                        .iter()
                        .cloned()
                        .fold(T::neg_infinity(), T::max)
                        .to_f64()
                        .unwrap_or(0.0)
                )?;

                writeln!(&mut plot, "</div>")?;
                writeln!(&mut plot, "</div>")?;
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='montage-info'>")?;
            writeln!(
                &mut plot,
                "<p>Global range: [{:.3}, {:.3}]</p>",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                &mut plot,
                "<p>Grid: {} rows × {} columns</p>",
                grid_rows, grid_cols
            )?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Image Montage)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "Grid layout: {} rows × {} columns",
                grid_rows, grid_cols
            )?;
            writeln!(
                &mut plot,
                "Global value range: [{:.3}, {:.3}]",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                let min_val = image.iter().cloned().fold(T::infinity(), T::min);
                let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

                writeln!(&mut plot, "### Image {}", idx + 1)?;
                writeln!(&mut plot, "- Dimensions: {}×{}", height, width)?;
                writeln!(
                    &mut plot,
                    "- Value range: [{:.3}, {:.3}]",
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
                writeln!(&mut plot)?;
            }
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Image Montage)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 16))?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "Grid layout: {} rows × {} columns",
                grid_rows, grid_cols
            )?;
            writeln!(
                &mut plot,
                "Global value range: [{:.3}, {:.3}]",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                let min_val = image.iter().cloned().fold(T::infinity(), T::min);
                let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

                writeln!(
                    &mut plot,
                    "Image {}: {}×{}, range [{:.3}, {:.3}]",
                    idx + 1,
                    height,
                    width,
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Generate a comparative statistical plot for multiple datasets
#[allow(dead_code)]
pub fn plot_statistical_comparison<T>(
    datasets: &[(&str, ArrayView1<T>)],
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if datasets.is_empty() {
        return Err(NdimageError::InvalidInput("No datasets provided".into()));
    }

    let mut plot = String::new();

    // Compute statistics for each dataset
    let mut stats = Vec::new();
    for (name, data) in datasets {
        if data.is_empty() {
            continue;
        }

        let mean = data.mean().unwrap_or(T::zero());
        let min_val = data.iter().cloned().fold(T::infinity(), T::min);
        let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
        let variance = data
            .mapv(|x| (x - mean) * (x - mean))
            .mean()
            .unwrap_or(T::zero());
        let std_dev = variance.sqrt();

        stats.push((name, mean, std_dev, min_val, max_val, data.len()));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='statistical-comparison'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<table class='stats-table'>")?;
            writeln!(&mut plot, "<tr><th>Dataset</th><th>Count</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>")?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "<tr><td>{}</td><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>",
                    name, count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }

            writeln!(&mut plot, "</table>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Statistical Comparison)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "| Dataset | Count | Mean | Std Dev | Min | Max |"
            )?;
            writeln!(
                &mut plot,
                "|---------|-------|------|---------|-----|-----|"
            )?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |",
                    name,
                    count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
            writeln!(&mut plot)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Statistical Comparison)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 25))?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "{:<15} {:>8} {:>10} {:>10} {:>10} {:>10}",
                "Dataset", "Count", "Mean", "Std Dev", "Min", "Max"
            )?;
            writeln!(&mut plot, "{}", "-".repeat(75))?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "{:<15} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    name,
                    count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Export utilities for saving visualization output to files
pub mod export {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Export configuration for file output
    #[derive(Debug, Clone)]
    pub struct ExportConfig {
        /// Output file path
        pub output_path: String,
        /// Image quality (for compressed formats)
        pub quality: Option<u8>,
        /// DPI for vector formats
        pub dpi: Option<u32>,
        /// Whether to include metadata
        pub include_metadata: bool,
    }

    impl Default for ExportConfig {
        fn default() -> Self {
            Self {
                output_path: "output.html".to_string(),
                quality: Some(95),
                dpi: Some(300),
                include_metadata: true,
            }
        }
    }

    /// Save a generated plot to file
    pub fn save_plot(content: &str, config: &ExportConfig) -> NdimageResult<()> {
        let path = Path::new(&config.output_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                NdimageError::ComputationError(format!("Failed to create directory: {}", e))
            })?;
        }

        // Add metadata if requested
        let mut output_content = content.to_string();
        if config.include_metadata {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let metadata = format!(
                "\n<!-- Generated by scirs2-ndimage visualization module at {} -->\n",
                timestamp
            );

            if content.contains("</html>") {
                output_content = content.replace("</html>", &format!("{}</html>", metadata));
            } else if content.contains("# ") {
                output_content = format!(
                    "{}\n{}",
                    content,
                    metadata.replace("<!--", "").replace("-->", "")
                );
            } else {
                output_content = format!(
                    "{}\n{}",
                    content,
                    metadata.replace("<!--", "").replace("-->", "")
                );
            }
        }

        fs::write(path, output_content)
            .map_err(|e| NdimageError::ComputationError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Generate and save a comprehensive analysis report
    pub fn export_analysis_report<T>(
        image: &ArrayView2<T>,
        qualitymetrics: Option<&crate::analysis::ImageQualityMetrics<T>>,
        texturemetrics: Option<&crate::analysis::TextureMetrics<T>>,
        output_path: &str,
        format: ReportFormat,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        let config = ReportConfig {
            title: "Comprehensive Image Analysis Report".to_string(),
            author: "SciRS2 NDImage".to_string(),
            format,
            ..ReportConfig::default()
        };

        let report = generate_report(image, qualitymetrics, texturemetrics, &config)?;

        let export_config = ExportConfig {
            output_path: output_path.to_string(),
            ..ExportConfig::default()
        };

        save_plot(&report, &export_config)?;
        Ok(())
    }
}

/// Advanced visualization utilities
pub mod advanced {
    use super::*;

    /// Create an interactive HTML visualization with multiple views
    pub fn create_interactive_visualization<T>(
        image: &ArrayView2<T>,
        title: &str,
    ) -> NdimageResult<String>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        let mut html = String::new();

        writeln!(&mut html, "<!DOCTYPE html>")?;
        writeln!(&mut html, "<html><head>")?;
        writeln!(&mut html, "<title>{}</title>", title)?;
        writeln!(&mut html, "<style>")?;
        writeln!(
            &mut html,
            r#"
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .visualization-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .plot-panel {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; min-width: 300px; }}
            .plot-title {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
            .controls {{ margin-bottom: 15px; }}
            .control-group {{ margin-bottom: 10px; }}
            label {{ display: inline-block; width: 100px; }}
            select, input {{ margin-left: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
            .stat-item {{ background: #f5f5f5; padding: 8px; border-radius: 3px; }}
            .heatmap-container {{ position: relative; width: 400px; height: 300px; }}
            .colorbar {{ width: 20px; height: 300px; background: linear-gradient(to top, blue, cyan, yellow, red); }}
        "#
        )?;
        writeln!(&mut html, "</style>")?;
        writeln!(&mut html, "<script>")?;
        writeln!(
            &mut html,
            r#"
            function updateVisualization() {{
                const colormap = document.getElementById('colormap').value;
                const plotType = document.getElementById('plotType').value;
                // Update visualization based on controls
                console.log('Updating visualization:', colormap, plotType);
            }}
            
            function exportView() {{
                const content = document.getElementById('main-content').innerHTML;
                const blob = new Blob([content], {{ type: 'text/html' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'visualization.html';
                a.click();
                URL.revokeObjectURL(url);
            }}
        "#
        )?;
        writeln!(&mut html, "</script>")?;
        writeln!(&mut html, "</head><body>")?;

        writeln!(&mut html, "<h1>{}</h1>", title)?;

        // Controls panel
        writeln!(&mut html, "<div class='controls'>")?;
        writeln!(&mut html, "<div class='control-group'>")?;
        writeln!(&mut html, "<label>Color Map:</label>")?;
        writeln!(
            &mut html,
            r#"<select id="colormap" onchange="updateVisualization()">"#
        )?;
        writeln!(&mut html, "<option value='viridis'>Viridis</option>")?;
        writeln!(&mut html, "<option value='plasma'>Plasma</option>")?;
        writeln!(&mut html, "<option value='jet'>Jet</option>")?;
        writeln!(&mut html, "<option value='hot'>Hot</option>")?;
        writeln!(&mut html, "</select>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(&mut html, "<div class='control-group'>")?;
        writeln!(&mut html, "<label>Plot Type:</label>")?;
        writeln!(
            &mut html,
            r#"<select id="plotType" onchange="updateVisualization()">"#
        )?;
        writeln!(&mut html, "<option value='heatmap'>Heatmap</option>")?;
        writeln!(&mut html, "<option value='contour'>Contour</option>")?;
        writeln!(&mut html, "<option value='surface'>3D Surface</option>")?;
        writeln!(&mut html, "</select>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(
            &mut html,
            r#"<button onclick="exportView()">Export View</button>"#
        )?;
        writeln!(&mut html, "</div>")?;

        writeln!(
            &mut html,
            "<div id='main-content' class='visualization-container'>"
        )?;

        // Statistics panel
        let (height, width) = image.dim();
        let mean = image.mean().unwrap_or(T::zero());
        let min_val = image.iter().cloned().fold(T::infinity(), T::min);
        let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

        writeln!(&mut html, "<div class='plot-panel'>")?;
        writeln!(&mut html, "<div class='plot-title'>Image Statistics</div>")?;
        writeln!(&mut html, "<div class='stats-grid'>")?;
        writeln!(&mut html, "<div class='stat-item'>Width: {}</div>", width)?;
        writeln!(&mut html, "<div class='stat-item'>Height: {}</div>", height)?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Mean: {:.4}</div>",
            mean.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Min: {:.4}</div>",
            min_val.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Max: {:.4}</div>",
            max_val.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Pixels: {}</div>",
            width * height
        )?;
        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</div>")?;

        // Heatmap panel
        writeln!(&mut html, "<div class='plot-panel'>")?;
        writeln!(&mut html, "<div class='plot-title'>Heatmap View</div>")?;
        writeln!(&mut html, "<div class='heatmap-container'>")?;

        // Generate a simplified heatmap representation
        let config = PlotConfig {
            format: ReportFormat::Html,
            colormap: ColorMap::Viridis,
            title: "Interactive Heatmap".to_string(),
            ..PlotConfig::default()
        };

        let heatmap = plot_heatmap(image, &config)?;
        writeln!(&mut html, "{}", heatmap)?;

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</body></html>")?;

        Ok(html)
    }

    /// Create a comparison visualization between multiple images
    pub fn create_comparison_view<T>(
        images: &[(&str, ArrayView2<T>)],
        title: &str,
    ) -> NdimageResult<String>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        if images.is_empty() {
            return Err(NdimageError::InvalidInput(
                "No images provided for comparison".into(),
            ));
        }

        let mut html = String::new();

        writeln!(&mut html, "<!DOCTYPE html>")?;
        writeln!(&mut html, "<html><head>")?;
        writeln!(&mut html, "<title>{}</title>", title)?;
        writeln!(&mut html, "<style>")?;
        writeln!(
            &mut html,
            r#"
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-panel {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
            .image-title {{ font-weight: bold; margin-bottom: 10px; color: #333; text-align: center; }}
            .image-stats {{ background: #f9f9f9; padding: 10px; margin-top: 10px; border-radius: 3px; }}
            .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
            .difference-highlight {{ background: #ffe6e6; }}
        "#
        )?;
        writeln!(&mut html, "</style>")?;
        writeln!(&mut html, "</head><body>")?;

        writeln!(&mut html, "<h1>{}</h1>", title)?;
        writeln!(&mut html, "<div class='comparison-grid'>")?;

        for (name, image) in images {
            writeln!(&mut html, "<div class='image-panel'>")?;
            writeln!(&mut html, "<div class='image-title'>{}</div>", name)?;

            // Generate heatmap for this image
            let config = PlotConfig {
                format: ReportFormat::Html,
                colormap: ColorMap::Viridis,
                title: name.to_string(),
                width: 250,
                height: 200,
                ..PlotConfig::default()
            };

            let heatmap = plot_heatmap(image, &config)?;
            writeln!(&mut html, "{}", heatmap)?;

            // Add statistics
            let (height, width) = image.dim();
            let mean = image.mean().unwrap_or(T::zero());
            let min_val = image.iter().cloned().fold(T::infinity(), T::min);
            let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

            writeln!(&mut html, "<div class='image-stats'>")?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Dimensions:</span><span>{}×{}</span></div>",
                height, width
            )?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Mean:</span><span>{:.4}</span></div>",
                mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Range:</span><span>[{:.3}, {:.3}]</span></div>",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut html, "</div>")?;

            writeln!(&mut html, "</div>")?;
        }

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</body></html>")?;

        Ok(html)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_create_colormap() {
        let colors = create_colormap(ColorMap::Viridis, 10);
        assert_eq!(colors.len(), 10);

        // First color should be dark, last should be bright
        assert!(
            colors[0].r < colors[9].r || colors[0].g < colors[9].g || colors[0].b < colors[9].b
        );
    }

    #[test]
    fn test_plot_histogram() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = PlotConfig {
            format: ReportFormat::Text,
            ..PlotConfig::default()
        };

        let plot =
            plot_histogram(&data.view(), &config).expect("plot_histogram should succeed for test");
        assert!(plot.contains("Image Analysis Plot"));
    }

    #[test]
    fn test_plot_profile() {
        let x_data = array![1.0, 2.0, 3.0, 4.0];
        let y_data = array![1.0, 4.0, 9.0, 16.0];
        let config = PlotConfig {
            format: ReportFormat::Text,
            ..PlotConfig::default()
        };

        let plot = plot_profile(&x_data.view(), &y_data.view(), &config)
            .expect("plot_profile should succeed for test");
        assert!(plot.contains("Image Analysis Plot"));
    }

    #[test]
    fn test_generate_report() {
        let image = array![[1.0, 2.0], [3.0, 4.0]];
        let config = ReportConfig {
            format: ReportFormat::Text,
            ..ReportConfig::default()
        };

        let report = generate_report(&image.view(), None, None, &config)
            .expect("generate_report should succeed for test");
        assert!(report.contains("Image Analysis Report"));
        assert!(report.contains("Image Information"));
    }

    #[test]
    fn test_rgb_color() {
        let color = RgbColor::new(255, 128, 64);
        assert_eq!(color.to_hex(), "#ff8040");
    }

    #[test]
    fn test_export_config() {
        let config = export::ExportConfig::default();
        assert_eq!(config.output_path, "output.html");
        assert_eq!(config.quality, Some(95));
        assert!(config.include_metadata);
    }

    #[test]
    fn test_interactive_visualization() {
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let html = advanced::create_interactive_visualization(&image.view(), "Test Visualization")
            .expect("Interactive visualization should succeed");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Visualization"));
        assert!(html.contains("colormap"));
        assert!(html.contains("Image Statistics"));
        assert!(html.contains("Width: 3"));
        assert!(html.contains("Height: 2"));
    }

    #[test]
    fn test_comparison_view() {
        let image1 = array![[1.0, 2.0], [3.0, 4.0]];
        let image2 = array![[2.0, 3.0], [4.0, 5.0]];

        let images = vec![("Original", image1.view()), ("Processed", image2.view())];

        let html = advanced::create_comparison_view(&images, "Image Comparison")
            .expect("Comparison view should succeed");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Image Comparison"));
        assert!(html.contains("Original"));
        assert!(html.contains("Processed"));
        assert!(html.contains("comparison-grid"));
    }

    #[test]
    fn test_empty_comparison_view() {
        let images: Vec<(&str, ndarray::ArrayView2<f64>)> = vec![];

        let result = advanced::create_comparison_view(&images, "Empty Comparison");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No images provided"));
    }
}
