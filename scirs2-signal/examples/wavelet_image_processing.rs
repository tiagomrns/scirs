use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d_image::{compress_image, denoise_image, detect_edges, DenoisingMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Wavelet-based Image Processing Example");
    println!("======================================\n");

    // Create a simple test image (32x32) with a pattern and some noise
    println!("Creating test image...");
    let image = create_test_image(32, 32);

    // Create a simple pattern with circular intensity
    print_image_stats(&image, "Original image");

    // ---- IMAGE DENOISING ----
    println!("\n1. Image Denoising Example");
    println!("----------------------");

    // Apply different denoising methods
    println!("\nApplying VisuShrink denoising...");
    let denoised_visu = denoise_image(
        &image,
        Wavelet::DB(4),
        3,
        0.0,
        DenoisingMethod::VisuShrink,
        Some(true),
    )?;
    print_image_stats(&denoised_visu, "VisuShrink denoised image");

    println!("\nApplying BayesShrink denoising...");
    let denoised_bayes = denoise_image(
        &image,
        Wavelet::DB(4),
        3,
        0.0,
        DenoisingMethod::BayesShrink,
        Some(true),
    )?;
    print_image_stats(&denoised_bayes, "BayesShrink denoised image");

    println!("\nComparing different thresholding methods at fixed threshold (5.0)...");
    let denoised_hard = denoise_image(
        &image,
        Wavelet::DB(4),
        3,
        5.0,
        DenoisingMethod::Hard,
        Some(true),
    )?;
    print_image_stats(&denoised_hard, "Hard thresholding");

    let denoised_soft = denoise_image(
        &image,
        Wavelet::DB(4),
        3,
        5.0,
        DenoisingMethod::Soft,
        Some(true),
    )?;
    print_image_stats(&denoised_soft, "Soft thresholding");

    // ---- EDGE DETECTION ----
    println!("\n\n2. Edge Detection Example");
    println!("----------------------");

    // Create an image with sharp edges
    println!("\nCreating edge test image...");
    let edge_test_image = create_edge_test_image(32, 32);
    print_image_stats(&edge_test_image, "Edge test image");

    // Detect edges with different wavelets
    println!("\nDetecting edges with Haar wavelet...");
    let edges_haar = detect_edges(&edge_test_image, Wavelet::Haar, 1, Some(5.0))?;
    print_image_stats(&edges_haar, "Haar wavelet edges");

    println!("\nDetecting edges with DB2 wavelet...");
    let edges_db2 = detect_edges(&edge_test_image, Wavelet::DB(2), 1, Some(5.0))?;
    print_image_stats(&edges_db2, "DB2 wavelet edges");

    // ---- IMAGE COMPRESSION ----
    println!("\n\n3. Image Compression Example");
    println!("--------------------------");

    // Create a more complex image
    println!("\nCreating complex test image...");
    let complex_image = create_complex_image(64, 64);
    print_image_stats(&complex_image, "Complex test image");

    // Apply different compression ratios
    println!("\nApplying mild compression (25%)...");
    let (mild_compressed, mild_ratio) = compress_image(&complex_image, Wavelet::DB(4), 3, 0.25)?;
    print_image_stats(
        &mild_compressed,
        &format!("Compressed image (ratio: {:.1}%)", mild_ratio * 100.0),
    );

    println!("\nApplying moderate compression (50%)...");
    let (mod_compressed, mod_ratio) = compress_image(&complex_image, Wavelet::DB(4), 3, 0.50)?;
    print_image_stats(
        &mod_compressed,
        &format!("Compressed image (ratio: {:.1}%)", mod_ratio * 100.0),
    );

    println!("\nApplying high compression (75%)...");
    let (high_compressed, high_ratio) = compress_image(&complex_image, Wavelet::DB(4), 3, 0.75)?;
    print_image_stats(
        &high_compressed,
        &format!("Compressed image (ratio: {:.1}%)", high_ratio * 100.0),
    );

    // Calculate and print compression quality metrics
    println!("\nCompression Quality Metrics:");
    println!("---------------------------");
    println!(
        "Mild compression:  PSNR={:.2}dB, MSE={:.4}",
        calculate_psnr(&complex_image, &mild_compressed),
        calculate_mse(&complex_image, &mild_compressed)
    );
    println!(
        "Moderate compression: PSNR={:.2}dB, MSE={:.4}",
        calculate_psnr(&complex_image, &mod_compressed),
        calculate_mse(&complex_image, &mod_compressed)
    );
    println!(
        "High compression: PSNR={:.2}dB, MSE={:.4}",
        calculate_psnr(&complex_image, &high_compressed),
        calculate_mse(&complex_image, &high_compressed)
    );

    println!("\nExample complete!");
    Ok(())
}

// Helper function to create a test image
fn create_test_image(width: usize, height: usize) -> Array2<f64> {
    let mut image = Array2::zeros((height, width));

    // Create a circular intensity pattern
    let center_x = (width / 2) as f64;
    let center_y = (height / 2) as f64;
    let max_radius = (width.min(height) / 2) as f64;

    for y in 0..height {
        for x in 0..width {
            // Base pattern - circular gradient
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            let radius = (dx * dx + dy * dy).sqrt();
            let normalized_radius = radius / max_radius;
            let intensity = (1.0 - normalized_radius.min(1.0)) * 100.0;

            // Add some noise
            let noise = (x as f64 * 0.1).sin() * (y as f64 * 0.2).cos() * 10.0;

            image[[y, x]] = intensity + noise;
        }
    }

    image
}

// Helper function to create an image with edges
fn create_edge_test_image(width: usize, height: usize) -> Array2<f64> {
    let mut image = Array2::zeros((height, width));

    // Horizontal edge at 1/3 from top
    let h_edge = height / 3;

    // Vertical edge at 2/3 from left
    let v_edge = width * 2 / 3;

    for y in 0..height {
        for x in 0..width {
            if y >= h_edge {
                image[[y, x]] += 50.0;
            }
            if x <= v_edge {
                image[[y, x]] += 30.0;
            }

            // Add diagonal edge
            if x == y {
                image[[y, x]] += 20.0;
            }
        }
    }

    image
}

// Helper function to create a more complex image
fn create_complex_image(width: usize, height: usize) -> Array2<f64> {
    let mut image = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            // Multiple frequency components
            let val1 = (x as f64 * 0.2).sin() * (y as f64 * 0.1).sin() * 50.0;
            let val2 = (x as f64 * 0.05).cos() * (y as f64 * 0.05).cos() * 30.0;
            let val3 = ((x as f64 - width as f64 / 2.0).powi(2)
                + (y as f64 - height as f64 / 2.0).powi(2))
            .sqrt()
                * 0.1;

            image[[y, x]] = val1 + val2 + val3;
        }
    }

    image
}

// Helper function to print image statistics
fn print_image_stats(image: &Array2<f64>, label: &str) {
    let min_val = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut nonzero_count = 0;

    for &val in image.iter() {
        sum += val;
        sum_sq += val * val;
        if val != 0.0 {
            nonzero_count += 1;
        }
    }

    let mean = sum / (image.len() as f64);
    let variance = sum_sq / (image.len() as f64) - mean * mean;
    let std_dev = variance.sqrt();

    println!("{}:", label);
    println!("  Dimensions: {}x{}", image.dim().0, image.dim().1);
    println!("  Range: [{:.2}, {:.2}]", min_val, max_val);
    println!("  Mean: {:.2}, StdDev: {:.2}", mean, std_dev);
    println!(
        "  Non-zero values: {} ({:.1}%)",
        nonzero_count,
        100.0 * nonzero_count as f64 / image.len() as f64
    );
}

// Calculate mean squared error between two images
fn calculate_mse(original: &Array2<f64>, processed: &Array2<f64>) -> f64 {
    if original.shape() != processed.shape() {
        return f64::NAN;
    }

    let mut sum_squared_error = 0.0;
    let n = original.len() as f64;

    for (a, b) in original.iter().zip(processed.iter()) {
        let error = a - b;
        sum_squared_error += error * error;
    }

    sum_squared_error / n
}

// Calculate peak signal-to-noise ratio
fn calculate_psnr(original: &Array2<f64>, processed: &Array2<f64>) -> f64 {
    let mse = calculate_mse(original, processed);
    if mse <= 1e-10 {
        return f64::INFINITY; // Perfect match
    }

    // Find data range (max value - min value)
    let min_val = original.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = original.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let data_range = max_val - min_val;

    // PSNR formula: 20 * log10(MAX / sqrt(MSE))
    20.0 * (data_range / mse.sqrt()).log10()
}
