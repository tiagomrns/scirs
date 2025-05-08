use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::swt2d::{iswt2d, swt2d, swt2d_decompose, swt2d_reconstruct};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("2D Stationary Wavelet Transform Example");
    println!("---------------------------------------");

    // Create a simple test image (8x8 gradient)
    let mut image = Array2::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            image[[i, j]] = (i * j) as f64;
        }
    }

    println!("Original image (8x8):");
    print_array2(&image);

    // Single-level decomposition
    println!("\nPerforming single-level 2D SWT decomposition with Haar wavelet...");
    let decomp = swt2d_decompose(&image, Wavelet::Haar, 1, None)?;

    println!("\nApproximation coefficients (LL):");
    print_array2(&decomp.approx);

    println!("\nHorizontal detail coefficients (LH):");
    print_array2(&decomp.detail_h);

    println!("\nVertical detail coefficients (HL):");
    print_array2(&decomp.detail_v);

    println!("\nDiagonal detail coefficients (HH):");
    print_array2(&decomp.detail_d);

    // Reconstruct from single-level decomposition
    println!("\nReconstructing from single-level decomposition...");
    let reconstructed = swt2d_reconstruct(&decomp, Wavelet::Haar, 1, None)?;

    println!("\nReconstructed image:");
    print_array2(&reconstructed);

    // Compute reconstruction error
    let mut max_error = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let error = (image[[i, j]] - reconstructed[[i, j]]).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }
    println!("\nMaximum reconstruction error: {:.6e}", max_error);

    // Multi-level decomposition
    println!("\nPerforming multi-level 2D SWT (3 levels) with Haar wavelet...");
    let decompositions = swt2d(&image, Wavelet::Haar, 3, None)?;

    println!("\nNumber of decomposition levels: {}", decompositions.len());

    for (level, decomp) in decompositions.iter().enumerate() {
        println!("\nLevel {} approximation coefficients:", level + 1);
        print_compact_array2(&decomp.approx);
    }

    // Reconstruct from multi-level decomposition
    println!("\nReconstructing from multi-level decomposition...");
    let multi_reconstructed = iswt2d(&decompositions, Wavelet::Haar, None)?;

    println!("\nReconstructed image from multi-level SWT:");
    print_array2(&multi_reconstructed);

    // Compute multi-level reconstruction error
    let mut max_multi_error = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let error = (image[[i, j]] - multi_reconstructed[[i, j]]).abs();
            if error > max_multi_error {
                max_multi_error = error;
            }
        }
    }
    println!(
        "\nMaximum multi-level reconstruction error: {:.6e}",
        max_multi_error
    );

    // Demonstration of denoising by thresholding wavelet coefficients
    println!("\nDemonstration of denoising by coefficient thresholding:");

    // Create a noisy version of the image
    let mut noisy_image = image.clone();
    let noise_level = 5.0;

    // Add random noise
    for i in 0..8 {
        for j in 0..8 {
            noisy_image[[i, j]] += noise_level * (rand::random::<f64>() - 0.5);
        }
    }

    println!("\nNoisy image:");
    print_array2(&noisy_image);

    // Decompose noisy image
    let noisy_decomp = swt2d(&noisy_image, Wavelet::Haar, 3, None)?;

    // Threshold the coefficients (simple hard thresholding)
    let threshold = 2.5;
    let mut denoised_decomp = Vec::with_capacity(noisy_decomp.len());

    for level_decomp in &noisy_decomp {
        // Create a copy of the level decomposition
        let mut thresholded = level_decomp.clone();

        // Threshold detail coefficients (leave approximation untouched)
        threshold_array(&mut thresholded.detail_h, threshold);
        threshold_array(&mut thresholded.detail_v, threshold);
        threshold_array(&mut thresholded.detail_d, threshold);

        denoised_decomp.push(thresholded);
    }

    // Reconstruct from thresholded coefficients
    let denoised_image = iswt2d(&denoised_decomp, Wavelet::Haar, None)?;

    println!("\nDenoised image:");
    print_array2(&denoised_image);

    // Compute denoising error relative to original image
    let mut mse_noisy = 0.0;
    let mut mse_denoised = 0.0;

    for i in 0..8 {
        for j in 0..8 {
            let noisy_error = image[[i, j]] - noisy_image[[i, j]];
            let denoised_error = image[[i, j]] - denoised_image[[i, j]];

            mse_noisy += noisy_error * noisy_error;
            mse_denoised += denoised_error * denoised_error;
        }
    }

    mse_noisy /= 64.0; // 8x8 = 64 pixels
    mse_denoised /= 64.0;

    println!("\nMean Squared Error:");
    println!("  Noisy image: {:.6}", mse_noisy);
    println!("  Denoised image: {:.6}", mse_denoised);
    println!(
        "  Improvement: {:.1}%",
        100.0 * (mse_noisy - mse_denoised) / mse_noisy
    );

    Ok(())
}

// Helper function to print a 2D array
fn print_array2(array: &Array2<f64>) {
    let (rows, cols) = array.dim();
    for i in 0..rows {
        for j in 0..cols {
            print!("{:8.2} ", array[[i, j]]);
        }
        println!();
    }
}

// Helper function to print a 2D array in compact form
fn print_compact_array2(array: &Array2<f64>) {
    let (rows, cols) = array.dim();
    // Only print the first few rows and columns
    let max_show = 4;
    let r_show = std::cmp::min(rows, max_show);
    let c_show = std::cmp::min(cols, max_show);

    for i in 0..r_show {
        for j in 0..c_show {
            print!("{:8.2} ", array[[i, j]]);
        }
        if cols > max_show {
            print!("...");
        }
        println!();
    }

    if rows > max_show {
        println!("...");
    }
}

// Helper function to apply hard thresholding to array elements
fn threshold_array(array: &mut Array2<f64>, threshold: f64) {
    for value in array.iter_mut() {
        if value.abs() < threshold {
            *value = 0.0;
        }
    }
}
