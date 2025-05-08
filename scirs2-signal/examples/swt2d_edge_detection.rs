use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::swt2d::swt2d_decompose;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("2D Stationary Wavelet Transform for Edge Detection");
    println!("--------------------------------------------------");

    // Create a test image with some edges (circle in a square)
    let size = 64;
    let circle_radius = 20.0;
    let circle_center = (size as f64 / 2.0, size as f64 / 2.0);

    let mut image = Array2::zeros((size, size));

    // Create a circle in the center and a square border
    for i in 0..size {
        for j in 0..size {
            // Square border
            if i < 5 || i >= size - 5 || j < 5 || j >= size - 5 {
                image[[i, j]] = 1.0;
            }

            // Circle in the center
            let x = j as f64 - circle_center.1;
            let y = i as f64 - circle_center.0;
            let distance = (x * x + y * y).sqrt();

            if distance <= circle_radius {
                image[[i, j]] = 1.0;
            }
        }
    }

    println!(
        "Created test image ({}x{}) with circle and square",
        size, size
    );

    // Add noise to the image
    let mut noisy_image = image.clone();
    let noise_level = 0.1;

    for i in 0..size {
        for j in 0..size {
            noisy_image[[i, j]] += noise_level * (rand::random::<f64>() - 0.5);
        }
    }

    println!("Added noise with level {}", noise_level);

    // Edge detection using 2D SWT
    println!("Performing edge detection using 2D SWT...");

    // Edge detection on clean image
    let edges_clean = detect_edges_swt(&image, Wavelet::DB(4), 2)?;
    // Edge detection on noisy image (not used in this example, but computed for demonstration)
    let _edges_noisy = detect_edges_swt(&noisy_image, Wavelet::DB(4), 2)?;

    println!("Edge detection completed");

    // Optionally save images to file for visualization
    println!("To visualize the results, save the arrays to image files");
    println!("The edge detection emphasizes the boundary of the circle and square");

    // Print some statistics
    let max_edge_value = edges_clean.iter().fold(0.0, |a: f64, &b| a.max(b));
    println!("Maximum edge response: {:.4}", max_edge_value);

    let ratio = count_nonzero(&edges_clean, 0.1 * max_edge_value) as f64 / (size * size) as f64;
    println!(
        "Percentage of pixels detected as edges: {:.2}%",
        ratio * 100.0
    );

    Ok(())
}

// Edge detection function using 2D SWT
fn detect_edges_swt<T>(
    image: &Array2<T>,
    wavelet: Wavelet,
    level: usize,
) -> Result<Array2<f64>, Box<dyn std::error::Error>>
where
    T: num_traits::Float + num_traits::NumCast + std::fmt::Debug,
{
    // Step 1: Decompose the image using 2D SWT
    let decomp = swt2d_decompose(image, wavelet, level, None)?;

    // Step 2: Use detail coefficients for edge detection
    // Combine horizontal and vertical detail coefficients
    let (rows, cols) = decomp.detail_h.dim();
    let mut edge_map = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            // Combine horizontal and vertical details to detect edges in all directions
            // We square the coefficients to emphasize stronger edges
            edge_map[[i, j]] =
                (decomp.detail_h[[i, j]].powi(2) + decomp.detail_v[[i, j]].powi(2)).sqrt();
        }
    }

    // Normalize the edge map
    let max_value = edge_map.iter().fold(0.0, |a: f64, &b| a.max(b));
    if max_value > 0.0 {
        for value in edge_map.iter_mut() {
            *value /= max_value;
        }
    }

    Ok(edge_map)
}

// Helper function to count non-zero (above threshold) elements in an array
fn count_nonzero(array: &Array2<f64>, threshold: f64) -> usize {
    array.iter().filter(|&&x| x > threshold).count()
}
