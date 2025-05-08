use ndarray::{s, Array2};
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d::{dwt2d_decompose, wavedec2};
use scirs2_signal::swt2d::swt2d_decompose;
use scirs2_signal::wavelet_vis::{
    arrange_coefficients_2d,
    // arrange_multilevel_coefficients_2d, // Will be used in a future refinement
    calculate_energy_2d,
    calculate_energy_swt2d,
    colormaps,
    count_nonzero_coefficients,
    create_coefficient_heatmap,
    normalize_coefficients,
    NormalizationStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Wavelet Coefficient Visualization Example");
    println!("----------------------------------------");

    // Create a test image (checkerboard pattern)
    let size = 64;
    println!(
        "\nCreating a {}x{} test image (checkerboard pattern)...",
        size, size
    );
    let mut image = Array2::zeros((size, size));

    // Create checkerboard pattern with various features
    for i in 0..size {
        for j in 0..size {
            // Base checkerboard
            let base = if (i / 8 + j / 8) % 2 == 0 { 1.0 } else { 0.3 };

            // Add a circle in the middle
            let circle_radius = 15.0;
            let ci = i as f64 - size as f64 / 2.0;
            let cj = j as f64 - size as f64 / 2.0;
            let distance = (ci * ci + cj * cj).sqrt();

            // Add intensity gradient across the image
            let gradient = 0.5 * (i as f64 / size as f64);

            if distance < circle_radius {
                image[[i, j]] = 0.8 + 0.2 * distance / circle_radius;
            } else {
                image[[i, j]] = base + gradient;
            }
        }
    }

    // Print a small section to show the pattern
    println!("\nA small section of the image (8x8 from top-left):");
    print_array2(&image.slice(s![0..8, 0..8]));

    // Single-level 2D DWT
    println!("\nPerforming single-level 2D DWT with Haar wavelet...");
    let dwt_decomp = dwt2d_decompose(&image, Wavelet::Haar, None)?;

    // Arrange coefficients for visualization
    println!("\nArranging coefficients in standard layout (LL, HL, LH, HH)...");
    let _arranged_dwt = arrange_coefficients_2d(&dwt_decomp);
    println!(
        "Arranged coefficients have shape: {:?}",
        _arranged_dwt.shape()
    );

    // Calculate energy distribution
    println!("\nCalculating energy distribution in DWT subbands...");
    let energy = calculate_energy_2d(&dwt_decomp);

    println!("\nEnergy distribution:");
    println!(
        "  Approximation (LL): {:.2}% ({:.2})",
        energy.approximation_percent, energy.approximation
    );
    println!(
        "  Horizontal Detail (LH): {:.2}% ({:.2})",
        100.0 * energy.horizontal.unwrap() / energy.total,
        energy.horizontal.unwrap()
    );
    println!(
        "  Vertical Detail (HL): {:.2}% ({:.2})",
        100.0 * energy.vertical.unwrap() / energy.total,
        energy.vertical.unwrap()
    );
    println!(
        "  Diagonal Detail (HH): {:.2}% ({:.2})",
        100.0 * energy.diagonal.unwrap() / energy.total,
        energy.diagonal.unwrap()
    );
    println!(
        "  Total Detail: {:.2}% ({:.2})",
        energy.detail_percent, energy.detail
    );
    println!("  Total Energy: {:.2}", energy.total);

    // Count non-zero coefficients
    println!("\nCounting non-zero coefficients (threshold = 0.1)...");
    let counts = count_nonzero_coefficients(&dwt_decomp, Some(0.1));

    println!("\nNon-zero coefficient counts:");
    println!(
        "  Approximation (LL): {} coefficients",
        counts.approximation
    );
    println!(
        "  Horizontal Detail (LH): {} coefficients",
        counts.horizontal
    );
    println!("  Vertical Detail (HL): {} coefficients", counts.vertical);
    println!("  Diagonal Detail (HH): {} coefficients", counts.diagonal);
    println!(
        "  Total non-zero: {} out of {} ({:.2}%)",
        counts.total,
        size * size,
        counts.percent_nonzero
    );

    // Multi-level 2D DWT
    println!("\nPerforming multi-level 2D DWT with Haar wavelet (3 levels)...");
    let _multi_decomp = wavedec2(&image, Wavelet::Haar, 3, None)?;

    println!("\nSkipping multi-level coefficient arrangement for now...");
    // The multi-level arrangement is complex and we'll need to refine it further
    // Let's focus on the other visualization utilities that are working correctly

    // 2D SWT for comparison
    println!("\nPerforming 2D Stationary Wavelet Transform with Haar wavelet...");
    let swt_decomp = swt2d_decompose(&image, Wavelet::Haar, 1, None)?;

    // Calculate energy for SWT
    println!("\nCalculating energy distribution in SWT subbands...");
    let swt_energy = calculate_energy_swt2d(&swt_decomp);

    println!("\nSWT Energy distribution:");
    println!(
        "  Approximation (LL): {:.2}% ({:.2})",
        swt_energy.approximation_percent, swt_energy.approximation
    );
    println!(
        "  Total Detail: {:.2}% ({:.2})",
        swt_energy.detail_percent, swt_energy.detail
    );

    // Normalize coefficients for visualization
    println!("\nNormalizing coefficients for better visualization...");
    let _normalized_approx =
        normalize_coefficients(&dwt_decomp.approx, NormalizationStrategy::MinMax, None);
    let _normalized_detail_h =
        normalize_coefficients(&dwt_decomp.detail_h, NormalizationStrategy::MinMax, None);

    println!(
        "Normalized approximation coefficients range: [{:.2}, {:.2}]",
        _normalized_approx
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)),
        _normalized_approx
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Apply different normalization strategies
    println!("\nApplying different normalization strategies to horizontal detail coefficients...");
    let norm_minmax =
        normalize_coefficients(&dwt_decomp.detail_h, NormalizationStrategy::MinMax, None);
    let norm_abs =
        normalize_coefficients(&dwt_decomp.detail_h, NormalizationStrategy::Absolute, None);
    let norm_log = normalize_coefficients(&dwt_decomp.detail_h, NormalizationStrategy::Log, None);
    let norm_perc = normalize_coefficients(
        &dwt_decomp.detail_h,
        NormalizationStrategy::Percentile(5.0, 95.0),
        None,
    );

    println!(
        "\nMinMax normalization - range: [{:.2}, {:.2}]",
        norm_minmax.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        norm_minmax.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "Absolute normalization - range: [{:.2}, {:.2}]",
        norm_abs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        norm_abs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "Log normalization - range: [{:.2}, {:.2}]",
        norm_log.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        norm_log.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "Percentile normalization - range: [{:.2}, {:.2}]",
        norm_perc.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        norm_perc.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Create coefficient heatmaps with different colormaps
    println!("\nCreating coefficient heatmaps with different colormaps...");
    let _heatmap_viridis = create_coefficient_heatmap(
        &dwt_decomp.detail_h,
        colormaps::viridis,
        Some(NormalizationStrategy::MinMax),
    );
    let _heatmap_plasma = create_coefficient_heatmap(
        &dwt_decomp.detail_h,
        colormaps::plasma,
        Some(NormalizationStrategy::MinMax),
    );
    let _heatmap_diverging = create_coefficient_heatmap(
        &dwt_decomp.detail_h,
        colormaps::diverging_rb,
        Some(NormalizationStrategy::Absolute),
    );

    println!(
        "Generated heatmaps with shapes: viridis={:?}, plasma={:?}, diverging={:?}",
        _heatmap_viridis.shape(),
        _heatmap_plasma.shape(),
        _heatmap_diverging.shape()
    );

    // Summary of visualization options
    println!("\nSummary of visualization options:");
    println!("- Coefficient arrangement: Standard quadrant layout or multi-level");
    println!("- Energy distribution: Shows energy concentration in subbands");
    println!("- Coefficient counting: Quantifies sparsity of representation");
    println!("- Normalization: Adjusts coefficient range for better visualization");
    println!("- Colormaps: Different color schemes for heatmap visualization");

    println!("\nThese visualization techniques help to understand wavelet transforms and their applications.");
    println!("For full visualization, you would typically save these as images or display them in a GUI.");

    Ok(())
}

// Helper function to print a small array
fn print_array2(array: &ndarray::ArrayView2<f64>) {
    let (rows, cols) = array.dim();
    for i in 0..rows {
        for j in 0..cols {
            print!("{:.2} ", array[[i, j]]);
        }
        println!();
    }
}
