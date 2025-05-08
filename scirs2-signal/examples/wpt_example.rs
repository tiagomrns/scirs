use plotly::common::Mode;
use plotly::layout::Layout;
use plotly::{Plot, Scatter};
use rand::Rng;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::waveforms::chirp;
use scirs2_signal::wpt::{get_level_coefficients, reconstruct_from_nodes, wp_decompose};

fn main() {
    // Generate a chirp signal (frequency increasing linearly with time)
    let fs = 1024.0; // Sample rate
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();

    // Add some noise to the signal
    let mut rng = rand::rng();
    let noisy_signal = signal
        .iter()
        .map(|&x| x + 0.05 * rng.random_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    // Perform wavelet packet decomposition to level 3
    let level = 3;
    let wpt = wp_decompose(&noisy_signal, Wavelet::DB(4), level, None).unwrap();

    // Get coefficients at each level
    let level1_coeffs = get_level_coefficients(&wpt, 1);
    let level2_coeffs = get_level_coefficients(&wpt, 2);
    let level3_coeffs = get_level_coefficients(&wpt, 3);

    println!("Wavelet Packet Transform Decomposition:");
    println!("Level 1: {} subbands", level1_coeffs.len());
    println!("Level 2: {} subbands", level2_coeffs.len());
    println!("Level 3: {} subbands", level3_coeffs.len());

    // Create plots of coefficients at each level
    plot_coefficients(&level1_coeffs, 1, "Level 1 Wavelet Packet Coefficients");
    plot_coefficients(&level2_coeffs, 2, "Level 2 Wavelet Packet Coefficients");
    plot_coefficients(&level3_coeffs, 3, "Level 3 Wavelet Packet Coefficients");

    // Perform denoising by thresholding level 3 coefficients
    let mut thresholded_coeffs = level3_coeffs.clone();
    let threshold = 0.1;

    // Apply thresholding
    for subband in thresholded_coeffs.iter_mut() {
        for val in subband.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
            }
        }
    }

    // Reconstruct from thresholded coefficients
    let mut nodes = Vec::new();
    for i in 0..thresholded_coeffs.len() {
        nodes.push((3, i));
    }

    let denoised_signal = reconstruct_from_nodes(&wpt, &nodes).unwrap();

    // Plot original, noisy, and denoised signals
    let mut plot = Plot::new();

    // Original signal
    let original_trace = Scatter::new(t.clone(), signal.clone().to_vec())
        .name("Original Signal")
        .mode(Mode::Lines);

    // Noisy signal
    let noisy_trace = Scatter::new(t.clone(), noisy_signal.clone().to_vec())
        .name("Noisy Signal")
        .mode(Mode::Lines);

    // Denoised signal
    let denoised_trace = Scatter::new(t.clone(), denoised_signal.to_vec())
        .name("Denoised Signal (WPT)")
        .mode(Mode::Lines);

    // Add traces to plot
    plot.add_trace(original_trace);
    plot.add_trace(noisy_trace);
    plot.add_trace(denoised_trace);

    // Set layout
    let layout = Layout::new().title("Wavelet Packet Transform Denoising");

    plot.set_layout(layout);

    // Save to HTML file
    plot.write_html("wpt_denoising.html");
    println!("Signal plot saved to wpt_denoising.html");

    // For a simpler approach, let's just use the lowest level nodes (level 1)
    // This is more reliable with our current implementation

    // Get number of level 1 coefficients
    let level1_count = get_level_coefficients(&wpt, 1).len();
    println!(
        "Using {} level 1 coefficients for reconstruction",
        level1_count
    );

    // Reconstruct from all level 1 nodes
    let selected_nodes: Vec<(usize, usize)> = vec![(1, 0), (1, 1)];

    // Reconstruct signal from level 1 nodes
    let best_basis_signal = reconstruct_from_nodes(&wpt, &selected_nodes).unwrap();

    // Plot best basis reconstruction
    let mut bb_plot = Plot::new();

    // Original signal
    let original_trace = Scatter::new(t.clone(), signal.clone().to_vec())
        .name("Original Signal")
        .mode(Mode::Lines);

    // Best basis signal
    let bb_trace = Scatter::new(t.clone(), best_basis_signal.to_vec())
        .name("Best Basis Signal")
        .mode(Mode::Lines);

    // Add traces to plot
    bb_plot.add_trace(original_trace);
    bb_plot.add_trace(bb_trace);

    // Set layout
    let layout = Layout::new().title("Wavelet Packet Transform - Best Basis Reconstruction");

    bb_plot.set_layout(layout);

    // Save to HTML file
    bb_plot.write_html("wpt_best_basis.html");
    println!("Best basis plot saved to wpt_best_basis.html");
}

// Helper function to plot coefficients at a specific level
fn plot_coefficients(coeffs: &[Vec<f64>], level: usize, title: &str) {
    let mut plot = Plot::new();

    // Create a simple layout with a title
    let layout = Layout::new().title(title);

    // Add a trace for each subband
    for (i, subband) in coeffs.iter().enumerate() {
        let x = (0..subband.len()).map(|x| x as f64).collect::<Vec<f64>>();

        let trace = Scatter::new(x, subband.clone())
            .name(&format!("Subband {}", i))
            .mode(Mode::Lines);

        plot.add_trace(trace);
    }

    plot.set_layout(layout);

    // Save to HTML file
    plot.write_html(&format!("wpt_level{}_coeffs.html", level));
    println!(
        "Level {} coefficients plot saved to wpt_level{}_coeffs.html",
        level, level
    );
}
