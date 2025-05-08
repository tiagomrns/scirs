use ndarray::Array2;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::wpt2d::{wpt2d_full, wpt2d_selective, WaveletPacket2D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("2D Wavelet Packet Transform Example");
    println!("-----------------------------------");

    // Create a test image (64x64)
    let size = 64;
    let mut image = Array2::zeros((size, size));

    // Generate a pattern with a circle in the center
    let center_x = size as f64 / 2.0;
    let center_y = size as f64 / 2.0;
    let radius = size as f64 / 4.0;

    println!(
        "Creating a {}x{} test image with a circle pattern...",
        size, size
    );
    for i in 0..size {
        for j in 0..size {
            let x = j as f64 - center_x;
            let y = i as f64 - center_y;
            let distance = (x * x + y * y).sqrt();

            if distance < radius {
                image[[i, j]] = 1.0;
            } else {
                image[[i, j]] = 0.0;
            }
        }
    }

    println!("Performing a full 2D wavelet packet decomposition (2 levels)...");
    // Perform full 2-level wavelet packet decomposition
    let full_decomp = wpt2d_full(&image, Wavelet::Haar, 2, None)?;

    // Print statistics about the decomposition
    println!(
        "Full decomposition contains {} wavelet packets:",
        full_decomp.len()
    );
    println!("  Level 0: 1 packet");
    println!("  Level 1: 4 packets");
    println!("  Level 2: 16 packets");

    println!("\nEnergy distribution across level 1 packets:");
    for row in 0..2 {
        for col in 0..2 {
            let packet = full_decomp.get_packet(1, row, col).unwrap();
            println!(
                "  Packet ({}, {}), path {}: Energy = {:.2}",
                row,
                col,
                packet.path,
                packet.energy()
            );
        }
    }

    // Now perform a selective decomposition
    println!("\nPerforming a selective 2D wavelet packet decomposition (3 levels)...");
    // Define a criterion that only decomposes packets with significant energy
    let energy_threshold = 10.0;
    let energy_criterion = move |packet: &WaveletPacket2D| -> bool {
        // Only decompose nodes with energy above the threshold,
        // or nodes at level 0 (the root node is always decomposed)
        packet.level == 0 || packet.energy() > energy_threshold
    };

    // Perform selective wavelet packet decomposition
    let selective_decomp = wpt2d_selective(&image, Wavelet::Haar, 3, energy_criterion, None)?;

    // Analyze the selective decomposition
    let level1_count = selective_decomp.get_level_packets(1).len();
    let level2_count = selective_decomp.get_level_packets(2).len();
    let level3_count = selective_decomp.get_level_packets(3).len();
    let total_count = selective_decomp.len();

    println!(
        "Selective decomposition contains {} wavelet packets:",
        total_count
    );
    println!("  Level 0: 1 packet");
    println!("  Level 1: {} packets", level1_count);
    println!("  Level 2: {} packets", level2_count);
    println!("  Level 3: {} packets", level3_count);

    println!(
        "\nSignificant packets at level 2 (with energy > {}):",
        energy_threshold
    );
    for packet in selective_decomp.get_level_packets(2) {
        if packet.energy() > energy_threshold {
            println!(
                "  Packet at level {}, position ({}, {}), path {}: Energy = {:.2}",
                packet.level,
                packet.row,
                packet.col,
                packet.path,
                packet.energy()
            );
        }
    }

    // For comparing computational savings...
    let full_nodes_at_level3 = 64; // 2^6 = 64 at level 3 for a full decomposition
    let savings_percent =
        100.0 * (1.0 - (total_count as f64 / (1 + 4 + 16 + full_nodes_at_level3) as f64));

    println!(
        "\nThe selective decomposition reduced the number of computed nodes by {:.1}%",
        savings_percent
    );
    println!("This demonstrates the efficiency of adaptive wavelet packet transforms.");

    // For a real image processing application, the selective decomposition could be
    // further analyzed for feature extraction, compression, or noise removal.

    Ok(())
}
