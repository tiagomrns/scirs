//! Generate API freeze report for scirs2-core

use scirs2_core::api_freeze::{generate_frozen_api_report, initialize_api_freeze};

#[allow(dead_code)]
fn main() {
    println!("Generating API freeze report for scirs2-core 1.0...\n");

    // Initialize the API freeze
    initialize_api_freeze();

    // Generate and print the report
    let report = generate_frozen_api_report();
    println!("{report}");

    // Optionally write to file
    if let Some(output_path) = std::env::args().nth(1) {
        match std::fs::write(&output_path, &report) {
            Ok(_) => println!("\nReport written to: {output_path}"),
            Err(e) => eprintln!("\nError writing report to file: {e}"),
        }
    }
}
