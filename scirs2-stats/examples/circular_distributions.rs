// Example file for circular distributions (not fully implemented yet)
//
// NOTE: This example uses features that are still in development.
// Circular distributions (vonmises, wrapcauchy) are listed in the TODO.md
// and will be implemented in a future release.
//
// This file is commented out until the implementation is complete.

/*
use ndarray::Array1;
use scirs2_stats::distributions::{vonmises, wrapcauchy};
use scirs2_stats::traits::CircularDistribution;
use std::f64::consts::PI;

fn main() {
    // Von Mises distribution
    println!("Von Mises Distribution Example");
    println!("-----------------------------");

    // Create a von Mises distribution with mean direction π/4 and concentration 2.0
    let vm = vonmises(PI / 4.0, 2.0).unwrap();

    // PDF values
    let angles = [-PI, -PI / 2.0, 0.0, PI / 4.0, PI / 2.0, PI];
    println!("PDF values at different angles:");
    for &angle in &angles {
        println!("  PDF at {:.4} rad: {:.6}", angle, vm.pdf(angle));
    }

    // Circular statistics
    println!("\nCircular statistics:");
    println!("  Mean direction: {:.6} rad", vm.circular_mean());
    println!("  Circular variance: {:.6}", vm.circular_variance());
    println!(
        "  Circular standard deviation: {:.6} rad",
        vm.circular_std()
    );
    println!("  Mean resultant length: {:.6}", vm.mean_resultant_length());

    // Generate random samples
    println!("\nGenerating 10 random samples:");
    let samples = vm.rvs(10).unwrap();
    for (i, &sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.6} rad", i + 1, sample);
    }

    // Wrapped Cauchy distribution
    println!("\n\nWrapped Cauchy Distribution Example");
    println!("----------------------------------");

    // Create a wrapped Cauchy distribution with mean direction 0 and concentration 0.6
    let wc = wrapcauchy(0.0, 0.6).unwrap();

    // PDF values
    println!("PDF values at different angles:");
    for &angle in &angles {
        println!("  PDF at {:.4} rad: {:.6}", angle, wc.pdf(angle));
    }

    // Circular statistics
    println!("\nCircular statistics:");
    println!("  Mean direction: {:.6} rad", wc.circular_mean());
    println!("  Circular variance: {:.6}", wc.circular_variance());
    println!(
        "  Circular standard deviation: {:.6} rad",
        wc.circular_std()
    );
    println!("  Mean resultant length: {:.6}", wc.mean_resultant_length());

    // Generate random samples
    println!("\nGenerating 10 random samples:");
    let samples = wc.rvs(10).unwrap();
    for (i, &sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.6} rad", i + 1, sample);
    }

    // Comparing distributions
    println!("\n\nComparison of Circular Distributions");
    println!("----------------------------------");

    // Create multiple distributions with different parameters
    let vm1 = vonmises(0.0, 1.0).unwrap();
    let vm2 = vonmises(0.0, 5.0).unwrap();
    let vm3 = vonmises(0.0, 20.0).unwrap();
    let wc1 = wrapcauchy(0.0, 0.3).unwrap();
    let wc2 = wrapcauchy(0.0, 0.7).unwrap();

    // Compare PDFs
    println!("PDF values at different angles:");
    let test_angles = Array1::linspace(-PI, PI, 13);
    println!("  Angle(rad) | VM(κ=1) | VM(κ=5) | VM(κ=20) | WC(γ=0.3) | WC(γ=0.7)");
    println!("  --------------------------------------------------------------------------");
    for &angle in test_angles.iter() {
        println!(
            "  {:8.4} | {:7.4} | {:7.4} | {:8.4} | {:9.4} | {:9.4}",
            angle,
            vm1.pdf(angle),
            vm2.pdf(angle),
            vm3.pdf(angle),
            wc1.pdf(angle),
            wc2.pdf(angle)
        );
    }

    // Compare concentration effects
    println!("\nEffect of concentration parameter on circular variance:");
    println!("  Distribution | Parameter | Circular Variance | Mean Resultant Length");
    println!("  ------------------------------------------------------------");
    for kappa in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
        let vm = vonmises(0.0, kappa).unwrap();
        println!(
            "  Von Mises    | κ = {:4.1}   | {:16.4} | {:20.4}",
            kappa,
            vm.circular_variance(),
            vm.mean_resultant_length()
        );
    }

    println!("  ------------------------------------------------------------");
    for gamma in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let wc = wrapcauchy(0.0, gamma).unwrap();
        println!(
            "  Wrap Cauchy  | γ = {:4.1}   | {:16.4} | {:20.4}",
            gamma,
            wc.circular_variance(),
            wc.mean_resultant_length()
        );
    }
}
*/

// Placeholder main function until circular distributions are implemented
fn main() {
    println!("Circular distributions are not yet fully implemented.");
    println!("This example will be enabled in a future release.");
    println!("See the TODO.md file for the implementation status.");
}
