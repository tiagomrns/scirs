//! Example comparing the numerical stability of different FrFT implementations
//!
//! This example demonstrates the improved numerical stability of the Ozaktas-Kutay
//! algorithm compared to the standard decomposition method, particularly for the
//! additivity property.

use num_complex::Complex64;
use scirs2_fft::{frft, frft_dft, frft_stable};
use std::f64::consts::PI;

fn main() {
    println!("Fractional Fourier Transform Numerical Stability Comparison");
    println!("=========================================================\n");

    // Test 1: Additivity property
    test_additivity_property();

    // Test 2: Energy conservation
    test_energy_conservation();

    // Test 3: Cascaded transforms
    test_cascaded_transforms();
}

fn test_additivity_property() {
    println!("1. Additivity Property Test");
    println!("---------------------------");
    println!("Testing: FrFT(α₁+α₂)[x] ≈ FrFT(α₁)[FrFT(α₂)[x]]\n");

    let n = 64;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            (2.0 * PI * 5.0 * i as f64 / n as f64).sin()
                + (2.0 * PI * 12.0 * i as f64 / n as f64).sin()
        })
        .collect();

    let test_cases = vec![(0.3, 0.4), (0.5, 0.7), (0.8, 0.9), (1.2, 0.6)];

    for (alpha1, alpha2) in test_cases {
        println!("α₁ = {}, α₂ = {}", alpha1, alpha2);

        // Original algorithm
        let direct_orig = frft(&signal, alpha1 + alpha2, None).unwrap();
        let temp_orig = frft(&signal, alpha2, None).unwrap();
        let sequential_orig = frft(
            &temp_orig.iter().map(|&c| c.re).collect::<Vec<_>>(),
            alpha1,
            None,
        )
        .unwrap();

        let energy_direct_orig: f64 = direct_orig.iter().map(|c| c.norm_sqr()).sum();
        let energy_sequential_orig: f64 = sequential_orig.iter().map(|c| c.norm_sqr()).sum();
        let ratio_orig = energy_direct_orig / energy_sequential_orig;

        // Ozaktas algorithm
        let direct_ozaktas = frft_stable(&signal, alpha1 + alpha2).unwrap();
        let temp_ozaktas = frft_stable(&signal, alpha2).unwrap();
        let sequential_ozaktas = frft_stable(
            &temp_ozaktas.iter().map(|&c| c.re).collect::<Vec<_>>(),
            alpha1,
        )
        .unwrap();

        let energy_direct_ozaktas: f64 = direct_ozaktas.iter().map(|c| c.norm_sqr()).sum();
        let energy_sequential_ozaktas: f64 = sequential_ozaktas.iter().map(|c| c.norm_sqr()).sum();
        let ratio_ozaktas = energy_direct_ozaktas / energy_sequential_ozaktas;

        // DFT-based algorithm
        let direct_dft = frft_dft(&signal, alpha1 + alpha2).unwrap();
        let temp_dft = frft_dft(&signal, alpha2).unwrap();
        let sequential_dft =
            frft_dft(&temp_dft.iter().map(|&c| c.re).collect::<Vec<_>>(), alpha1).unwrap();

        let energy_direct_dft: f64 = direct_dft.iter().map(|c| c.norm_sqr()).sum();
        let energy_sequential_dft: f64 = sequential_dft.iter().map(|c| c.norm_sqr()).sum();
        let ratio_dft = energy_direct_dft / energy_sequential_dft;

        println!(
            "  Original algorithm - Energy ratio: {:.4} (deviation: {:.1}%)",
            ratio_orig,
            ((ratio_orig - 1.0).abs() * 100.0)
        );
        println!(
            "  Ozaktas algorithm  - Energy ratio: {:.4} (deviation: {:.1}%)",
            ratio_ozaktas,
            ((ratio_ozaktas - 1.0).abs() * 100.0)
        );
        println!(
            "  DFT-based algorithm - Energy ratio: {:.4} (deviation: {:.1}%)",
            ratio_dft,
            ((ratio_dft - 1.0).abs() * 100.0)
        );
        println!();
    }
}

fn test_energy_conservation() {
    println!("2. Energy Conservation Test");
    println!("---------------------------");
    println!("Testing energy conservation across different α values\n");

    let n = 128;
    let signal: Vec<f64> = (0..n)
        .map(|i| (-((i as f64 - n as f64 / 2.0).powi(2)) / 100.0).exp())
        .collect();

    let input_energy: f64 = signal.iter().map(|&x| x * x).sum();
    println!("Input signal energy: {:.6}\n", input_energy);

    let alphas = vec![0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];

    println!("α      Original   Ozaktas    DFT-based  Orig Dev%  Ozak Dev%  DFT Dev%");
    println!("---    --------   --------   ---------  ---------  ---------  --------");

    for alpha in alphas {
        let result_orig = frft(&signal, alpha, None).unwrap();
        let energy_orig: f64 = result_orig.iter().map(|c| c.norm_sqr()).sum();
        let deviation_orig = ((energy_orig - input_energy) / input_energy * 100.0).abs();

        let result_ozaktas = frft_stable(&signal, alpha).unwrap();
        let energy_ozaktas: f64 = result_ozaktas.iter().map(|c| c.norm_sqr()).sum();
        let deviation_ozaktas = ((energy_ozaktas - input_energy) / input_energy * 100.0).abs();

        let result_dft = frft_dft(&signal, alpha).unwrap();
        let energy_dft: f64 = result_dft.iter().map(|c| c.norm_sqr()).sum();
        let deviation_dft = ((energy_dft - input_energy) / input_energy * 100.0).abs();

        println!(
            "{:.1}    {:.6}   {:.6}   {:.6}   {:8.2}   {:8.2}   {:7.2}",
            alpha,
            energy_orig,
            energy_ozaktas,
            energy_dft,
            deviation_orig,
            deviation_ozaktas,
            deviation_dft
        );
    }
    println!();
}

fn test_cascaded_transforms() {
    println!("3. Cascaded Transforms Test");
    println!("---------------------------");
    println!("Testing multiple sequential transforms\n");

    let n = 32;
    let signal: Vec<f64> = (0..n).map(|i| if i == n / 4 { 1.0 } else { 0.0 }).collect();

    // Apply 10 sequential transforms with α = 0.1
    let alpha = 0.1;
    let num_iterations = 10;

    println!(
        "Applying {} transforms with α = {} each",
        num_iterations, alpha
    );
    println!("Total effective α = {}\n", alpha * num_iterations as f64);

    // Original algorithm
    let mut result_orig = signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect::<Vec<_>>();
    for _ in 0..num_iterations {
        let temp = frft(
            &result_orig.iter().map(|&c| c.re).collect::<Vec<_>>(),
            alpha,
            None,
        )
        .unwrap();
        result_orig = temp;
    }
    let energy_orig: f64 = result_orig.iter().map(|c| c.norm_sqr()).sum();

    // Ozaktas algorithm
    let mut result_ozaktas = signal.clone();
    for _ in 0..num_iterations {
        let temp = frft_stable(&result_ozaktas, alpha).unwrap();
        result_ozaktas = temp.iter().map(|&c| c.re).collect();
    }
    let final_ozaktas = frft_stable(&result_ozaktas, 0.0).unwrap(); // Convert to complex for comparison
    let energy_ozaktas: f64 = final_ozaktas.iter().map(|c| c.norm_sqr()).sum();

    // Direct computation
    let direct_orig = frft(&signal, alpha * num_iterations as f64, None).unwrap();
    let energy_direct_orig: f64 = direct_orig.iter().map(|c| c.norm_sqr()).sum();

    let direct_ozaktas = frft_stable(&signal, alpha * num_iterations as f64).unwrap();
    let energy_direct_ozaktas: f64 = direct_ozaktas.iter().map(|c| c.norm_sqr()).sum();

    println!("Original algorithm:");
    println!("  Cascaded energy: {:.6}", energy_orig);
    println!("  Direct energy:   {:.6}", energy_direct_orig);
    println!(
        "  Ratio:           {:.6}\n",
        energy_orig / energy_direct_orig
    );

    println!("Ozaktas algorithm:");
    println!("  Cascaded energy: {:.6}", energy_ozaktas);
    println!("  Direct energy:   {:.6}", energy_direct_ozaktas);
    println!(
        "  Ratio:           {:.6}\n",
        energy_ozaktas / energy_direct_ozaktas
    );

    // Peak location preservation
    let peak_orig = result_orig
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let peak_ozaktas = final_ozaktas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("Peak preservation:");
    println!("  Original algorithm peak at: {}", peak_orig);
    println!("  Ozaktas algorithm peak at:  {}", peak_ozaktas);
}
