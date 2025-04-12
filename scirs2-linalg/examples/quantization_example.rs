//! Example demonstrating the use of quantization-aware linear algebra operations
//!
//! This example demonstrates how to use the quantization module for
//! various linear algebra operations with reduced precision.

use ndarray::{Array1, Array2};
use scirs2_linalg::quantization::{
    dequantize_matrix, fake_quantize, quantize_matrix, quantize_vector, quantized_dot,
    quantized_matmul, quantized_matvec, QuantizationMethod,
};

fn main() {
    println!("Quantization-aware Linear Algebra Example");
    println!("=======================================\n");

    // Create some sample matrices and vectors
    let a =
        Array2::from_shape_vec((3, 3), vec![1.2, 2.5, 3.7, 4.2, 5.0, 6.1, 7.3, 8.4, 9.5]).unwrap();

    let b = Array2::from_shape_vec((3, 2), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();

    let x = Array1::from_shape_vec(3, vec![0.1, 0.2, 0.3]).unwrap();

    println!("Original Matrix A:");
    println!("{:?}\n", a);

    println!("Original Matrix B:");
    println!("{:?}\n", b);

    println!("Original Vector x:");
    println!("{:?}\n", x);

    // Demonstrate basic quantization and dequantization
    println!("Basic Quantization-Dequantization");
    println!("--------------------------------");

    // Uniform quantization
    let (a_q_uniform, a_params_uniform) =
        quantize_matrix(&a.view(), 8, QuantizationMethod::Uniform);
    let a_dequant_uniform = dequantize_matrix(&a_q_uniform, &a_params_uniform);

    println!("Uniform Quantization Parameters:");
    println!("  Bits: {}", a_params_uniform.bits);
    println!("  Scale: {}", a_params_uniform.scale);
    println!("  Zero point: {}", a_params_uniform.zero_point);
    println!("  Min value: {}", a_params_uniform.min_val);
    println!("  Max value: {}", a_params_uniform.max_val);
    println!();

    println!("Quantized Matrix A (Uniform, 8-bit):");
    println!("{:?}\n", a_q_uniform.data);

    println!("Dequantized Matrix A (Uniform):");
    println!("{:?}\n", a_dequant_uniform);

    println!("Quantization Error (Uniform):");
    println!("{:?}\n", &a - &a_dequant_uniform);

    // Symmetric quantization
    let (a_q_symmetric, a_params_symmetric) =
        quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let a_dequant_symmetric = dequantize_matrix(&a_q_symmetric, &a_params_symmetric);

    println!("Symmetric Quantization Parameters:");
    println!("  Bits: {}", a_params_symmetric.bits);
    println!("  Scale: {}", a_params_symmetric.scale);
    println!("  Zero point: {}", a_params_symmetric.zero_point);
    println!("  Min value: {}", a_params_symmetric.min_val);
    println!("  Max value: {}", a_params_symmetric.max_val);
    println!();

    println!("Quantized Matrix A (Symmetric, 8-bit):");
    println!("{:?}\n", a_q_symmetric.data);

    println!("Dequantized Matrix A (Symmetric):");
    println!("{:?}\n", a_dequant_symmetric);

    println!("Quantization Error (Symmetric):");
    println!("{:?}\n", &a - &a_dequant_symmetric);

    // Affine quantization
    let (a_q_affine, a_params_affine) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
    let a_dequant_affine = dequantize_matrix(&a_q_affine, &a_params_affine);

    println!("Affine Quantization Parameters:");
    println!("  Bits: {}", a_params_affine.bits);
    println!("  Scale: {}", a_params_affine.scale);
    println!("  Zero point: {}", a_params_affine.zero_point);
    println!("  Min value: {}", a_params_affine.min_val);
    println!("  Max value: {}", a_params_affine.max_val);
    println!();

    println!("Quantized Matrix A (Affine, 8-bit):");
    println!("{:?}\n", a_q_affine.data);

    println!("Dequantized Matrix A (Affine):");
    println!("{:?}\n", a_dequant_affine);

    println!("Quantization Error (Affine):");
    println!("{:?}\n", &a - &a_dequant_affine);

    // Quantized Matrix Operations
    println!("Quantized Matrix Operations");
    println!("--------------------------");

    // Regular matrix multiplication
    println!("Regular Matrix Multiplication A * B:");
    let c = a.dot(&b);
    println!("{:?}\n", c);

    // Quantized matrix multiplication
    let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
    println!("Quantized Matrix Multiplication (8-bit):");
    let c_q = quantized_matmul(&a_q_symmetric, &a_params_symmetric, &b_q, &b_params).unwrap();
    println!("{:?}\n", c_q);

    println!("Quantization Error for Matrix Multiplication:");
    println!("{:?}\n", &c - &c_q);

    // Calculate relative error
    let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
    println!(
        "Relative Error for Matrix Multiplication: {:.6}\n",
        rel_error
    );

    // Regular matrix-vector multiplication
    println!("Regular Matrix-Vector Multiplication A * x:");
    let y = a.dot(&x);
    println!("{:?}\n", y);

    // Quantized matrix-vector multiplication
    let (x_q, x_params) = quantize_vector(&x.view(), 8, QuantizationMethod::Symmetric);
    println!("Quantized Matrix-Vector Multiplication (8-bit):");
    let y_q = quantized_matvec(&a_q_symmetric, &a_params_symmetric, &x_q, &x_params).unwrap();
    println!("{:?}\n", y_q);

    println!("Quantization Error for Matrix-Vector Multiplication:");
    println!("{:?}\n", &y - &y_q);

    // Calculate relative error
    let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();
    println!(
        "Relative Error for Matrix-Vector Multiplication: {:.6}\n",
        rel_error
    );

    // Regular dot product
    println!("Regular Dot Product x . x:");
    let dot = x.dot(&x);
    println!("{:?}\n", dot);

    // Quantized dot product
    println!("Quantized Dot Product (8-bit):");
    let dot_q = quantized_dot(&x_q, &x_params, &x_q, &x_params).unwrap();
    println!("{:?}\n", dot_q);

    println!("Quantization Error for Dot Product:");
    println!("{:?}\n", dot - dot_q);

    // Calculate relative error
    let rel_error = (dot - dot_q).abs() / dot;
    println!("Relative Error for Dot Product: {:.6}\n", rel_error);

    // Fake Quantization Example
    println!("Fake Quantization Example");
    println!("------------------------");

    // Apply fake quantization with different bit widths
    let bits_list = [4, 6, 8];
    for &bits in &bits_list {
        println!("Fake Quantization with {}-bit:", bits);
        let a_fake_q = fake_quantize(&a.view(), bits, QuantizationMethod::Symmetric);

        println!("Fake Quantized Matrix:");
        println!("{:?}\n", a_fake_q);

        println!("Quantization Error:");
        println!("{:?}\n", &a - &a_fake_q);

        // Calculate mean squared error
        let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;
        println!("Mean Squared Error: {:.6}\n", mse);
    }

    // Comparison across different quantization methods and bit widths
    println!("Quantization Comparison");
    println!("----------------------");

    let methods = [
        QuantizationMethod::Uniform,
        QuantizationMethod::Symmetric,
        QuantizationMethod::Affine,
        QuantizationMethod::PowerOfTwo,
    ];

    let bits_list = [4, 6, 8, 10];

    println!("Mean Squared Error (MSE) for different methods and bit widths:");
    println!();

    // Print header
    print!("{:<12}", "Bits");
    for method in &methods {
        let method_name = match method {
            QuantizationMethod::Uniform => "Uniform",
            QuantizationMethod::Symmetric => "Symmetric",
            QuantizationMethod::Affine => "Affine",
            QuantizationMethod::PowerOfTwo => "PowerOfTwo",
        };
        print!("{:<12}", method_name);
    }
    println!();

    // Print MSE for each combination
    for &bits in &bits_list {
        print!("{:<12}", bits);

        for &method in &methods {
            let a_fake_q = fake_quantize(&a.view(), bits, method);
            let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;
            print!("{:<12.6}", mse);
        }

        println!();
    }
}
