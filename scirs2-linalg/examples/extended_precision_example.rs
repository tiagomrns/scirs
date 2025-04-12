//! Extended precision matrix operations example
//!
//! This example demonstrates the use of extended precision operations for improved accuracy.

use ndarray::{Array1, Array2};
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::extended_precision::{extended_matmul, extended_solve};

fn main() -> LinalgResult<()> {
    println!("Extended Precision Matrix Operations Example");
    println!("==========================================\n");

    // Demo with a Hilbert matrix, which is notoriously ill-conditioned
    println!("Example with Hilbert Matrix (ill-conditioned)");
    println!("-------------------------------------------\n");

    // Create a small Hilbert matrix
    let n = 5;
    let mut hilbert = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            hilbert[[i, j]] = 1.0 / ((i + j + 1) as f64);
        }
    }

    println!("Hilbert matrix of size {}x{}:", n, n);
    for i in 0..n {
        for j in 0..n {
            print!("{:.6} ", hilbert[[i, j]]);
        }
        println!();
    }
    println!();

    // Create a known solution vector
    let x_true: Array1<f64> = Array1::from_vec((0..n).map(|i| (i + 1) as f64).collect());
    println!("True solution x: {:?}", x_true);

    // Compute right-hand side b = A*x
    let b = hilbert.dot(&x_true);
    println!("Right-hand side b: {:?}\n", b);

    // Solve using standard precision
    let x_std = match scirs2_linalg::solve(&hilbert.view(), &b.view()) {
        Ok(result) => result,
        Err(e) => {
            println!("Error in standard precision solve: {}", e);
            Array1::zeros(n)
        }
    };

    // Solve using extended precision (f64 calculation internally for f32 data)
    // For this example, we'll manually convert to f32 and back
    let hilbert_f32: Array2<f32> = Array2::from_shape_fn((n, n), |(i, j)| hilbert[[i, j]] as f32);
    let b_f32: Array1<f32> = Array1::from_shape_fn(n, |i| b[i] as f32);

    let x_ext = extended_solve::<f32, f64>(&hilbert_f32.view(), &b_f32.view())?;

    // Compute errors
    let error_std = x_std
        .iter()
        .zip(x_true.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |max, val| if val > max { val } else { max });

    let error_ext = x_ext
        .iter()
        .zip((0..n).map(|i| (i + 1) as f32))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |max, val| if val > max { val } else { max });

    println!("Solution using standard precision:");
    println!("{:?}", x_std);
    println!("Maximum absolute error: {:.6e}\n", error_std);

    println!("Solution using extended precision:");
    println!("{:?}", x_ext);
    println!("Maximum absolute error: {:.6e}\n", error_ext);

    // Matrix-matrix multiplication example
    println!("Extended Precision Matrix Multiplication");
    println!("-------------------------------------\n");

    let a_f32 = Array2::from_shape_fn((3, 3), |(i, j)| 1.0 / ((i + j + 1) as f32));

    let b_f32 = Array2::from_shape_fn((3, 2), |(i, j)| ((i + 1) * (j + 1)) as f32);

    println!("Matrix A (f32):");
    for i in 0..a_f32.nrows() {
        for j in 0..a_f32.ncols() {
            print!("{:.6} ", a_f32[[i, j]]);
        }
        println!();
    }

    println!("\nMatrix B (f32):");
    for i in 0..b_f32.nrows() {
        for j in 0..b_f32.ncols() {
            print!("{:.6} ", b_f32[[i, j]]);
        }
        println!();
    }

    // Standard precision multiplication
    let c_std = a_f32.dot(&b_f32);

    // Extended precision multiplication
    let c_ext = extended_matmul::<f32, f64>(&a_f32.view(), &b_f32.view())?;

    println!("\nResult with standard precision:");
    for i in 0..c_std.nrows() {
        for j in 0..c_std.ncols() {
            print!("{:.10} ", c_std[[i, j]]);
        }
        println!();
    }

    println!("\nResult with extended precision:");
    for i in 0..c_ext.nrows() {
        for j in 0..c_ext.ncols() {
            print!("{:.10} ", c_ext[[i, j]]);
        }
        println!();
    }

    // Calculate the difference
    let mut max_diff = 0.0f32;
    for i in 0..c_std.nrows() {
        for j in 0..c_std.ncols() {
            let diff = (c_std[[i, j]] - c_ext[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!("\nMaximum difference between methods: {:.6e}", max_diff);

    Ok(())
}
