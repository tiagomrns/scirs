use ndarray::array;
use scirs2_linalg::prelude::*;

fn main() {
    println!("Random Matrix Generation Examples");
    println!("=================================\n");

    // Example 1: Uniform random matrix
    println!("Example 1: Uniform Random Matrix");
    let u = uniform::<f64>(3, 4, -1.0, 1.0, Some(42));
    println!("Uniform random 3x4 matrix in range [-1, 1]:");
    println!("{:.4}", u);
    println!();

    // Example 2: Normal (Gaussian) random matrix
    println!("Example 2: Normal (Gaussian) Random Matrix");
    let n = normal::<f64>(3, 3, 0.0, 1.0, Some(42));
    println!("Normal random 3x3 matrix with mean 0 and std 1:");
    println!("{:.4}", n);
    println!();

    // Example 3: Orthogonal random matrix
    println!("Example 3: Orthogonal Random Matrix");
    let q = orthogonal::<f64>(4, Some(42));
    println!("Random 4x4 orthogonal matrix (Q^T * Q = I):");
    println!("{:.4}", q);
    // Verify orthogonality
    let qt = q.t();
    let result = qt.dot(&q);
    println!("Verification - Q^T * Q:");
    println!("{:.4}", result);
    println!();

    // Example 4: Symmetric Positive-Definite matrix
    println!("Example 4: Symmetric Positive-Definite Matrix");
    let s = spd::<f64>(3, 1.0, 10.0, Some(42));
    println!("Random 3x3 SPD matrix with eigenvalues in [1, 10]:");
    println!("{:.4}", s);
    // Verify symmetry
    let st = s.t();
    println!(
        "Verification - S == S^T: {}",
        s.iter().zip(st.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
    );
    // Compute eigenvalues to show they're positive
    let evals = eigvals(&s.view()).unwrap();
    println!("Eigenvalues: {:.4}", evals);
    println!();

    // Example 5: Diagonal matrix
    println!("Example 5: Diagonal Matrix");
    let d = diagonal::<f64>(4, 1.0, 5.0, Some(42));
    println!("Random 4x4 diagonal matrix with values in [1, 5]:");
    println!("{:.4}", d);
    println!();

    // Example 6: Banded matrix
    println!("Example 6: Banded Matrix");
    let b = banded::<f64>(5, 5, 1, 1, -2.0, 2.0, Some(42));
    println!("Random 5x5 tridiagonal matrix (bandwidth 1) with values in [-2, 2]:");
    println!("{:.4}", b);
    println!();

    // Example 7: Sparse matrix
    println!("Example 7: Sparse Matrix");
    let sp = sparse::<f64>(6, 6, 0.3, -1.0, 1.0, Some(42));
    println!("Random 6x6 sparse matrix with 30% density and values in [-1, 1]:");
    println!("{:.4}", sp);
    // Count non-zeros
    let nnz = sp.iter().filter(|&&x| x != 0.0).count();
    println!(
        "Non-zero elements: {} ({}%)",
        nnz,
        100.0 * nnz as f64 / 36.0
    );
    println!();

    // Example 8: Toeplitz matrix
    println!("Example 8: Toeplitz Matrix");
    let t = toeplitz::<f64>(4, -1.0, 1.0, Some(42));
    println!("Random 4x4 Toeplitz matrix with values in [-1, 1]:");
    println!("{:.4}", t);
    println!();

    // Example 9: Matrix with specific condition number
    println!("Example 9: Matrix with Specific Condition Number");
    let c = with_condition_number::<f64>(3, 100.0, Some(42));
    println!("Random 3x3 matrix with condition number ≈ 100:");
    println!("{:.4}", c);
    // Note: condition number verification skipped as implementation is pending
    println!("Note: Matrix should have condition number close to 100");
    println!();

    // Example 10: Matrix with specific eigenvalues
    println!("Example 10: Matrix with Specific Eigenvalues");
    let eigenvalues = array![1.0, 5.0, 10.0];
    let e = with_eigenvalues(&eigenvalues, Some(42));
    println!("Random 3x3 matrix with eigenvalues [1, 5, 10]:");
    println!("{:.4}", e);
    // Verify eigenvalues
    let computed_evals = eigvals(&e.view()).unwrap();
    println!("Computed eigenvalues: {:.4}", computed_evals);
    println!();

    // Applications
    println!("Applications of Random Matrices");
    println!("===============================");

    // Application 1: Testing numerical stability
    println!("1. Testing numerical stability of algorithms");
    println!("   - Use matrices with high condition numbers to test solver robustness");
    println!("   - For example, create matrices with condition number ≈ 1e6");

    // Application 2: Benchmarking
    println!("2. Benchmarking linear algebra operations");
    println!("   - Generate matrices of different sizes and sparsity patterns");
    println!("   - Test algorithm performance on different matrix types");

    // Application 3: Statistical simulations
    println!("3. Statistical simulations and Monte Carlo methods");
    println!("   - Generate random data with specific distributions");
    println!("   - Test statistical properties of algorithms");

    // Application 4: Machine learning
    println!("4. Machine learning applications");
    println!("   - Random weight initialization");
    println!("   - Dropout matrices for regularization");
    println!("   - Random projections for dimensionality reduction");

    // Application 5: Testing sparse solvers
    println!("5. Testing sparse matrix algorithms");
    println!("   - Generate matrices with specific sparsity patterns");
    println!("   - Benchmark sparse vs. dense algorithm performance");
}
