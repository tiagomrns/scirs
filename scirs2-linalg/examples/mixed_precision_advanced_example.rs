use ndarray::{array, Array1, Array2};
use scirs2_linalg::prelude::*;

fn main() {
    println!("Advanced Mixed Precision Linear Algebra Examples");
    println!("===============================================\n");

    // Example 1: Mixed Precision Cholesky Decomposition
    println!("Example 1: Mixed Precision Cholesky Decomposition");
    println!("------------------------------------------");

    // Create a symmetric positive definite matrix in f32 precision
    let a = array![[4.0f32, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]];

    println!("Original matrix A:");
    println!("{}", a);

    // Compute Cholesky decomposition using f64 precision internally
    let l = mixed_precision_cholesky::<_, f32, f64>(&a.view()).unwrap();

    println!("\nCholesky factor L (with f64 computation):");
    println!("{}", l);

    // Verify A = L * L^T
    let lt = l.t();
    let reconstructed = l.dot(&lt);

    println!("\nReconstructed A = L * L^T:");
    println!("{}", reconstructed);

    // Compute the element-wise error
    let mut max_error = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let error = (reconstructed[[i, j]] - a[[i, j]]).abs();
            max_error = max_error.max(error);
        }
    }
    println!("\nMaximum element-wise error: {:.2e}", max_error);

    // Example 2: Solving Linear Systems with Iterative Refinement
    println!("\nExample 2: Iterative Refinement for Linear Systems");
    println!("-------------------------------------------");

    // Create an ill-conditioned matrix - Hilbert matrix of order 5
    let mut hilbert = Array2::zeros((5, 5));
    for i in 0..5 {
        for j in 0..5 {
            hilbert[[i, j]] = 1.0 / ((i + j + 1) as f32);
        }
    }

    println!("Hilbert matrix (5x5, ill-conditioned):");
    for i in 0..5 {
        println!("{}", hilbert.row(i));
    }

    // Create a right-hand side vector
    let b = array![1.0f32, 0.0, 0.0, 0.0, 0.0];
    println!("\nRight-hand side b: {}", b);

    // Solve using standard precision
    let x_std = solve(&hilbert.view(), &b.view()).unwrap();
    println!("\nSolution with standard f32 precision:");
    println!("{}", x_std);

    // Solve using iterative refinement (mixed precision)
    let x_mixed = iterative_refinement_solve::<_, _, f32, f64, f32>(
        &hilbert.view(),
        &b.view(),
        Some(5),
        Some(1e-10),
    )
    .unwrap();
    println!("\nSolution with iterative refinement (f32->f64->f32):");
    println!("{}", x_mixed);

    // Check the accuracy of both solutions
    let ax_std = hilbert.dot(&x_std);
    let ax_mixed = hilbert.dot(&x_mixed);

    let err_std = (&ax_std - &b).mapv(|x| x.abs()).sum();
    let err_mixed = (&ax_mixed - &b).mapv(|x| x.abs()).sum();

    println!("\nResidual error with standard solution: {:.6e}", err_std);
    println!("Residual error with mixed precision: {:.6e}", err_mixed);
    println!("Improvement factor: {:.2}x", err_std / err_mixed);

    // Example 3: Enhanced Dot Product with Kahan Summation
    println!("\nExample 3: Enhanced Dot Product with Kahan Summation");
    println!("----------------------------------------------");

    // Create vectors with both large and small magnitudes
    let size = 1000;
    let mut a = Array1::<f32>::zeros(size);
    let mut b = Array1::<f32>::zeros(size);

    // Fill with values that would cause numerical instability
    for i in 0..size {
        if i % 2 == 0 {
            a[i] = 1.0;
            b[i] = 1.0e8;
        } else {
            a[i] = 1.0;
            b[i] = -1.0e8;
        }
    }

    // Add a small value at the end that shouldn't be lost
    a[size - 2] = 1.0;
    b[size - 2] = 1.0e-8;

    // Add one more normal value
    a[size - 1] = 1.0;
    b[size - 1] = 1.0;

    // Expected result: all the large values cancel out, leaving just 1.0e-8 + 1.0
    let expected_result = 1.00000001;

    // Compute using standard precision
    let standard_result = a.dot(&b);

    // Compute using mixed precision with enhanced summation
    let mixed_result = mixed_precision_dot::<f32, f32, f64, f64>(&a.view(), &b.view()).unwrap();

    println!("Vector dot product with cancellation and small values:");
    println!("Expected result: {:.8}", expected_result);
    println!("Standard result: {:.8}", standard_result);
    println!("Mixed precision result: {:.8}", mixed_result);

    // Calculate errors
    let standard_error = ((standard_result as f64) - expected_result).abs();
    let mixed_error = (mixed_result - expected_result).abs();

    println!("\nStandard error: {:.2e}", standard_error);
    println!("Mixed precision error: {:.2e}", mixed_error);

    if mixed_error < standard_error {
        println!(
            "\nMixed precision dot product is {:.2e} times more accurate!",
            standard_error / mixed_error
        );
    } else {
        println!("\nIn this case, standard precision is sufficient.");
    }

    // Example 4: Mixed-Precision SVD for Nearly Singular Matrices
    println!("\nExample 4: Mixed-Precision SVD for Nearly Singular Matrices");
    println!("--------------------------------------------------");

    // Create a nearly singular matrix
    let nearly_singular = array![
        [1.0f32, 1.0, 1.0],
        [1.0, 1.0 + 1.0e-5, 1.0],
        [1.0, 1.0, 1.0 + 2.0e-5]
    ];

    println!("Nearly singular matrix:");
    println!("{}", nearly_singular);

    // Compute SVD using standard f32 precision
    let svd_result = svd(&nearly_singular.view(), false);
    let (u_f32, s_f32, vt_f32) = match svd_result {
        Ok(result) => result,
        Err(e) => {
            println!("Standard precision SVD failed: {}", e);
            return;
        }
    };

    // Compute SVD using mixed precision
    let (u_mixed, s_mixed, vt_mixed) =
        mixed_precision_svd::<_, f32, f64>(&nearly_singular.view(), false).unwrap();

    println!("\nSingular values (standard precision):");
    println!("{}", s_f32);

    println!("\nSingular values (mixed precision):");
    println!("{}", s_mixed);

    // Compute the condition number
    let cond_f32 = s_f32[0] / s_f32[s_f32.len() - 1];
    let cond_mixed = s_mixed[0] / s_mixed[s_mixed.len() - 1];

    println!("\nCondition number (standard precision): {:.6e}", cond_f32);
    println!("Condition number (mixed precision): {:.6e}", cond_mixed);

    // Reconstruct the original matrix and check error
    let s_diag_f32 = {
        let mut s_mat = Array2::<f32>::zeros((3, 3));
        for i in 0..3 {
            s_mat[[i, i]] = s_f32[i];
        }
        s_mat
    };

    let s_diag_mixed = {
        let mut s_mat = Array2::<f32>::zeros((3, 3));
        for i in 0..3 {
            s_mat[[i, i]] = s_mixed[i];
        }
        s_mat
    };

    let reconstructed_f32 = u_f32.dot(&s_diag_f32).dot(&vt_f32);
    let reconstructed_mixed = u_mixed.dot(&s_diag_mixed).dot(&vt_mixed);

    let err_f32 = (&reconstructed_f32 - &nearly_singular)
        .mapv(|x| x.abs())
        .sum();
    let err_mixed = (&reconstructed_mixed - &nearly_singular)
        .mapv(|x| x.abs())
        .sum();

    println!(
        "\nReconstruction error (standard precision): {:.6e}",
        err_f32
    );
    println!("Reconstruction error (mixed precision): {:.6e}", err_mixed);

    // Example 5: Mixed-Precision Least Squares Solver
    println!("\nExample 5: Mixed-Precision Least Squares Solver");
    println!("------------------------------------------");

    // Create an ill-conditioned matrix for solving a least squares problem
    // Hilbert matrix is a classic example of an ill-conditioned matrix
    let size = 6;
    let mut hilbert_large = Array2::<f32>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            hilbert_large[[i, j]] = 1.0 / ((i + j + 1) as f32);
        }
    }

    // Create a right-hand side vector with noise
    let mut b_large = Array1::<f32>::zeros(size);
    b_large[0] = 1.0;
    // Add some noise to make the problem more challenging
    for i in 1..size {
        b_large[i] = 0.00001 * (i as f32);
    }

    println!("Ill-conditioned Hilbert matrix ({}x{})", size, size);
    println!("Condition number: Very high (approximately 1e8 for 6x6 Hilbert)");

    // Solve using standard precision
    let result_std = lstsq(&hilbert_large.view(), &b_large.view()).unwrap();

    // Solve using mixed precision
    let result_mixed =
        mixed_precision_lstsq::<f32, f32, f32, f64>(&hilbert_large.view(), &b_large.view())
            .unwrap();

    println!(
        "\nResiduals (standard f32 precision): {:.6e}",
        result_std.residuals
    );
    println!(
        "Residuals (mixed f32/f64 precision): {:.6e}",
        result_mixed.residuals
    );

    if result_mixed.residuals < result_std.residuals {
        println!(
            "Improvement factor: {:.2}x",
            result_std.residuals / result_mixed.residuals
        );
    }

    // Check the accuracy by computing A*x and comparing to b
    let ax_std = hilbert_large.dot(&result_std.x);
    let ax_mixed = hilbert_large.dot(&result_mixed.x);

    // Calculate error norms
    let err_std = (&ax_std - &b_large).mapv(|x| x.abs()).sum();
    let err_mixed = (&ax_mixed - &b_large).mapv(|x| x.abs()).sum();

    println!("\nL1 error with standard precision: {:.6e}", err_std);
    println!("L1 error with mixed precision: {:.6e}", err_mixed);

    // Example 6: Overdetermined System
    println!("\nExample 6: Overdetermined System with Mixed Precision");
    println!("----------------------------------------------");

    // Create an overdetermined system (more equations than unknowns)
    let a_over = array![
        [1.0f32, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0]
    ];
    let b_over = array![3.0f32, 5.0, 7.0, 9.0, 11.0];

    println!("Overdetermined system: 5 equations, 2 unknowns");
    println!("Expected solution: [1.0, 2.0]");

    // Solve using standard precision
    let result_std_over = lstsq(&a_over.view(), &b_over.view()).unwrap();

    // Solve using mixed precision
    let result_mixed_over =
        mixed_precision_lstsq::<f32, f32, f32, f64>(&a_over.view(), &b_over.view()).unwrap();

    println!(
        "\nSolution (standard precision): [{:.6}, {:.6}]",
        result_std_over.x[0], result_std_over.x[1]
    );
    println!(
        "Solution (mixed precision): [{:.6}, {:.6}]",
        result_mixed_over.x[0], result_mixed_over.x[1]
    );

    println!(
        "\nResiduals (standard precision): {:.6e}",
        result_std_over.residuals
    );
    println!(
        "Residuals (mixed precision): {:.6e}",
        result_mixed_over.residuals
    );

    // Example 7: Mixed-Precision Matrix Inversion
    println!("\nExample 7: Mixed-Precision Matrix Inversion");
    println!("------------------------------------------");

    // Create a particularly ill-conditioned matrix
    // Hilbert matrix of order 5 is extremely ill-conditioned
    let size = 5;
    let mut hilbert5 = Array2::<f32>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            hilbert5[[i, j]] = 1.0 / ((i + j + 1) as f32);
        }
    }

    println!("Hilbert matrix (5x5, extremely ill-conditioned):");
    for i in 0..size {
        println!("{}", hilbert5.row(i));
    }

    // Compute inversion using standard precision
    println!("\nComputing inverse with standard f32 precision...");
    let hilbert_inv_std = match inv(&hilbert5.view()) {
        Ok(inv) => inv,
        Err(e) => {
            println!("Standard precision inversion failed: {}", e);
            Array2::<f32>::zeros((size, size))
        }
    };

    // Compute inversion using mixed precision
    println!("Computing inverse with mixed precision (f32->f64->f32)...");
    let hilbert_inv_mixed = mixed_precision_inv::<_, f32, f64>(&hilbert5.view()).unwrap();

    // Verify the quality of inverses by computing A * A^(-1) (should be close to identity)
    let id_std = hilbert5.dot(&hilbert_inv_std);
    let id_mixed = hilbert5.dot(&hilbert_inv_mixed);

    // Calculate the total error in each case (deviation from identity matrix)
    let mut error_std = 0.0;
    let mut error_mixed = 0.0;

    for i in 0..size {
        for j in 0..size {
            let expected = if i == j { 1.0 } else { 0.0 };
            error_std += (id_std[[i, j]] - expected).abs();
            error_mixed += (id_mixed[[i, j]] - expected).abs();
        }
    }

    println!(
        "\nError from identity (standard precision): {:.6e}",
        error_std
    );
    println!("Error from identity (mixed precision): {:.6e}", error_mixed);

    if error_mixed < error_std {
        println!("Improvement factor: {:.2}x", error_std / error_mixed);
    }

    // Print a sample of the results to show difference
    println!("\nSample of A * A^(-1) with standard precision:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.6}  ", id_std[[i, j]]);
        }
        println!("...");
    }
    println!("...");

    println!("\nSample of A * A^(-1) with mixed precision:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.6}  ", id_mixed[[i, j]]);
        }
        println!("...");
    }
    println!("...");

    // Example 8: Mixed-Precision Determinant Calculation
    println!("\nExample 8: Mixed-Precision Determinant Calculation");
    println!("---------------------------------------------");

    // Create a Hilbert matrix (a classic ill-conditioned matrix)
    let size = 6;
    let mut hilbert6 = Array2::<f32>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            hilbert6[[i, j]] = 1.0 / ((i + j + 1) as f32);
        }
    }

    println!("Hilbert matrix (6x6, extremely ill-conditioned):");
    for i in 0..2 {
        println!("{}", hilbert6.row(i));
    }
    println!("...");

    // Compute determinant using standard precision
    println!("\nComputing determinant with standard f32 precision...");
    let det_std = match det(&hilbert6.view()) {
        Ok(det) => det,
        Err(e) => {
            println!("Standard precision determinant calculation failed: {}", e);
            0.0f32
        }
    };

    // Compute determinant using mixed precision
    println!("Computing determinant with mixed precision (f32->f64->f32)...");
    let det_mixed = mixed_precision_det::<_, f32, f64>(&hilbert6.view()).unwrap();

    // Compute determinant using higher precision directly
    // Convert matrix to f64 for a direct comparison
    let hilbert6_f64 = convert_2d::<f32, f64>(&hilbert6.view());
    let det_f64 = det(&hilbert6_f64.view()).unwrap() as f32;

    println!("\nDeterminant (standard f32 precision): {:.6e}", det_std);
    println!(
        "Determinant (mixed f32->f64->f32 precision): {:.6e}",
        det_mixed
    );
    println!("Determinant (direct f64 computation): {:.6e}", det_f64);

    // Compare accuracy using f64 as the reference
    let err_std = (det_std - det_f64).abs();
    let err_mixed = (det_mixed - det_f64).abs();

    println!(
        "\nError from f64 reference (standard precision): {:.6e}",
        err_std
    );
    println!(
        "Error from f64 reference (mixed precision): {:.6e}",
        err_mixed
    );

    if err_mixed < err_std {
        println!("Improvement factor: {:.2}x", err_std / err_mixed);
    } else {
        println!("Standard precision is sufficient for this case.");
    }

    // Summary
    println!("\nSummary of Mixed Precision Benefits");
    println!("================================");
    println!(
        "1. Cholesky decomposition: Improved numerical stability for positive definite matrices"
    );
    println!(
        "2. Iterative refinement: {:.2}x improvement for ill-conditioned systems",
        err_std / err_mixed
    );
    println!("3. Enhanced dot product: Better precision for sums with cancellation");
    println!("4. SVD: More accurate decomposition for nearly singular matrices");
    println!("5. Least squares solver: Better accuracy for ill-conditioned systems");
    println!("6. Matrix inversion: More accurate inverse for poorly conditioned matrices");
    println!("7. Determinant calculation: More accurate determinants for ill-conditioned matrices");
    println!("\nMixed precision techniques are especially valuable for:");
    println!("- Ill-conditioned systems");
    println!("- Computations with values of vastly different magnitudes");
    println!("- Algorithms requiring high accuracy intermediates");
    println!("- Applications where reducing precision sacrifices too much accuracy");
    println!("- Least squares problems with near-dependent columns");
    println!("- Matrix inversion of challenging matrices");
    println!("- Determinant calculation of large or ill-conditioned matrices");
}
