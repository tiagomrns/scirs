//! Example demonstrating advanced linear algebra operations with automatic differentiation
//!
//! This example shows how to use matrix inverse, SVD, eigendecomposition, and matrix
//! exponential with gradient tracking.

#[cfg(feature = "autograd")]
fn main() {
    use ndarray::{array, Array};
    use scirs2_autograd::variable::Variable;
    use scirs2_linalg::prelude::autograd::{
        var_det, var_eig, var_expm, var_inv, var_matmul, var_svd, var_trace,
    };

    println!("Advanced matrix operations with automatic differentiation");
    println!("=======================================================");

    // Create a 2x2 matrix with gradient tracking
    let a_data = array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn();
    let mut a = Variable::new(a_data, true);

    // 1. Matrix inverse with gradients
    println!("\n1. Matrix inverse gradient example");
    println!("A = [[1, 2], [3, 4]]");
    let mut a_inv = var_inv(&a).unwrap();
    println!("A^(-1) = \n{}", a_inv.data().as_standard_layout());

    // Sum of elements as a scalar loss
    let mut loss = a_inv.sum(None).unwrap();
    loss.backward(None).unwrap();

    println!("Loss = sum(A^(-1))");
    println!("dLoss/dA = \n{:?}", a.grad().unwrap());

    // Reset gradients
    a.zero_grad();

    // 2. Singular Value Decomposition (SVD) with gradients
    println!("\n2. SVD gradient example");
    let (mut u, s, vt) = var_svd(&a).unwrap();

    println!("SVD of A:");
    println!("U = \n{}", u.data().as_standard_layout());
    println!("S = \n{}", s.data().as_standard_layout());
    println!("V^T = \n{}", vt.data().as_standard_layout());

    // Compute gradient of U with respect to A
    let mut loss = u.sum(None).unwrap();
    loss.backward(None).unwrap();

    println!("Loss = sum(U)");
    println!("dLoss/dA = \n{:?}", a.grad().unwrap());

    // Reset gradients
    a.zero_grad();

    // 3. Eigendecomposition with gradients
    println!("\n3. Eigendecomposition gradient example");
    let (mut eigenvals, eigenvecs) = var_eig(&a).unwrap();

    println!(
        "Eigenvalues of A: {}",
        eigenvals.data().as_standard_layout()
    );
    println!(
        "Eigenvectors of A: \n{}",
        eigenvecs.data().as_standard_layout()
    );

    // Compute gradient of eigenvalues with respect to A
    let mut loss = eigenvals.sum(None).unwrap();
    loss.backward(None).unwrap();

    println!("Loss = sum(eigenvalues)");
    println!("dLoss/dA = \n{:?}", a.grad().unwrap());

    // Reset gradients
    a.zero_grad();

    // 4. Matrix exponential with gradients
    println!("\n4. Matrix exponential gradient example");
    let mut expm_a = var_expm(&a).unwrap();

    println!("exp(A) = \n{}", expm_a.data().as_standard_layout());

    // Compute gradient of matrix exponential with respect to A
    let mut loss = expm_a.sum(None).unwrap();
    loss.backward(None).unwrap();

    println!("Loss = sum(exp(A))");
    println!("dLoss/dA = \n{:?}", a.grad().unwrap());

    // Reset gradients
    a.zero_grad();

    // 5. Complex operation combining multiple advanced matrix operations
    println!("\n5. Complex operation example");
    println!("Computing: L = trace(A^(-1) * exp(A))");

    let a_inv = var_inv(&a).unwrap();
    let expm_a = var_expm(&a).unwrap();
    let product = var_matmul(&a_inv, &expm_a).unwrap();
    let mut trace_result = var_trace(&product).unwrap();

    println!("Result: {}", trace_result.data().as_standard_layout());

    // Compute gradient through the entire computation graph
    trace_result.backward(None).unwrap();

    println!("dL/dA = \n{:?}", a.grad().unwrap());
}

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature.");
    println!("Run with: cargo run --example autodiff_advanced --features autograd");
}
