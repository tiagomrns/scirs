//! Comprehensive example of linear algebra operations with autodiff support
//!
//! This example demonstrates the full range of automatic differentiation capabilities
//! for linear algebra operations, including:
//! - Basic operations (matmul, inv, det, etc.)
//! - Batch operations
//! - Matrix calculus (gradient, jacobian)
//! - Special matrix functions (pinv, sqrtm, logm)

#[cfg(feature = "autograd")]
fn main() {
    use ndarray::{array, Array};
    use scirs2_autograd::variable::Variable;
    use scirs2_linalg::prelude::autograd::*;

    println!("Comprehensive Linear Algebra Operations with Automatic Differentiation");
    println!("===================================================================");

    // 1. Basic Operations
    println!("\n1. Basic Matrix Operations");
    println!("-------------------------");

    // Create some input matrices and vectors with gradient tracking
    let a_data = array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn();
    let mut a = Variable::new(a_data, true);

    let b_data = array![[5.0_f64, 6.0], [7.0, 8.0]].into_dyn();
    let mut b = Variable::new(b_data, true);

    let x_data = array![1.0_f64, 2.0].into_dyn();
    let mut x = Variable::new(x_data, true);

    // Matrix multiplication
    let mut c = var_matmul(&a, &b).unwrap();
    println!("A = [[1, 2], [3, 4]]");
    println!("B = [[5, 6], [7, 8]]");
    println!("C = A @ B = \n{}", c.data().as_standard_layout());

    // Matrix inverse
    let mut a_inv = var_inv(&a).unwrap();
    println!("\nA^(-1) = \n{}", a_inv.data().as_standard_layout());

    // Determinant
    let mut det_a = var_det(&a).unwrap();
    println!("\ndet(A) = {}", det_a.data().as_standard_layout());

    // Matrix-vector multiplication
    let mut y = var_matvec(&a, &x).unwrap();
    println!("\nx = [1, 2]");
    println!("y = A @ x = {}", y.data().as_standard_layout());

    // 2. Matrix Decompositions
    println!("\n2. Matrix Decompositions");
    println!("----------------------");

    // SVD
    let (mut u, s, vt) = var_svd(&a).unwrap();
    println!("SVD of A:");
    println!("U = \n{}", u.data().as_standard_layout());
    println!("S = {}", s.data().as_standard_layout());
    println!("V^T = \n{}", vt.data().as_standard_layout());

    // Eigendecomposition
    let (mut eigenvals, eigenvecs) = var_eig(&a).unwrap();
    println!("\nEigendecomposition of A:");
    println!("Eigenvalues = {}", eigenvals.data().as_standard_layout());
    println!("Eigenvectors = \n{}", eigenvecs.data().as_standard_layout());

    // 3. Special Matrix Functions
    println!("\n3. Special Matrix Functions");
    println!("--------------------------");

    // Matrix exponential
    let mut expm_a = var_expm(&a).unwrap();
    println!("exp(A) = \n{}", expm_a.data().as_standard_layout());

    // Matrix square root
    let mut sqrtm_a = var_sqrtm(&a).unwrap();
    println!("\nsqrt(A) = \n{}", sqrtm_a.data().as_standard_layout());

    // Matrix logarithm
    let mut logm_a = var_logm(&a).unwrap();
    println!("\nlog(A) = \n{}", logm_a.data().as_standard_layout());

    // Pseudo-inverse
    let g_data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
    let mut g = Variable::new(g_data, true);
    let mut pinv_g = var_pinv(&g, None).unwrap();
    println!("\nG = [[1, 2], [3, 4], [5, 6]]");
    println!("pinv(G) = \n{}", pinv_g.data().as_standard_layout());

    // 4. Batch Operations
    println!("\n4. Batch Operations");
    println!("------------------");

    // Create batch of matrices
    let batch_a_data =
        array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0_f64, 6.0], [7.0, 8.0]],].into_dyn();
    let mut batch_a = Variable::new(batch_a_data, true);

    let batch_b_data = array![
        [[9.0_f64, 10.0], [11.0, 12.0]],
        [[13.0_f64, 14.0], [15.0, 16.0]],
    ]
    .into_dyn();
    let mut batch_b = Variable::new(batch_b_data, true);

    // Batch matrix multiplication
    let mut batch_c = var_batch_matmul(&batch_a, &batch_b).unwrap();
    println!("Batch A = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]");
    println!("Batch B = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]");
    println!("Batch A @ B = \n{}", batch_c.data().as_standard_layout());

    // Batch matrix inverse
    let mut batch_a_inv = var_batch_inv(&batch_a).unwrap();
    println!(
        "\nBatch A^(-1) = \n{}",
        batch_a_inv.data().as_standard_layout()
    );

    // Batch determinant
    let mut batch_det_a = var_batch_det(&batch_a).unwrap();
    println!(
        "\nBatch det(A) = {}",
        batch_det_a.data().as_standard_layout()
    );

    // 5. Matrix Calculus
    println!("\n5. Matrix Calculus");
    println!("-----------------");

    // Define a scalar function of a vector
    let scalar_fn =
        |x: &Variable<f64>| -> Result<Variable<f64>, scirs2_autograd::error::AutogradError> {
            // Function: f(x) = x_1^2 + 2*x_2^2
            let x_data = x.data();
            let result = x_data[0].powi(2) + 2.0 * x_data[1].powi(2);
            let result_data = array![result].into_dyn();
            Ok(Variable::new(result_data, x.tensor.requires_grad))
        };

    // Define a vector function of a vector
    let vector_fn =
        |x: &Variable<f64>| -> Result<Variable<f64>, scirs2_autograd::error::AutogradError> {
            // Function: f(x) = [x_1^2, x_1*x_2]
            let x_data = x.data();
            let result_data = array![x_data[0].powi(2), x_data[0] * x_data[1]].into_dyn();
            Ok(Variable::new(result_data, x.tensor.requires_grad))
        };

    // Create input vector
    let v_data = array![2.0_f64, 3.0].into_dyn();
    let v = Variable::new(v_data, true);

    // Compute gradient of scalar function
    let grad_f = var_gradient(scalar_fn, &v, None).unwrap();
    println!("v = [2, 3]");
    println!("f(v) = v_1^2 + 2*v_2^2 = 22");
    println!("∇f(v) = {}", grad_f.data().as_standard_layout());

    // Compute Jacobian of vector function
    let jac_f = var_jacobian(vector_fn, &v, None).unwrap();
    println!("\nf(v) = [v_1^2, v_1*v_2] = [4, 6]");
    println!("J_f(v) = \n{}", jac_f.data().as_standard_layout());

    // Compute Hessian of scalar function
    let hess_f = var_hessian(scalar_fn, &v, None).unwrap();
    println!("\nH_f(v) = \n{}", hess_f.data().as_standard_layout());

    // 6. Jacobian-Vector and Vector-Jacobian Products
    println!("\n6. Jacobian-Vector and Vector-Jacobian Products");
    println!("--------------------------------------------");

    // Create vector for JVP and VJP
    let p_data = array![1.0_f64, 1.0].into_dyn();
    let p = Variable::new(p_data, false);

    // Compute JVP: J_f(v) * p
    let jvp = var_jacobian_vector_product(vector_fn, &v, &p, None).unwrap();
    println!("p = [1, 1]");
    println!("J_f(v) * p = {}", jvp.data().as_standard_layout());

    // Compute VJP: p^T * J_f(v)
    let vjp = var_vector_jacobian_product(vector_fn, &v, &p).unwrap();
    println!("p^T * J_f(v) = {}", vjp.data().as_standard_layout());

    // 7. Gradient Propagation Examples
    println!("\n7. Gradient Propagation Examples");
    println!("------------------------------");

    // Reset gradients
    a.zero_grad();

    // Chain of operations with gradient tracking
    println!("Computing: L = trace(A^(-1) * exp(A))");

    let a_inv = var_inv(&a).unwrap();
    let expm_a = var_expm(&a).unwrap();
    let product = var_matmul(&a_inv, &expm_a).unwrap();
    let mut trace_result = var_trace(&product).unwrap();

    println!("Result = {}", trace_result.data().as_standard_layout());

    // Backpropagate
    trace_result.backward(None).unwrap();

    println!("dL/dA = \n{:?}", a.grad().unwrap());

    // Reset gradients and try another example
    a.zero_grad();

    println!("\nComputing: L = det(sqrtm(A)) * norm(A)");

    let sqrtm_a = var_sqrtm(&a).unwrap();
    let det_sqrtm_a = var_det(&sqrtm_a).unwrap();
    let norm_a = var_norm(&a, "fro").unwrap();
    let mut result = var_matmul(
        &det_sqrtm_a.reshape(&[1, 1]).unwrap(),
        &norm_a.reshape(&[1, 1]).unwrap(),
    )
    .unwrap();

    println!("Result = {}", result.data().as_standard_layout());

    // Backpropagate
    result.backward(None).unwrap();

    println!("dL/dA = \n{:?}", a.grad().unwrap());
}

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature.");
    println!("Run with: cargo run --example autodiff_comprehensive --features autograd");
}
