use ag::ndarray::Array2;
use ag::ndarray_ext::ArrayRng;
use ag::tensor_ops as T;
use scirs2_autograd as ag;

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_matrix_inverse() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a simple test matrix with known inverse
        let a_data = Array2::<f32>::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();

        // The inverse should be:
        // [ 0.6  -0.7 ]
        // [-0.2   0.4 ]

        println!("Test matrix A:\n{:?}", a_data);

        // Create a placeholder and feed the data
        let a = ctx.placeholder("a", &[2, 2]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Compute inverse of A
        let a_inv = T::matrix_inverse(&a);

        // Verify A * A^(-1) ≈ I
        let identity_approx = T::matmul(&a, &a_inv);

        // Evaluate with feeder
        let identity_result = ctx
            .evaluator()
            .push(&identity_approx)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("A * A^(-1):\n{:?}", identity_result);

        // Check if close to identity
        let identity = ag::ndarray::Array2::<f32>::eye(3);
        let error = (&identity_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap()
            - &identity)
            .mapv(|x| x.abs())
            .sum();

        println!("Error from identity: {}", error);
        assert!(error < 1e-5, "Matrix inverse failed, error: {}", error);

        // Test gradient computation for matrix inverse
        let y = T::sum_all(&T::matrix_inverse(&a));
        let grad_a = T::grad(&[y], &[a])[0];

        // Evaluate with feeder
        let grad_result = ctx
            .evaluator()
            .push(&grad_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!(
            "Gradient of sum(A^(-1)) with respect to A:\n{:?}",
            grad_result
        );

        // The gradient should not contain any NaNs or infinities
        let has_bad_values = grad_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        assert!(!has_bad_values, "Gradient contains NaN or infinite values");
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_determinant() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a test matrix with known determinant
        let mut a_data = ag::ndarray::Array2::<f32>::eye(3);
        a_data[[0, 0]] = 2.0;
        a_data[[1, 1]] = 3.0;
        a_data[[2, 2]] = 4.0;

        println!("Test matrix A:\n{:?}", a_data);
        // Det(A) should be 2*3*4 = 24

        // Create a placeholder and feed the data
        let a = ctx.placeholder("a", &[3, 3]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Compute determinant of A
        let det_a = T::determinant(&a);

        // Evaluate with feeder
        let det_result = ctx
            .evaluator()
            .push(&det_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("Det(A): {:?}", det_result);

        // Check if det is correct
        let expected_det = 24.0;
        let error = (det_result[[]] - expected_det).abs();

        println!("Determinant error: {}", error);
        assert!(error < 1e-5, "Determinant failed, error: {}", error);

        // Test gradient computation for determinant
        let grad_a = T::grad(&[det_a], &[a])[0];

        // Evaluate with feeder
        let grad_result = ctx
            .evaluator()
            .push(&grad_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("Gradient of det(A) with respect to A:\n{:?}", grad_result);

        // The gradient should match known values for diagonal matrix
        // ∂det(A)/∂A_ii = det(A)/A_ii for diagonal matrix
        let grad_00 = grad_result[[0, 0]];
        let expected_grad_00 = expected_det / a_data[[0, 0]];
        let grad_error = (grad_00 - expected_grad_00).abs();

        println!("Gradient error: {}", grad_error);
        assert!(
            grad_error < 1e-5,
            "Gradient of determinant failed, error: {}",
            grad_error
        );
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_matrix_solve() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a simple system Ax = b with known solution
        let a_data = ag::ndarray::array![[2.0, 1.0], [1.0, 3.0]];
        let b_data = ag::ndarray::array![[5.0], [10.0]];

        // Expected solution: x = [1, 3]^T
        let expected_x = ag::ndarray::array![[1.0], [3.0]];

        println!("A:\n{:?}", a_data);
        println!("b:\n{:?}", b_data);
        println!("Expected x:\n{:?}", expected_x);

        // Create placeholders and feed the data
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 1]);
        let feeder = ag::Feeder::new()
            .push(a, a_data.view().into_dyn())
            .push(b, b_data.view().into_dyn());

        // Solve the system
        let x = T::solve(&a, &b);

        // Evaluate with feeder
        let x_result = ctx.evaluator().push(&x).set_feeder(feeder.clone()).run()[0]
            .clone()
            .unwrap();

        println!("Solved x:\n{:?}", x_result);

        // Check if solution is correct
        let x_result_2d = x_result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();
        let error = (&x_result_2d - &expected_x).mapv(|x| x.abs()).sum();

        println!("Solution error: {}", error);
        assert!(error < 1e-5, "Linear solve failed, error: {}", error);

        // Test gradient computation for solver
        let y = T::sum_all(&x);
        let grad_a = T::grad(&[y], &[a])[0];
        let grad_b = T::grad(&[y], &[b])[0];

        // Evaluate with feeder
        let grad_results = ctx
            .evaluator()
            .push(&grad_a)
            .push(&grad_b)
            .set_feeder(feeder.clone())
            .run();

        let grad_a_result = grad_results[0].clone().unwrap();
        let grad_b_result = grad_results[1].clone().unwrap();

        println!("Gradient of sum(x) with respect to A:\n{:?}", grad_a_result);
        println!("Gradient of sum(x) with respect to b:\n{:?}", grad_b_result);

        // The gradients should not contain any NaNs or infinities
        let has_bad_values_a = grad_a_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        let has_bad_values_b = grad_b_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        assert!(
            !has_bad_values_a,
            "Gradient for A contains NaN or infinite values"
        );
        assert!(
            !has_bad_values_b,
            "Gradient for b contains NaN or infinite values"
        );
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_qr_decomposition() {
    ag::run::<f32, _, _>(|ctx| {
        let rng = &mut ArrayRng::<f32>::default();

        // Create a random matrix
        let a_data = rng.uniform(&[3, 3], -1.0, 1.0);
        println!("Test matrix A:\n{:?}", a_data);

        // Create a placeholder and feed the data
        let a = ctx.placeholder("a", &[3, 3]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Compute QR decomposition
        let (q, r) = T::qr(&a);

        // Evaluate with feeder
        let results = ctx
            .evaluator()
            .push(&q)
            .push(&r)
            .set_feeder(feeder.clone())
            .run();

        let q_result = results[0].clone().unwrap();
        let r_result = results[1].clone().unwrap();

        println!("Q:\n{:?}", q_result);
        println!("R:\n{:?}", r_result);

        // Check if Q is orthogonal (Q^T * Q ≈ I)
        let q_2d = q_result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();
        let q_t = q_2d.t();
        let q_orthogonal = q_t.dot(&q_2d);
        let identity = ag::ndarray::Array2::<f32>::eye(3);

        let orthogonal_error = (&q_orthogonal - &identity).mapv(|x| x.abs()).sum();
        println!("Orthogonality error: {}", orthogonal_error);
        assert!(
            orthogonal_error < 1e-5,
            "Q is not orthogonal, error: {}",
            orthogonal_error
        );

        // Check if A = Q * R
        let r_2d = r_result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();
        let a_reconstructed = q_2d.dot(&r_2d);

        let reconstruction_error = (&a_reconstructed - &a_data).mapv(|x| x.abs()).sum();
        println!("Reconstruction error: {}", reconstruction_error);
        assert!(
            reconstruction_error < 1e-5,
            "A ≠ QR, error: {}",
            reconstruction_error
        );

        // Test gradient computation
        let y = T::sum_all(&q) + T::sum_all(&r);
        let grad_a = T::grad(&[y], &[a])[0];

        // Evaluate with feeder
        let grad_result = ctx
            .evaluator()
            .push(&grad_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!(
            "Gradient of sum(Q) + sum(R) with respect to A:\n{:?}",
            grad_result
        );

        // The gradient should not contain any NaNs or infinities
        let has_bad_values = grad_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        assert!(!has_bad_values, "Gradient contains NaN or infinite values");
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_matrix_exp() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a simple matrix to test exponential
        let a_data = ag::ndarray::array![[0.0, 1.0], [-1.0, 0.0]];
        // This is a rotation matrix, exp(A) should be rotation by 1 radian

        println!("Test matrix A:\n{:?}", a_data);

        // Create a placeholder and feed the data
        let a = ctx.placeholder("a", &[2, 2]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Compute matrix exponential
        let exp_a = T::matrix_exp(&a);

        // Evaluate with feeder
        let exp_result = ctx
            .evaluator()
            .push(&exp_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("exp(A):\n{:?}", exp_result);

        // Expected result: rotation matrix by 1 radian
        let cos_1 = f32::cos(1.0);
        let sin_1 = f32::sin(1.0);
        let expected_exp = ag::ndarray::array![[cos_1, sin_1], [-sin_1, cos_1]];

        let exp_2d = exp_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap();
        let error = (&exp_2d - &expected_exp).mapv(|x| x.abs()).sum();

        println!("Matrix exponential error: {}", error);
        assert!(error < 1e-5, "Matrix exponential failed, error: {}", error);

        // Test gradient computation
        let y = T::sum_all(&exp_a);
        let grad_a = T::grad(&[y], &[a])[0];

        // Evaluate with feeder
        let grad_result = ctx
            .evaluator()
            .push(&grad_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!(
            "Gradient of sum(exp(A)) with respect to A:\n{:?}",
            grad_result
        );

        // The gradient should not contain any NaNs or infinities
        let has_bad_values = grad_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        assert!(!has_bad_values, "Gradient contains NaN or infinite values");
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_near_singular_matrix_operations() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a nearly singular matrix
        let mut a_data = Array2::<f32>::eye(3);

        // Make it nearly singular by making rows nearly linearly dependent
        a_data[[0, 0]] = 1.0;
        a_data[[0, 1]] = 1.0;
        a_data[[0, 2]] = 1.0;

        a_data[[1, 0]] = 1.0;
        a_data[[1, 1]] = 1.0001; // Tiny difference to avoid exact singularity
        a_data[[1, 2]] = 1.0;

        a_data[[2, 0]] = 1.0;
        a_data[[2, 1]] = 1.0;
        a_data[[2, 2]] = 1.0001; // Tiny difference to avoid exact singularity

        println!("Near-singular matrix A:\n{:?}", a_data);

        // Create placeholders and feed the data
        let a = ctx.placeholder("a", &[3, 3]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Test matrix inverse
        let a_inv = T::matrix_inverse(&a);

        // Evaluate with feeder
        let a_inv_result = ctx
            .evaluator()
            .push(&a_inv)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("A^(-1):\n{:?}", a_inv_result);

        // A*A^(-1) should still be approximately identity
        let identity_approx = T::matmul(&a, &a_inv);

        // Evaluate with feeder
        let identity_result = ctx
            .evaluator()
            .push(&identity_approx)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        let identity_2d = identity_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap();

        // Check if close to identity with a larger tolerance
        let identity = Array2::<f32>::eye(3);
        let error = (&identity_2d - &identity).mapv(|x| x.abs()).sum();

        println!("Error from identity for near-singular matrix: {}", error);
        // Use a larger tolerance for nearly singular matrices
        assert!(
            error < 1e-3,
            "Matrix inverse failed for near-singular matrix, error: {}",
            error
        );

        // Test determinant
        let det_a = T::determinant(&a);

        // Evaluate with feeder
        let det_result = ctx
            .evaluator()
            .push(&det_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("Det(A) for near-singular matrix: {:?}", det_result);

        // Determinant should be very small but not exactly zero
        assert!(
            det_result[[]] < 1e-3,
            "Determinant should be very small for near-singular matrix"
        );
        assert!(
            det_result[[]] > 0.0,
            "Determinant should not be exactly zero"
        );

        // Test linear solver
        let b_data = ag::ndarray::array![[3.0], [3.0001], [3.0001]];
        let b = ctx.placeholder("b", &[3, 1]);
        let feeder_ab = ag::Feeder::new()
            .push(a, a_data.view().into_dyn())
            .push(b, b_data.view().into_dyn());

        let x = T::solve(&a, &b);

        // Evaluate with feeder
        let x_result = ctx.evaluator().push(&x).set_feeder(feeder_ab.clone()).run()[0]
            .clone()
            .unwrap();

        println!("Solution x for near-singular matrix:\n{:?}", x_result);

        // Verify Ax ≈ b
        let b_reconstructed = T::matmul(&a, &x);

        // Evaluate with feeder
        let b_reconstructed_result = ctx
            .evaluator()
            .push(&b_reconstructed)
            .set_feeder(feeder_ab.clone())
            .run()[0]
            .clone()
            .unwrap();

        let b_reconstructed_2d = b_reconstructed_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap();

        let b_error = (&b_reconstructed_2d - &b_data).mapv(|x| x.abs()).sum();
        println!("Error in Ax = b for near-singular matrix: {}", b_error);
        assert!(
            b_error < 1e-3,
            "Linear solve failed for near-singular matrix, error: {}",
            b_error
        );

        // Test gradients with higher tolerance
        println!("Testing gradients for near-singular matrices...");

        // Test inverse gradient
        let y1 = T::sum_all(&T::matrix_inverse(&a));
        let grad_a1 = T::grad(&[y1], &[a])[0];

        // Test determinant gradient
        let y2 = T::determinant(&a);
        let grad_a2 = T::grad(&[y2], &[a])[0];

        // Test solver gradient
        let y3 = T::sum_all(&T::solve(&a, &b));
        let grad_a3 = T::grad(&[y3], &[a])[0];

        // Evaluate all gradients with feeder
        let grad_results1 = ctx
            .evaluator()
            .push(&grad_a1)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        let grad_results2 = ctx
            .evaluator()
            .push(&grad_a2)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        let grad_results3 = ctx
            .evaluator()
            .push(&grad_a3)
            .set_feeder(feeder_ab.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!(
            "Gradient of sum(A^(-1)) for near-singular matrix:\n{:?}",
            grad_results1
        );
        println!(
            "Gradient of det(A) for near-singular matrix:\n{:?}",
            grad_results2
        );
        println!(
            "Gradient of sum(solve(A,b)) for near-singular matrix:\n{:?}",
            grad_results3
        );

        // Check that all gradients are finite (no NaNs or infinities)
        let has_bad_values1 = grad_results1.iter().any(|&x| x.is_nan() || x.is_infinite());
        let has_bad_values2 = grad_results2.iter().any(|&x| x.is_nan() || x.is_infinite());
        let has_bad_values3 = grad_results3.iter().any(|&x| x.is_nan() || x.is_infinite());

        assert!(
            !has_bad_values1,
            "Gradient of inverse contains NaN or infinite values"
        );
        assert!(
            !has_bad_values2,
            "Gradient of determinant contains NaN or infinite values"
        );
        assert!(
            !has_bad_values3,
            "Gradient of solve contains NaN or infinite values"
        );
    });
}

#[test]
#[ignore = "Placeholder/feeder system needs investigation"]
fn test_matrix_sqrt() {
    ag::run::<f32, _, _>(|ctx| {
        // Create a positive definite matrix for testing matrix square root
        let a_data = ag::ndarray::array![[4.0, 1.0], [1.0, 9.0]];

        println!("Test matrix A for sqrt:\n{:?}", a_data);

        // Create a placeholder and feed the data
        let a = ctx.placeholder("a", &[2, 2]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Compute matrix square root
        let sqrt_a = T::matrix_sqrt(&a);

        // Evaluate with feeder
        let sqrt_result = ctx
            .evaluator()
            .push(&sqrt_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!("sqrt(A):\n{:?}", sqrt_result);

        // Verify sqrt(A) * sqrt(A) ≈ A
        let sqrt_squared = T::matmul(&sqrt_a, &sqrt_a);

        // Evaluate with feeder
        let sqrt_squared_result = ctx
            .evaluator()
            .push(&sqrt_squared)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        let sqrt_squared_2d = sqrt_squared_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap();

        let error = (&sqrt_squared_2d - &a_data).mapv(|x| x.abs()).sum();
        println!("Matrix square root verification error: {}", error);
        assert!(error < 1e-5, "Matrix square root failed, error: {}", error);

        // Test gradient computation
        let y = T::sum_all(&sqrt_a);
        let grad_a = T::grad(&[y], &[a])[0];

        // Evaluate with feeder
        let grad_result = ctx
            .evaluator()
            .push(&grad_a)
            .set_feeder(feeder.clone())
            .run()[0]
            .clone()
            .unwrap();

        println!(
            "Gradient of sum(sqrt(A)) with respect to A:\n{:?}",
            grad_result
        );

        // The gradient should not contain any NaNs or infinities
        let has_bad_values = grad_result.iter().any(|&x| x.is_nan() || x.is_infinite());
        assert!(!has_bad_values, "Gradient contains NaN or infinite values");
    });
}
