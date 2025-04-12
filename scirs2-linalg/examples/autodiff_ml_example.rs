//! Machine learning example using autodiff for linear algebra operations
//!
//! This example demonstrates how to use the autodiff capabilities for a simple
//! machine learning task: training a linear regression model with L2 regularization
//! using pure linear algebra operations.

#[cfg(feature = "autograd")]
fn main() {
    use ndarray::{array, Array};
    use scirs2_autograd::variable::Variable;
    use scirs2_linalg::prelude::autograd::*;
    use std::f64::consts::PI;

    println!("Linear Regression with Autodiff");
    println!("==============================");

    // 1. Generate synthetic data for regression
    let num_samples = 20;
    let num_features = 3;

    // Generate input features
    let mut x_data = Array::zeros((num_samples, num_features));
    let mut y_data = Array::zeros(num_samples);

    let true_weights = array![2.5, -1.3, 0.7];
    let true_bias = 0.5;

    // Generate synthetic data with some noise: y = X*w + b + noise
    for i in 0..num_samples {
        // Generate features
        let x1 = (i as f64) / 10.0;
        let x2 = (i as f64 / 5.0).sin();
        let x3 = ((i as f64) / 4.0 * PI).cos();

        x_data[[i, 0]] = x1;
        x_data[[i, 1]] = x2;
        x_data[[i, 2]] = x3;

        // Compute true value with some noise
        let true_value =
            x1 * true_weights[0] + x2 * true_weights[1] + x3 * true_weights[2] + true_bias;
        let noise = (i as f64 / 5.0).sin() * 0.1;
        y_data[i] = true_value + noise;
    }

    println!("Generated synthetic data:");
    println!("X shape: {:?}", x_data.shape());
    println!("y shape: {:?}", y_data.shape());
    println!("True weights: {:?}", true_weights);
    println!("True bias: {}", true_bias);

    // 2. Initialize model parameters with random values
    let w_data = array![1.0, 1.0, 1.0].into_dyn();
    let mut w = Variable::new(w_data, true);

    let b_data = array![0.0].into_dyn();
    let mut b = Variable::new(b_data, true);

    let x = Variable::new(x_data.into_dyn(), false);
    let y = Variable::new(y_data.into_dyn(), false);

    // 3. Define hyperparameters
    let learning_rate = 0.01;
    let l2_reg = 0.1;
    let epochs = 200;

    println!("\nTraining with gradient descent:");
    println!("Learning rate: {}", learning_rate);
    println!("L2 regularization: {}", l2_reg);
    println!("Epochs: {}", epochs);

    // 4. Training loop
    for epoch in 0..epochs {
        // Forward pass: predict y = X*w + b
        let xw = var_matvec(&x, &w).unwrap();
        let y_pred = (&xw + &b.reshape(&[num_samples]).unwrap()).unwrap();

        // Compute loss: MSE + L2 regularization
        let diff = (&y_pred - &y).unwrap();
        let squared_diff = var_matmul(
            &diff.reshape(&[1, num_samples]).unwrap(),
            &diff.reshape(&[num_samples, 1]).unwrap(),
        )
        .unwrap();
        let mse =
            (&squared_diff / &Variable::new(array![num_samples as f64].into_dyn(), false)).unwrap();

        // L2 regularization term
        let w_squared = var_matmul(
            &w.reshape(&[1, num_features]).unwrap(),
            &w.reshape(&[num_features, 1]).unwrap(),
        )
        .unwrap();
        let l2_term = (&Variable::new(array![l2_reg].into_dyn(), false) * &w_squared).unwrap();

        // Total loss
        let mut loss = (&mse + &l2_term).unwrap();

        // Print progress every 20 epochs
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}: Loss = {}",
                epoch,
                loss.data().as_standard_layout()
            );
        }

        // Backward pass: compute gradients
        loss.backward(None).unwrap();

        // Update parameters using gradients
        if let Some(w_grad) = w.grad() {
            // Clone the gradients to avoid borrowing issues
            let w_grad_clone = w_grad.to_owned();
            for i in 0..num_features {
                w.tensor.data[i] = w.tensor.data[i] - learning_rate * w_grad_clone[i];
            }
        }

        if let Some(b_grad) = b.grad() {
            // Clone the gradient value to avoid borrowing issues
            let b_grad_val = b_grad[0];
            b.tensor.data[0] = b.tensor.data[0] - learning_rate * b_grad_val;
        }

        // Zero gradients for next iteration
        w.zero_grad();
        b.zero_grad();
    }

    // 5. Evaluate final model
    println!("\nTraining complete!");
    println!("Learned weights: {:?}", w.data().as_standard_layout());
    println!("Learned bias: {}", b.data().as_standard_layout());
    println!("True weights: {:?}", true_weights);
    println!("True bias: {}", true_bias);

    // 6. Make predictions on the training data
    let xw = var_matvec(&x, &w).unwrap();
    let y_pred = (&xw + &b.reshape(&[num_samples]).unwrap()).unwrap();

    // Compute R^2 score
    let y_mean = y.data().mean().unwrap();
    let mut ss_total = 0.0;
    let mut ss_residual = 0.0;

    for i in 0..num_samples {
        ss_total += (y.data()[i] - y_mean).powi(2);
        ss_residual += (y.data()[i] - y_pred.data()[i]).powi(2);
    }

    let r2_score = 1.0 - (ss_residual / ss_total);
    println!("\nR² Score: {:.4}", r2_score);

    // 7. Compare predictions with actual values
    println!("\nSample predictions vs actual values:");
    for i in 0..5 {
        println!(
            "Sample {}: Predicted = {:.4}, Actual = {:.4}",
            i,
            y_pred.data()[i],
            y.data()[i]
        );
    }

    // 8. Analytical solution using normal equations
    println!("\nAnalytical solution using normal equations:");

    // Compute X^T
    let x_t = var_transpose(&x).unwrap();

    // Compute X^T * X
    let xt_x = var_matmul(&x_t, &x).unwrap();

    // Add regularization term
    let reg_matrix = Variable::new(
        array![[l2_reg, 0.0, 0.0], [0.0, l2_reg, 0.0], [0.0, 0.0, l2_reg]].into_dyn(),
        false,
    );
    let xt_x_reg = (&xt_x + &reg_matrix).unwrap();

    // Compute (X^T * X + λI)^(-1)
    let xt_x_reg_inv = var_inv(&xt_x_reg).unwrap();

    // Compute X^T * y
    let xt_y = var_matvec(&x_t, &y).unwrap();

    // Compute w_analytical = (X^T * X + λI)^(-1) * X^T * y
    let w_analytical = var_matvec(&xt_x_reg_inv, &xt_y).unwrap();

    println!(
        "Analytical weights: {:?}",
        w_analytical.data().as_standard_layout()
    );
    println!("Learned weights (GD): {:?}", w.data().as_standard_layout());
    println!("True weights: {:?}", true_weights);

    // 9. Demonstrate advanced autodiff operations (SVD, QR decomposition)
    println!("\nAdvanced autodiff operations:");

    // SVD of X
    println!("\nSingular Value Decomposition:");
    let (u, s, vt) = var_svd(&x).unwrap();
    println!("Singular values: {:?}", s.data().as_standard_layout());

    // QR decomposition of X
    println!("\nQR Decomposition:");
    let (q, r) = var_qr(&x).unwrap();
    println!("R matrix (first 3x3):");
    for i in 0..3 {
        let row = [r.data()[[i, 0]], r.data()[[i, 1]], r.data()[[i, 2]]];
        println!("{:?}", row);
    }

    // Demonstrate using Cholesky decomposition for solving the normal equations
    println!("\nSolving normal equations with Cholesky decomposition:");

    let xt_x_spd = var_matmul(&x_t, &x).unwrap();
    let l = var_cholesky(&xt_x_spd).unwrap();

    println!("Cholesky factor L (first 3x3):");
    for i in 0..3 {
        let row = [l.data()[[i, 0]], l.data()[[i, 1]], l.data()[[i, 2]]];
        println!("{:?}", row);
    }
}

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature.");
    println!("Run with: cargo run --example autodiff_ml_example --features autograd");
}
