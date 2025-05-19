use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array2, Ix2};

/// Matrix inverse operation
pub struct MatrixInverseOp;

impl<F: Float> Op<F> for MatrixInverseOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Get input as ndarray
        let input_array = input.view();
        let shape = input_array.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix inverse requires square matrix".into(),
            ));
        }

        let n = shape[0];

        // Debug information
        println!("Computing matrix inverse of shape: {:?}", shape);

        let input_2d = input_array
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute inverse using Gauss-Jordan elimination
        let inv = compute_inverse(&input_2d)?;

        // Verify the shape of the result
        let result_shape = inv.shape();
        println!("Matrix inverse result shape: {:?}", result_shape);

        // No need to reshape, just use the computed inverse directly
        // but make a deep copy of it to ensure we have a clean array
        let output_inv = inv.to_owned();

        // Verify shape before output
        println!("Final inverse shape: {:?}", output_inv.shape());
        assert_eq!(output_inv.shape(), &[n, n]);

        // Append the array as output
        ctx.append_output(output_inv.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let output = ctx.output(); // This is the inverse
        let g = ctx.graph();

        // Evaluate tensors
        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Gradient of matrix inverse: -A^{-T} @ grad_output @ A^{-T}
        let inv = match output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_out_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let inv_t = inv.t();
        let temp = inv_t.dot(&grad_out_2d);
        let grad_input = -temp.dot(&inv_t);

        // Convert gradient to tensor
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// Matrix pseudo-inverse (Moore-Penrose) operation
pub struct PseudoInverseOp;

impl<F: Float> Op<F> for PseudoInverseOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute pseudo-inverse using SVD
        let pinv = compute_pseudo_inverse(&input_2d)?;

        ctx.append_output(pinv.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let output = ctx.output(); // This is the pseudo-inverse
        let input = ctx.input(0);
        let g = ctx.graph();

        // Evaluate tensors
        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D arrays
        let pinv = match output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_out_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_2d = match input_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Gradient calculation
        let pinv_t = pinv.t();
        let term1 = -pinv_t.dot(&grad_out_2d).dot(&pinv_t);
        let term2 = pinv_t
            .dot(&pinv_t.t())
            .dot(&grad_out_2d)
            .dot(&input_2d.t())
            .dot(&pinv_t);
        let term3 = pinv_t
            .dot(&input_2d.t())
            .dot(&grad_out_2d)
            .dot(&pinv_t.t())
            .dot(&pinv_t);

        let grad_input = term1 + term2 + term3;

        // Convert gradient to tensor
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// Matrix determinant for larger matrices
pub struct GeneralDeterminantOp;

impl<F: Float + ndarray::ScalarOperand> Op<F> for GeneralDeterminantOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Get input as ndarray
        let input_view = input.view();
        let shape = input_view.shape().to_vec(); // Clone the shape

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Determinant requires square matrix".into(),
            ));
        }

        let input_2d = input_view
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        println!("Computing determinant for matrix of shape: {:?}", shape);
        let det = compute_determinant_lu(&input_2d)?;
        println!("Determinant result: {}", det);

        // Create a scalar (0-dimensional) array with the determinant value
        // Use explicit arr0 to ensure we get a 0-dimensional array
        let det_array = ndarray::arr0(det);

        // Verify the shape to make sure we're creating a scalar
        assert_eq!(det_array.ndim(), 0);
        println!("Determinant array shape: {:?}", det_array.shape());

        // Output the determinant as a 0-dimensional array
        ctx.append_output(det_array.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let output = ctx.output();
        let g = ctx.graph();

        println!("Computing gradient for determinant");

        // Evaluate tensors
        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate gradient output");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate output");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate input");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Access scalar values
        let grad_scalar = grad_output_array[[0]];
        let det = output_array[[0]];
        println!("Determinant: {}, Gradient scale: {}", det, grad_scalar);

        // Gradient of determinant: det(A) * A^{-T}
        if det.abs() > F::epsilon() {
            let input_2d = match input_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    println!("Failed to convert input to 2D");
                    ctx.append_input_grad(0, None);
                    return;
                }
            };

            match compute_inverse(&input_2d) {
                Ok(inv) => {
                    // Scale transpose of inverse by det and grad_scalar
                    let inv_t = inv.t();
                    println!("Inverse transpose shape: {:?}", inv_t.shape());

                    // Correctly compute gradient: grad = grad_scalar * det * A^(-T)
                    let scaled_grad = inv_t.mapv(|x| det * grad_scalar * x);
                    let grad_tensor =
                        crate::tensor_ops::convert_to_tensor(scaled_grad.into_dyn(), g);

                    println!("Determinant gradient computed successfully");
                    ctx.append_input_grad(0, Some(grad_tensor));
                }
                Err(_) => {
                    println!("Matrix is nearly singular, using approximate gradient");
                    // For nearly singular matrices, use regularized inverse
                    let eps = F::epsilon() * F::from(10.0).unwrap();
                    let n = input_2d.shape()[0];
                    let regularized = &input_2d + &(Array2::<F>::eye(n) * eps);

                    if let Ok(reg_inv) = compute_inverse(&regularized.view()) {
                        let reg_inv_t = reg_inv.t();
                        let scaled_grad = reg_inv_t.mapv(|x| det * grad_scalar * x);
                        let grad_tensor =
                            crate::tensor_ops::convert_to_tensor(scaled_grad.into_dyn(), g);
                        ctx.append_input_grad(0, Some(grad_tensor));
                    } else {
                        println!("Failed to compute even regularized inverse, returning zeros");
                        let zeros = ndarray::Array::zeros(input_array.raw_dim());
                        let grad_tensor = crate::tensor_ops::convert_to_tensor(zeros, g);
                        ctx.append_input_grad(0, Some(grad_tensor));
                    }
                }
            }
            return;
        }

        println!("Matrix is singular, gradient is undefined, returning zeros");
        // If matrix is singular, gradient is undefined
        let zeros = ndarray::Array::zeros(input_array.raw_dim());
        let grad_tensor = crate::tensor_ops::convert_to_tensor(zeros, g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

// Helper functions
fn compute_inverse<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut inv = Array2::<F>::eye(n);

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < F::epsilon() {
            return Err(OpError::IncompatibleShape("Matrix is singular".into()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                a.swap((i, j), (max_row, j));
                inv.swap((i, j), (max_row, j));
            }
        }

        // Scale pivot row
        let pivot = a[[i, i]];
        for j in 0..n {
            a[[i, j]] /= pivot;
            inv[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = a[[k, i]];
                for j in 0..n {
                    a[[k, j]] = a[[k, j]] - factor * a[[i, j]];
                    inv[[k, j]] = inv[[k, j]] - factor * inv[[i, j]];
                }
            }
        }
    }

    Ok(inv)
}

fn compute_pseudo_inverse<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<Array2<F>, OpError> {
    // Simplified pseudo-inverse using transpose
    // For a full implementation, use SVD
    let m = matrix.shape()[0];
    let n = matrix.shape()[1];

    if m >= n {
        // A^+ = (A^T A)^(-1) A^T
        let at = matrix.t();
        let ata = at.dot(matrix);
        let ata_inv = compute_inverse(&ata.view())?;
        Ok(ata_inv.dot(&at))
    } else {
        // A^+ = A^T (A A^T)^(-1)
        let at = matrix.t();
        let aat = matrix.dot(&at);
        let aat_inv = compute_inverse(&aat.view())?;
        Ok(at.dot(&aat_inv))
    }
}

fn compute_determinant_lu<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<F, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut det = F::one();
    let mut swaps = 0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[[i, k]].abs() > a[[max_row, k]].abs() {
                max_row = i;
            }
        }

        if a[[max_row, k]].abs() < F::epsilon() {
            return Ok(F::zero()); // Singular matrix
        }

        // Swap rows
        if max_row != k {
            for j in k..n {
                a.swap((k, j), (max_row, j));
            }
            swaps += 1;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[[i, k]] / a[[k, k]];
            for j in (k + 1)..n {
                a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
            }
        }

        det *= a[[k, k]];
    }

    // Account for row swaps
    if swaps % 2 == 1 {
        det = -det;
    }

    Ok(det)
}

// Public API functions
pub fn matrix_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrix_shape = crate::tensor_ops::shape(matrix);

    // Build the tensor with shape information
    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixInverseOp)
}

pub fn pseudo_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(PseudoInverseOp)
}

pub fn determinant<'g, F: Float + ndarray::ScalarOperand>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // For determinant, we're creating a scalar output (0-dimensional tensor)
    // We'll use zeros(0) to create a scalar tensor shape
    let scalar_shape = crate::tensor_ops::zeros(&[0], g);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&scalar_shape)
        .build(GeneralDeterminantOp)
}
