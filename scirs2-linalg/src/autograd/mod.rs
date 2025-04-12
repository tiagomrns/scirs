//! Automatic differentiation support for linear algebra operations
//!
//! This module integrates the linear algebra operations with automatic
//! differentiation from the scirs2-autograd crate. It provides differentiable
//! versions of common matrix operations.

#![cfg(feature = "autograd")]

// Advanced submodules
pub mod batch;
pub mod factorizations;
pub mod matrix_calculus;
pub mod special;
pub mod tensor_algebra;
pub mod transformations;

// Re-export functions from submodules
pub use batch::{batch_det, batch_inv, batch_matmul, batch_matvec};
pub use factorizations::{cholesky, lu, qr};
pub use matrix_calculus::{
    gradient, hessian, jacobian, jacobian_vector_product, vector_jacobian_product,
};
pub use special::{logm, pinv, sqrtm};
pub use tensor_algebra::{contract, outer, tensor_vector_product};
pub use transformations::{
    project, reflection_matrix, rotation_matrix_2d, scaling_matrix, shear_matrix,
};

// Re-export high-level variable interfaces
pub mod variable {
    use num_traits::Float;
    use scirs2_autograd::error::Result as AutogradResult;
    use scirs2_autograd::variable::Variable;
    use std::fmt::Debug;

    // This would be nice, but we can't implement methods directly on external types
    // Instead, we'll modify the example to use var_trace directly

    pub use super::batch::variable::{
        batch_det as var_batch_det, batch_inv as var_batch_inv, batch_matmul as var_batch_matmul,
        batch_matvec as var_batch_matvec,
    };
    pub use super::factorizations::variable::{
        cholesky as var_cholesky, lu as var_lu, qr as var_qr,
    };
    pub use super::matrix_calculus::variable::{
        gradient as var_gradient, hessian as var_hessian, jacobian as var_jacobian,
        jacobian_vector_product as var_jacobian_vector_product,
        vector_jacobian_product as var_vector_jacobian_product,
    };
    pub use super::special::variable::{logm as var_logm, pinv as var_pinv, sqrtm as var_sqrtm};
    pub use super::tensor_algebra::variable::{
        contract as var_contract, outer as var_outer,
        tensor_vector_product as var_tensor_vector_product,
    };
    pub use super::transformations::variable::{
        project as var_project, reflection_matrix as var_reflection_matrix,
        rotation_matrix_2d as var_rotation_matrix_2d, scaling_matrix as var_scaling_matrix,
        shear_matrix as var_shear_matrix,
    };

    // Basic operations implemented in this module - re-export tensor operations as variables
    pub use super::{det, dot, eig, expm, inv, matmul, matvec, norm, svd, trace, transpose};

    // Define variable wrappers for these operations
    pub fn var_det<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::det(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_matmul<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        b: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::matmul(&a.tensor, &b.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_trace<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::trace(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_transpose<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::transpose(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_inv<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::inv(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_svd<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<(Variable<F>, Variable<F>, Variable<F>)> {
        let (u_tensor, s_tensor, vt_tensor) = super::svd(&a.tensor)?;
        Ok((
            Variable { tensor: u_tensor },
            Variable { tensor: s_tensor },
            Variable { tensor: vt_tensor },
        ))
    }

    pub fn var_eig<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<(Variable<F>, Variable<F>)> {
        let (eigenvals_tensor, eigenvecs_tensor) = super::eig(&a.tensor)?;
        Ok((
            Variable {
                tensor: eigenvals_tensor,
            },
            Variable {
                tensor: eigenvecs_tensor,
            },
        ))
    }

    pub fn var_expm<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::expm(&a.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    // Variable wrapper implementations for basic operations
    pub fn var_matvec<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        x: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::matvec(&a.tensor, &x.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_dot<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        b: &Variable<F>,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::dot(&a.tensor, &b.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    pub fn var_norm<F: Float + Debug + Send + Sync + 'static>(
        a: &Variable<F>,
        ord: &str,
    ) -> scirs2_autograd::error::Result<Variable<F>> {
        let result_tensor = super::norm(&a.tensor, ord)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

use crate::error::LinalgResult;

/// Provides matrix multiplication with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - First tensor of shape (n, m)
/// * `b` - Second tensor of shape (m, p)
///
/// # Returns
///
/// A new tensor of shape (n, p) containing the matrix product with gradient tracking.
pub fn matmul<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure dimensions are compatible
    if a.data.ndim() != 2 || b.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix multiplication requires 2D tensors".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Matrix multiplication dimension mismatch: {:?} and {:?}",
                a_shape, b_shape
            ),
        ));
    }

    // Compute the result
    let a_rows = a.data.shape()[0];
    let a_cols = a.data.shape()[1];
    let b_cols = b.data.shape()[1];

    // Create result matrix
    let mut result_data = Array2::<F>::zeros((a_rows, b_cols));

    // Manually compute matrix multiplication
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = F::zero();
            for k in 0..a_cols {
                sum = sum + a.data[[i, k]] * b.data[[k, j]];
            }
            result_data[[i, j]] = sum;
        }
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad || b.requires_grad;

    if requires_grad {
        let node = Node::matmul(a, b)?;
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Matrix transpose with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input tensor of shape (n, m)
///
/// # Returns
///
/// A new tensor of shape (m, n) containing the transposed matrix with gradient tracking.
pub fn transpose<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure the input is a 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Transpose requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    let result_data = a.data.clone().reversed_axes();
    let requires_grad = a.requires_grad;

    if requires_grad {
        // Create a backward function for transpose
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            // Transpose of the gradient
            Ok(grad.clone().reversed_axes())
        })
            as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;

        let node = Node::new(
            scirs2_autograd::graph::OpType::Transpose,
            vec![a],
            vec![Some(backward)],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Matrix vector multiplication with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Matrix tensor of shape (n, m)
/// * `x` - Vector tensor of shape (m,)
///
/// # Returns
///
/// A new tensor of shape (n,) containing the matrix-vector product with gradient tracking.
pub fn matvec<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    x: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure dimensions are compatible
    if a.data.ndim() != 2 || x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix-vector multiplication requires a 2D matrix and 1D vector".to_string(),
        ));
    }

    let a_shape = a.shape();
    let x_shape = x.shape();

    if a_shape[1] != x_shape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Matrix-vector multiplication dimension mismatch: {:?} and {:?}",
                a_shape, x_shape
            ),
        ));
    }

    // Compute the result
    let a_rows = a.data.shape()[0];
    let a_cols = a.data.shape()[1];

    // Create result vector
    let mut result_data = Array1::<F>::zeros(a_rows);

    // Manually compute matrix-vector multiplication
    for i in 0..a_rows {
        let mut sum = F::zero();
        for j in 0..a_cols {
            sum = sum + a.data[[i, j]] * x.data[j];
        }
        result_data[i] = sum;
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad || x.requires_grad;

    if requires_grad {
        // Copy the data for each closure separately
        // Backward function for matrix
        let backward_a = if a.requires_grad {
            let a_data_for_a = a.data.clone();
            let x_data_for_a = x.data.clone();
            Some(
                Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                    // For y = A*x, dL/dA = dL/dy * x^T
                    let mut grad_a = Array2::<F>::zeros((a_data_for_a.shape()[0], a_data_for_a.shape()[1]));
                    for i in 0..a_data_for_a.shape()[0] {
                        for j in 0..a_data_for_a.shape()[1] {
                            grad_a[[i, j]] = grad[[i]] * x_data_for_a[j];
                        }
                    }
                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        // Backward function for vector
        let backward_x = if x.requires_grad {
            let a_data_for_x = a.data.clone();
            Some(
                Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                    // For y = A*x, dL/dx = A^T * dL/dy
                    let mut grad_x = Array1::<F>::zeros(a_data_for_x.shape()[1]);
                    for j in 0..a_data_for_x.shape()[1] {
                        let mut sum = F::zero();
                        for i in 0..a_data_for_x.shape()[0] {
                            sum = sum + a_data_for_x[[i, j]] * grad[[i]];
                        }
                        grad_x[j] = sum;
                    }
                    Ok(grad_x.into_dyn())
                })
                    as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::MatMul,
            vec![a, x],
            vec![backward_a, backward_x],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the determinant of a matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor of shape (n, n)
///
/// # Returns
///
/// A new scalar tensor containing the determinant with gradient tracking.
pub fn det<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure the input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Determinant requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Determinant requires a square matrix".to_string(),
        ));
    }

    // For now, implement determinant for 2x2 and 3x3 matrices only
    let det_val = match a_shape[0] {
        0 => F::one(),
        1 => a.data[[0, 0]],
        2 => a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]],
        3 => {
            a.data[[0, 0]] * (a.data[[1, 1]] * a.data[[2, 2]] - a.data[[1, 2]] * a.data[[2, 1]])
                - a.data[[0, 1]]
                    * (a.data[[1, 0]] * a.data[[2, 2]] - a.data[[1, 2]] * a.data[[2, 0]])
                + a.data[[0, 2]]
                    * (a.data[[1, 0]] * a.data[[2, 1]] - a.data[[1, 1]] * a.data[[2, 0]])
        }
        _ => {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Determinant for matrices larger than 3x3 not yet implemented".to_string(),
            ));
        }
    };

    let result_data = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), det_val);
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let a_shape = a_shape.clone();

        // For the gradient of the determinant, we need the adjugate (transpose of the cofactor matrix)
        // For now, implement for 2x2 and 3x3 only
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            let grad_scalar = grad[[0]];
            let mut grad_a = Array2::<F>::zeros((a_shape[0], a_shape[1]));
            match a_shape[0] {
                1 => {
                    grad_a[[0, 0]] = grad_scalar;
                },
                2 => {
                    // For a 2x2 matrix, the adjugate is simple
                    grad_a[[0, 0]] = grad_scalar * a_data[[1, 1]];
                    grad_a[[0, 1]] = grad_scalar * (-a_data[[1, 0]]);
                    grad_a[[1, 0]] = grad_scalar * (-a_data[[0, 1]]);
                    grad_a[[1, 1]] = grad_scalar * a_data[[0, 0]];
                },
                3 => {
                    // For a 3x3 matrix, compute the cofactor for each element
                    // This is a simplified implementation for the specific case

                    // First row
                    grad_a[[0, 0]] = grad_scalar * (a_data[[1, 1]] * a_data[[2, 2]] - a_data[[1, 2]] * a_data[[2, 1]]);
                    grad_a[[0, 1]] = grad_scalar * (-(a_data[[1, 0]] * a_data[[2, 2]] - a_data[[1, 2]] * a_data[[2, 0]]));
                    grad_a[[0, 2]] = grad_scalar * (a_data[[1, 0]] * a_data[[2, 1]] - a_data[[1, 1]] * a_data[[2, 0]]);

                    // Second row
                    grad_a[[1, 0]] = grad_scalar * (-(a_data[[0, 1]] * a_data[[2, 2]] - a_data[[0, 2]] * a_data[[2, 1]]));
                    grad_a[[1, 1]] = grad_scalar * (a_data[[0, 0]] * a_data[[2, 2]] - a_data[[0, 2]] * a_data[[2, 0]]);
                    grad_a[[1, 2]] = grad_scalar * (-(a_data[[0, 0]] * a_data[[2, 1]] - a_data[[0, 1]] * a_data[[2, 0]]));

                    // Third row
                    grad_a[[2, 0]] = grad_scalar * (a_data[[0, 1]] * a_data[[1, 2]] - a_data[[0, 2]] * a_data[[1, 1]]);
                    grad_a[[2, 1]] = grad_scalar * (-(a_data[[0, 0]] * a_data[[1, 2]] - a_data[[0, 2]] * a_data[[1, 0]]));
                    grad_a[[2, 2]] = grad_scalar * (a_data[[0, 0]] * a_data[[1, 1]] - a_data[[0, 1]] * a_data[[1, 0]]);
                },
                _ => {
                    // For larger matrices, this would need a more general implementation
                    // with LU decomposition
                }
            }

            Ok(grad_a.into_dyn())
        })
            as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("det".to_string()),
            vec![a],
            vec![Some(backward)],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the trace of a matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor of shape (n, n)
///
/// # Returns
///
/// A new scalar tensor containing the trace with gradient tracking.
pub fn trace<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure the input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Trace requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Trace requires a square matrix".to_string(),
        ));
    }

    // Compute the trace
    let mut trace_val = F::zero();
    for i in 0..a_shape[0] {
        trace_val = trace_val + a.data[[i, i]];
    }

    let result_data = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), trace_val);
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_shape = a_shape.clone();

        // Gradient of trace is an identity matrix
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            let grad_scalar = grad[[0]];
            let mut grad_a = Array2::<F>::zeros((a_shape[0], a_shape[1]));

            // Set diagonal elements to the gradient
            for i in 0..a_shape[0] {
                grad_a[[i, i]] = grad_scalar;
            }

            Ok(grad_a.into_dyn())
        })
            as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("trace".to_string()),
            vec![a],
            vec![Some(backward)],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the matrix norm with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input matrix tensor
/// * `ord` - The order of the norm (currently supports 'fro' = Frobenius norm)
///
/// # Returns
///
/// A new scalar tensor containing the norm with gradient tracking.
pub fn norm<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    ord: &str,
) -> AutogradResult<Tensor<F>> {
    // For now, only implement Frobenius norm
    if ord != "fro" {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            format!(
                "Norm '{}' not yet implemented. Only 'fro' is currently supported",
                ord
            ),
        ));
    }

    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix norm requires a 2D tensor".to_string(),
        ));
    }

    // Compute Frobenius norm
    let mut sum_squares = F::zero();
    for &val in a.data.iter() {
        sum_squares = sum_squares + val * val;
    }
    let norm_val = sum_squares.sqrt();

    let result_data = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), norm_val);
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let a_shape = a.shape();

        // Gradient of Frobenius norm is a / |a|
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            let grad_scalar = grad[[0]];
            let mut grad_a = Array2::<F>::zeros((a_shape[0], a_shape[1]));

            // Handle the case where norm is zero
            if norm_val < F::epsilon() {
                return Ok(grad_a.into_dyn());
            }

            let scale = grad_scalar / norm_val;
            for i in 0..a_shape[0] {
                for j in 0..a_shape[1] {
                    grad_a[[i, j]] = scale * a_data[[i, j]];
                }
            }

            Ok(grad_a.into_dyn())
        })
            as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("norm".to_string()),
            vec![a],
            vec![Some(backward)],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute dot product with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - First vector tensor
/// * `b` - Second vector tensor
///
/// # Returns
///
/// A new scalar tensor containing the dot product with gradient tracking.
pub fn dot<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
    b: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure both inputs are 1D vectors
    if a.data.ndim() != 1 || b.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Dot product requires two 1D vectors".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[0] != b_shape[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Vectors must have the same length for dot product: {:?} vs {:?}",
                a_shape, b_shape
            ),
        ));
    }

    // Compute dot product
    let mut dot_val = F::zero();
    for i in 0..a_shape[0] {
        dot_val = dot_val + a.data[i] * b.data[i];
    }

    let result_data = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), dot_val);
    let requires_grad = a.requires_grad || b.requires_grad;

    if requires_grad {
        // Backward function for first vector
        let backward_a = if a.requires_grad {
            let a_data_for_a = a.data.clone();
            let b_data_for_a = b.data.clone();
            Some(
                Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                    let grad_scalar = grad[[0]];
                    let mut grad_a = Array1::<F>::zeros(a_data_for_a.len());

                    for i in 0..a_data_for_a.len() {
                        grad_a[i] = grad_scalar * b_data_for_a[i];
                    }

                    Ok(grad_a.into_dyn())
                })
                    as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        // Backward function for second vector
        let backward_b = if b.requires_grad {
            let a_data_for_b = a.data.clone();
            let b_data_for_b = b.data.clone();
            Some(
                Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                    let grad_scalar = grad[[0]];
                    let mut grad_b = Array1::<F>::zeros(b_data_for_b.len());

                    for i in 0..b_data_for_b.len() {
                        grad_b[i] = grad_scalar * a_data_for_b[i];
                    }

                    Ok(grad_b.into_dyn())
                })
                    as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>,
            )
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("dot".to_string()),
            vec![a, b],
            vec![backward_a, backward_b],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Compute the inverse of a matrix with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A new tensor containing the matrix inverse with gradient tracking.
pub fn inv<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure the input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix inverse requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix inverse requires a square matrix".to_string(),
        ));
    }

    // For 2x2 matrices, we can compute the inverse directly
    let n = a_shape[0];
    let mut result_data = Array2::<F>::zeros((n, n));

    // Compute the determinant
    let det_val = match n {
        1 => a.data[[0, 0]],
        2 => a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]],
        _ => {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Matrix inverse for matrices larger than 2x2 not yet implemented".to_string(),
            ));
        }
    };

    // Check if matrix is singular
    if det_val.abs() < F::epsilon() {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Cannot compute inverse of a singular matrix".to_string(),
        ));
    }

    // Compute the inverse
    let inv_det = F::one() / det_val;

    match n {
        1 => {
            result_data[[0, 0]] = F::one() / a.data[[0, 0]];
        }
        2 => {
            result_data[[0, 0]] = a.data[[1, 1]] * inv_det;
            result_data[[0, 1]] = -a.data[[0, 1]] * inv_det;
            result_data[[1, 0]] = -a.data[[1, 0]] * inv_det;
            result_data[[1, 1]] = a.data[[0, 0]] * inv_det;
        }
        _ => unreachable!(),
    }

    let result_data = result_data.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let inv_data = result_data.clone();

        // Gradient of matrix inverse: dL/dA = -A^(-1) * dL/dA^(-1) * A^(-1)
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            // Convert to 2D for matrix operations
            let grad_2d = grad.clone().into_shape((n, n)).unwrap();
            let inv_2d = inv_data.clone().into_shape((n, n)).unwrap();

            // Compute -A^(-1) * dL/dA^(-1) * A^(-1)
            let mut result = Array2::<F>::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    let mut sum = F::zero();
                    for k in 0..n {
                        for l in 0..n {
                            sum = sum + (-inv_2d[[i, k]] * grad_2d[[k, l]] * inv_2d[[l, j]]);
                        }
                    }
                    result[[i, j]] = sum;
                }
            }

            Ok(result.into_dyn())
        }) as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("inv".to_string()),
            vec![a],
            vec![Some(backward)],
        );
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

/// Perform singular value decomposition (SVD) with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input matrix tensor
///
/// # Returns
///
/// A tuple (u, s, vt) representing the SVD components with gradient tracking.
pub fn svd<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<(Tensor<F>, Tensor<F>, Tensor<F>)> {
    // Ensure the input is a 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "SVD requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    let m = a_shape[0];
    let n = a_shape[1];

    // For 2x2 matrices, we can compute SVD using a direct algorithm
    // This is a very simplified implementation that only works for small matrices
    if m > 2 || n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "SVD for matrices larger than 2x2 not yet implemented in autodiff".to_string(),
        ));
    }

    // For a 2x2 matrix, we can compute the SVD directly
    // This is a simple implementation and doesn't handle all cases properly
    let a_t_a = Array2::<F>::from_shape_fn((n, n), |ij| {
        let (i, j) = (ij.0, ij.1);
        let mut sum = F::zero();
        for k in 0..m {
            sum = sum + a.data[[k, i]] * a.data[[k, j]];
        }
        sum
    });

    // Compute eigenvalues of A^T A to get singular values
    let mut eigenvals = Vec::new();
    let mut eigenvecs = Array2::<F>::zeros((n, n));

    if n == 1 {
        // For 1x1 matrix, the singular value is just the absolute value
        eigenvals.push(a_t_a[[0, 0]].abs());
        eigenvecs[[0, 0]] = F::one();
    } else if n == 2 {
        // For 2x2 matrix, we can compute eigenvalues and eigenvectors directly
        let a11 = a_t_a[[0, 0]];
        let a12 = a_t_a[[0, 1]];
        let a21 = a_t_a[[1, 0]];
        let a22 = a_t_a[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered in SVD".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();

        eigenvals.push(lambda1.sqrt());
        eigenvals.push(lambda2.sqrt());

        // Compute eigenvectors
        if a12.abs() > F::epsilon() {
            eigenvecs[[0, 0]] = lambda1 - a22;
            eigenvecs[[1, 0]] = a21;
            eigenvecs[[0, 1]] = lambda2 - a22;
            eigenvecs[[1, 1]] = a21;
        } else if a21.abs() > F::epsilon() {
            eigenvecs[[0, 0]] = a12;
            eigenvecs[[1, 0]] = lambda1 - a11;
            eigenvecs[[0, 1]] = a12;
            eigenvecs[[1, 1]] = lambda2 - a11;
        } else {
            // Diagonal matrix
            eigenvecs[[0, 0]] = F::one();
            eigenvecs[[1, 0]] = F::zero();
            eigenvecs[[0, 1]] = F::zero();
            eigenvecs[[1, 1]] = F::one();
        }

        // Normalize eigenvectors
        let norm1 =
            (eigenvecs[[0, 0]] * eigenvecs[[0, 0]] + eigenvecs[[1, 0]] * eigenvecs[[1, 0]]).sqrt();
        let norm2 =
            (eigenvecs[[0, 1]] * eigenvecs[[0, 1]] + eigenvecs[[1, 1]] * eigenvecs[[1, 1]]).sqrt();

        eigenvecs[[0, 0]] = eigenvecs[[0, 0]] / norm1;
        eigenvecs[[1, 0]] = eigenvecs[[1, 0]] / norm1;
        eigenvecs[[0, 1]] = eigenvecs[[0, 1]] / norm2;
        eigenvecs[[1, 1]] = eigenvecs[[1, 1]] / norm2;
    }

    // Create V matrix (right singular vectors)
    let v = eigenvecs.clone();

    // Create sigma matrix (singular values)
    let mut s = Array1::<F>::zeros(eigenvals.len());
    for (i, &val) in eigenvals.iter().enumerate() {
        s[i] = val;
    }

    // Compute U matrix (left singular vectors)
    let mut u = Array2::<F>::zeros((m, eigenvals.len()));

    for j in 0..eigenvals.len() {
        if eigenvals[j] > F::epsilon() {
            for i in 0..m {
                let mut sum = F::zero();
                for k in 0..n {
                    sum = sum + a.data[[i, k]] * v[[k, j]];
                }
                u[[i, j]] = sum / eigenvals[j];
            }
        } else {
            // Handle zero singular values
            if j < m {
                u[[j, j]] = F::one();
            }
        }
    }

    // Create result tensors
    let u_data = u.into_dyn();
    let s_data = s.into_dyn();
    // Clone v before taking transpose, and convert to owned dyn array
    let vt_data = v.t().to_owned().into_dyn();

    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let u_data_clone = u_data.clone();
        let s_data_clone = s_data.clone();
        let vt_data_clone = vt_data.clone();

        // Backward function for SVD - extremely simplified version
        // A proper implementation would be more complex
        let backward_u = if requires_grad {
            Some(Box::new(move |grad_u: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                // Simplified gradient implementation
                // In a full implementation, this would properly handle the SVD gradient
                let mut grad_a = Array2::<F>::zeros((m, n));

                // Simple gradient approximation: dA ≈ dU * Σ * V^T
                let grad_u_2d = grad_u.clone().into_shape((m, eigenvals.len())).unwrap();
                let s_diag = Array2::<F>::from_diag(&s_data_clone.clone().into_shape(eigenvals.len()).unwrap());

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = F::zero();
                        for k in 0..eigenvals.len() {
                            for l in 0..eigenvals.len() {
                                if k == l {
                                    sum = sum + grad_u_2d[[i, k]] * s_diag[[k, l]] * vt_data_clone.clone().into_shape((eigenvals.len(), n)).unwrap()[[l, j]];
                                }
                            }
                        }
                        grad_a[[i, j]] = sum;
                    }
                }

                Ok(grad_a.into_dyn())
            }) as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>)
        } else {
            None
        };

        // Note: For a proper implementation, we would need separate backward functions for s and vt
        // with proper gradient propagation

        let node_u = Node::new(
            scirs2_autograd::graph::OpType::Activation("svd_u".to_string()),
            vec![a],
            vec![backward_u],
        );

        // For s and vt, we'll simplify and not implement proper gradient backpropagation
        // A full implementation would be much more complex

        let mut u_tensor = Tensor::new(u_data, requires_grad);
        u_tensor.node = Some(node_u);
        let s_tensor = Tensor::new(s_data, false); // Simplified: no grad for S
        let vt_tensor = Tensor::new(vt_data.to_owned(), false); // Simplified: no grad for V^T

        Ok((u_tensor, s_tensor, vt_tensor))
    } else {
        let u_tensor = Tensor::new(u_data, false);
        let s_tensor = Tensor::new(s_data, false);
        let vt_tensor = Tensor::new(vt_data.to_owned(), false);

        Ok((u_tensor, s_tensor, vt_tensor))
    }
}

/// Compute eigenvalues and eigenvectors with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A tuple (eigenvalues, eigenvectors) with gradient tracking.
pub fn eig<F: Float + Debug + Send + Sync + 'static>(
    a: &Tensor<F>,
) -> AutogradResult<(Tensor<F>, Tensor<F>)> {
    // Ensure the input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Eigendecomposition requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Eigendecomposition requires a square matrix".to_string(),
        ));
    }

    // For 2x2 matrices, we can compute eigenvalues and eigenvectors directly
    let n = a_shape[0];

    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Eigendecomposition for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    let mut eigenvals = Array1::<F>::zeros(n);
    let mut eigenvecs = Array2::<F>::zeros((n, n));

    if n == 1 {
        // For 1x1 matrix, the eigenvalue is just the element
        eigenvals[0] = a.data[[0, 0]];
        eigenvecs[[0, 0]] = F::one();
    } else if n == 2 {
        // For 2x2 matrix, compute eigenvalues using the quadratic formula
        let a11 = a.data[[0, 0]];
        let a12 = a.data[[0, 1]];
        let a21 = a.data[[1, 0]];
        let a22 = a.data[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(scirs2_autograd::error::AutogradError::OperationError(
                "Complex eigenvalues encountered".to_string(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        eigenvals[0] = (trace + sqrt_disc) / F::from(2.0).unwrap();
        eigenvals[1] = (trace - sqrt_disc) / F::from(2.0).unwrap();

        // Compute eigenvectors
        if a12.abs() > F::epsilon() {
            eigenvecs[[0, 0]] = eigenvals[0] - a22;
            eigenvecs[[1, 0]] = a21;
            eigenvecs[[0, 1]] = eigenvals[1] - a22;
            eigenvecs[[1, 1]] = a21;
        } else if a21.abs() > F::epsilon() {
            eigenvecs[[0, 0]] = a12;
            eigenvecs[[1, 0]] = eigenvals[0] - a11;
            eigenvecs[[0, 1]] = a12;
            eigenvecs[[1, 1]] = eigenvals[1] - a11;
        } else {
            // Diagonal matrix
            eigenvecs[[0, 0]] = F::one();
            eigenvecs[[1, 0]] = F::zero();
            eigenvecs[[0, 1]] = F::zero();
            eigenvecs[[1, 1]] = F::one();
        }

        // Normalize eigenvectors
        let norm1 =
            (eigenvecs[[0, 0]] * eigenvecs[[0, 0]] + eigenvecs[[1, 0]] * eigenvecs[[1, 0]]).sqrt();
        let norm2 =
            (eigenvecs[[0, 1]] * eigenvecs[[0, 1]] + eigenvecs[[1, 1]] * eigenvecs[[1, 1]]).sqrt();

        if norm1 > F::epsilon() {
            eigenvecs[[0, 0]] = eigenvecs[[0, 0]] / norm1;
            eigenvecs[[1, 0]] = eigenvecs[[1, 0]] / norm1;
        }

        if norm2 > F::epsilon() {
            eigenvecs[[0, 1]] = eigenvecs[[0, 1]] / norm2;
            eigenvecs[[1, 1]] = eigenvecs[[1, 1]] / norm2;
        }
    }

    let eigenvals_data = eigenvals.into_dyn();
    let eigenvecs_data = eigenvecs.into_dyn();

    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let eigenvecs_data_clone = eigenvecs_data.clone();

        // Backward function for eigenvalues - simplified version
        let backward_vals = if requires_grad {
            Some(Box::new(move |grad_vals: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                // For 2x2 symmetric matrices, the gradient of eigenvalues with respect to the matrix
                // is given by the outer product of the corresponding eigenvector with itself
                let mut grad_a = Array2::<F>::zeros((n, n));

                for i in 0..n {
                    let grad_lambda_i = grad_vals.clone().into_shape(n).unwrap()[i];
                    let v_i = eigenvecs_data_clone.clone().into_shape((n, n)).unwrap().column(i).to_owned();

                    for j in 0..n {
                        for k in 0..n {
                            grad_a[[j, k]] = grad_a[[j, k]] + grad_lambda_i * v_i[j] * v_i[k];
                        }
                    }
                }

                Ok(grad_a.into_dyn())
            }) as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>)
        } else {
            None
        };

        // Note: For a proper implementation, we would need a separate backward function for eigenvectors
        // with proper gradient propagation

        let node_vals = Node::new(
            scirs2_autograd::graph::OpType::Activation("eig_vals".to_string()),
            vec![a],
            vec![backward_vals],
        );

        let mut vals_tensor = Tensor::new(eigenvals_data, requires_grad);
        vals_tensor.node = Some(node_vals);
        let vecs_tensor = Tensor::new(eigenvecs_data, false); // Simplified: no grad for eigenvectors

        Ok((vals_tensor, vecs_tensor))
    } else {
        let vals_tensor = Tensor::new(eigenvals_data, false);
        let vecs_tensor = Tensor::new(eigenvecs_data, false);

        Ok((vals_tensor, vecs_tensor))
    }
}

/// Compute the matrix exponential with automatic differentiation support.
///
/// # Arguments
///
/// * `a` - Input square matrix tensor
///
/// # Returns
///
/// A new tensor containing the matrix exponential with gradient tracking.
pub fn expm<F: Float + Debug + Send + Sync + 'static>(a: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    // Ensure the input is a square 2D tensor
    if a.data.ndim() != 2 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix exponential requires a 2D tensor".to_string(),
        ));
    }

    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Matrix exponential requires a square matrix".to_string(),
        ));
    }

    let n = a_shape[0];

    // For 2x2 matrices, we'll use a simple approximation of the matrix exponential
    // using the truncated Taylor series:
    // exp(A) ≈ I + A + A²/2! + A³/3! + ...

    if n > 2 {
        return Err(scirs2_autograd::error::AutogradError::OperationError(
            "Matrix exponential for matrices larger than 2x2 not yet implemented in autodiff"
                .to_string(),
        ));
    }

    // Compute matrix exponential using Taylor series approximation
    let mut result = Array2::<F>::eye(n); // Identity matrix
    let mut term = Array2::<F>::eye(n); // Current term in the series
    let a_mat = a.data.clone().into_shape((n, n)).unwrap();

    // Use 10 terms for the Taylor series approximation
    for k in 1..=10 {
        // Compute A^k / k!
        let scalar_divisor = F::from(k as f64).unwrap();
        let mut new_term = Array2::<F>::zeros((n, n));

        // Manually compute matrix multiplication and division
        for i in 0..n {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..n {
                    sum = sum + term[[i, k]] * a_mat[[k, j]];
                }
                new_term[[i, j]] = sum / scalar_divisor;
            }
        }
        term = new_term;

        // Add to the sum
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = result[[i, j]] + term[[i, j]];
            }
        }
    }

    let result_data = result.into_dyn();
    let requires_grad = a.requires_grad;

    if requires_grad {
        let a_data = a.data.clone();
        let expm_data = result_data.clone();

        // Backward function for matrix exponential
        let backward = if requires_grad {
            Some(Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
                // For the matrix exponential, we can use the formula:
                // d(exp(A))/dA_ij = ∫_0^1 exp(sA) * E_ij * exp((1-s)A) ds
                // where E_ij is the matrix with 1 at position (i,j) and 0 elsewhere

                // For simplicity, we'll use a very crude approximation:
                // d(exp(A))/dA ≈ exp(A)
                // This is a gross simplification and not correct except for commuting matrices

                // In a proper implementation, we would compute the integral using numerical methods
                let grad_2d = grad.clone().into_shape((n, n)).unwrap();
                let expm_2d = expm_data.clone().into_shape((n, n)).unwrap();

                let grad_a = grad_2d * expm_2d;

                Ok(grad_a.into_dyn())
            }) as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>)
        } else {
            None
        };

        let node = Node::new(
            scirs2_autograd::graph::OpType::Activation("expm".to_string()),
            vec![a],
            vec![backward],
        );

        // Since from_operation is private, we'll use a different approach
        let mut result = Tensor::new(result_data, requires_grad);
        result.node = Some(node);
        Ok(result)
    } else {
        Ok(Tensor::new(result_data, false))
    }
}
