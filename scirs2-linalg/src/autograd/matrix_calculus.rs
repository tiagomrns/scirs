//! Matrix calculus operations with automatic differentiation support
//!
//! This module provides differentiable matrix calculus operations like
//! gradients, Jacobians, and Hessians that integrate with the autograd system.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_autograd::error::Result as AutogradResult;
use scirs2_autograd::graph::Node;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::variable::Variable;

use super::matmul;

/// Compute the gradient of a scalar-valued function with respect to its inputs.
///
/// # Arguments
///
/// * `f` - Function that takes a tensor and returns a scalar tensor
/// * `x` - Input tensor at which to evaluate the gradient
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// A tensor containing the gradient vector.
pub fn gradient<F: Float + Debug + Send + Sync + 'static>(
    f: impl Fn(&Tensor<F>) -> AutogradResult<Tensor<F>>,
    x: &Tensor<F>,
    epsilon: Option<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure x is a vector (1D tensor)
    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Input must be a 1D tensor for gradient computation".to_string(),
        ));
    }

    let n = x.shape()[0];
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());

    // Initialize gradient vector
    let mut grad = Array1::<F>::zeros(n);

    // Compute the gradient using finite differences
    // Note: In a full autograd implementation, this would use reverse-mode autodiff
    for i in 0..n {
        // Create perturbed input x + eps*e_i
        let mut x_plus = x.data.clone();
        x_plus[i] = x_plus[i] + eps;
        let x_plus_tensor = Tensor::new(x_plus, false);

        // Create perturbed input x - eps*e_i
        let mut x_minus = x.data.clone();
        x_minus[i] = x_minus[i] - eps;
        let x_minus_tensor = Tensor::new(x_minus, false);

        // Evaluate function at perturbed points
        let f_plus = f(&x_plus_tensor)?;
        let f_minus = f(&x_minus_tensor)?;

        // Compute central difference approximation
        grad[i] = (f_plus.data[[0]] - f_minus.data[[0]]) / (F::from(2.0).unwrap() * eps);
    }

    // Create result tensor
    let grad_data = grad.into_dyn();

    // For proper autodiff, we would create nodes in the computation graph
    // Here we're simply returning the result without gradient tracking
    Ok(Tensor::new(grad_data, false))
}

/// Compute the Jacobian matrix of a vector-valued function.
///
/// # Arguments
///
/// * `f` - Function that takes a tensor and returns a vector tensor
/// * `x` - Input tensor at which to evaluate the Jacobian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// A tensor containing the Jacobian matrix.
pub fn jacobian<F: Float + Debug + Send + Sync + 'static>(
    f: impl Fn(&Tensor<F>) -> AutogradResult<Tensor<F>>,
    x: &Tensor<F>,
    epsilon: Option<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure x is a vector (1D tensor)
    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Input must be a 1D tensor for Jacobian computation".to_string(),
        ));
    }

    let n = x.shape()[0];
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());

    // Evaluate function at the input point
    let f_x = f(x)?;

    // Ensure output is a vector
    if f_x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Function must return a 1D tensor for Jacobian computation".to_string(),
        ));
    }

    let m = f_x.shape()[0];

    // Initialize Jacobian matrix
    let mut jac = Array2::<F>::zeros((m, n));

    // Compute the Jacobian using finite differences
    for j in 0..n {
        // Create perturbed input x + eps*e_j
        let mut x_plus = x.data.clone();
        x_plus[j] = x_plus[j] + eps;
        let x_plus_tensor = Tensor::new(x_plus, false);

        // Evaluate function at perturbed point
        let f_plus = f(&x_plus_tensor)?;

        // Compute forward difference approximation for column j
        for i in 0..m {
            jac[[i, j]] = (f_plus.data[i] - f_x.data[i]) / eps;
        }
    }

    // Create result tensor
    let jac_data = jac.into_dyn();

    // For proper autodiff, we would create nodes in the computation graph
    // Here we're simply returning the result without gradient tracking
    Ok(Tensor::new(jac_data, false))
}

/// Compute the Hessian matrix of a scalar-valued function.
///
/// # Arguments
///
/// * `f` - Function that takes a tensor and returns a scalar tensor
/// * `x` - Input tensor at which to evaluate the Hessian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// A tensor containing the Hessian matrix.
pub fn hessian<F: Float + Debug + Send + Sync + 'static>(
    f: impl Fn(&Tensor<F>) -> AutogradResult<Tensor<F>> + Copy,
    x: &Tensor<F>,
    epsilon: Option<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure x is a vector (1D tensor)
    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Input must be a 1D tensor for Hessian computation".to_string(),
        ));
    }

    let n = x.shape()[0];
    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt().sqrt());

    // Initialize Hessian matrix
    let mut hess = Array2::<F>::zeros((n, n));

    // Compute the Hessian using finite differences on the gradient
    let grad_f = |y: &Tensor<F>| -> AutogradResult<Tensor<F>> { gradient(f, y, Some(eps)) };

    for j in 0..n {
        // Create perturbed input x + eps*e_j
        let mut x_plus = x.data.clone();
        x_plus[j] = x_plus[j] + eps;
        let x_plus_tensor = Tensor::new(x_plus, false);

        // Compute gradient at perturbed point
        let grad_plus = grad_f(&x_plus_tensor)?;

        // Compute gradient at original point
        let grad_x = grad_f(x)?;

        // Compute forward difference approximation for column j
        for i in 0..n {
            hess[[i, j]] = (grad_plus.data[i] - grad_x.data[i]) / eps;
        }
    }

    // Create result tensor
    let hess_data = hess.into_dyn();

    // For proper autodiff, we would create nodes in the computation graph
    // Here we're simply returning the result without gradient tracking
    Ok(Tensor::new(hess_data, false))
}

/// Compute the vector-Jacobian product (VJP) efficiently.
///
/// # Arguments
///
/// * `f` - Function that takes a tensor and returns a tensor
/// * `x` - Input tensor at which to evaluate the VJP
/// * `v` - Vector to multiply with the Jacobian
///
/// # Returns
///
/// The vector-Jacobian product.
pub fn vector_jacobian_product<F: Float + Debug + Send + Sync + 'static>(
    f: impl Fn(&Tensor<F>) -> AutogradResult<Tensor<F>>,
    x: &Tensor<F>,
    v: &Tensor<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure x is a vector (1D tensor)
    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Input must be a 1D tensor for VJP computation".to_string(),
        ));
    }

    // Ensure v is a vector with compatible shape
    if v.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Vector v must be a 1D tensor for VJP computation".to_string(),
        ));
    }

    // Evaluate function at x
    let y = f(x)?;

    if y.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Function must return a 1D tensor for VJP computation".to_string(),
        ));
    }

    if v.shape()[0] != y.shape()[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Vector v shape {} incompatible with function output shape {}",
                v.shape()[0],
                y.shape()[0]
            ),
        ));
    }

    // Create a new function that computes the dot product of f(x) and v
    let scalar_f = |z: &Tensor<F>| -> AutogradResult<Tensor<F>> {
        let f_z = f(z)?;
        let mut dot_product = F::zero();
        for i in 0..f_z.shape()[0] {
            dot_product = dot_product + f_z.data[i] * v.data[i];
        }
        Ok(Tensor::new(
            ndarray::Array::from_elem(ndarray::IxDyn(&[1]), dot_product),
            false,
        ))
    };

    // The gradient of this scalar function is exactly the VJP
    gradient(scalar_f, x, None)
}

/// Compute the Jacobian-vector product (JVP) efficiently.
///
/// # Arguments
///
/// * `f` - Function that takes a tensor and returns a tensor
/// * `x` - Input tensor at which to evaluate the JVP
/// * `v` - Vector to multiply with the Jacobian
///
/// # Returns
///
/// The Jacobian-vector product.
pub fn jacobian_vector_product<F: Float + Debug + Send + Sync + 'static>(
    f: impl Fn(&Tensor<F>) -> AutogradResult<Tensor<F>>,
    x: &Tensor<F>,
    v: &Tensor<F>,
    epsilon: Option<F>,
) -> AutogradResult<Tensor<F>> {
    // Ensure x is a vector (1D tensor)
    if x.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Input must be a 1D tensor for JVP computation".to_string(),
        ));
    }

    // Ensure v is a vector with compatible shape
    if v.data.ndim() != 1 {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            "Vector v must be a 1D tensor for JVP computation".to_string(),
        ));
    }

    if v.shape()[0] != x.shape()[0] {
        return Err(scirs2_autograd::error::AutogradError::ShapeMismatch(
            format!(
                "Vector v shape {} incompatible with input shape {}",
                v.shape()[0],
                x.shape()[0]
            ),
        ));
    }

    let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());

    // Compute f(x + eps*v) and f(x)
    let mut x_plus_eps_v = x.data.clone();
    for i in 0..x.shape()[0] {
        x_plus_eps_v[i] = x_plus_eps_v[i] + eps * v.data[i];
    }

    let x_plus_eps_v_tensor = Tensor::new(x_plus_eps_v, false);
    let f_x_plus_eps_v = f(&x_plus_eps_v_tensor)?;
    let f_x = f(x)?;

    // Compute the directional derivative (f(x + eps*v) - f(x)) / eps
    let mut result = f_x_plus_eps_v.data.clone();
    for i in 0..result.len() {
        result[i] = (result[i] - f_x.data[i]) / eps;
    }

    Ok(Tensor::new(result, false))
}

/// High-level interface for matrix calculus operations with autodiff support
pub mod variable {
    use super::*;
    use scirs2_autograd::variable::Variable;

    /// Compute the gradient of a scalar-valued function with respect to its inputs.
    pub fn gradient<F: Float + Debug + Send + Sync + 'static>(
        f: impl Fn(&Variable<F>) -> AutogradResult<Variable<F>>,
        x: &Variable<F>,
        epsilon: Option<F>,
    ) -> AutogradResult<Variable<F>> {
        // Convert the Variable-based function to a Tensor-based function
        let tensor_f = |x_tensor: &Tensor<F>| -> AutogradResult<Tensor<F>> {
            let x_var = Variable {
                tensor: x_tensor.clone(),
            };
            let result_var = f(&x_var)?;
            Ok(result_var.tensor)
        };

        let result_tensor = super::gradient(tensor_f, &x.tensor, epsilon)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Compute the Jacobian matrix of a vector-valued function.
    pub fn jacobian<F: Float + Debug + Send + Sync + 'static>(
        f: impl Fn(&Variable<F>) -> AutogradResult<Variable<F>>,
        x: &Variable<F>,
        epsilon: Option<F>,
    ) -> AutogradResult<Variable<F>> {
        // Convert the Variable-based function to a Tensor-based function
        let tensor_f = |x_tensor: &Tensor<F>| -> AutogradResult<Tensor<F>> {
            let x_var = Variable {
                tensor: x_tensor.clone(),
            };
            let result_var = f(&x_var)?;
            Ok(result_var.tensor)
        };

        let result_tensor = super::jacobian(tensor_f, &x.tensor, epsilon)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Compute the Hessian matrix of a scalar-valued function.
    pub fn hessian<F: Float + Debug + Send + Sync + 'static>(
        f: impl Fn(&Variable<F>) -> AutogradResult<Variable<F>> + Copy,
        x: &Variable<F>,
        epsilon: Option<F>,
    ) -> AutogradResult<Variable<F>> {
        // Convert the Variable-based function to a Tensor-based function
        let tensor_f = |x_tensor: &Tensor<F>| -> AutogradResult<Tensor<F>> {
            let x_var = Variable {
                tensor: x_tensor.clone(),
            };
            let result_var = f(&x_var)?;
            Ok(result_var.tensor)
        };

        let result_tensor = super::hessian(tensor_f, &x.tensor, epsilon)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Compute the vector-Jacobian product (VJP) efficiently.
    pub fn vector_jacobian_product<F: Float + Debug + Send + Sync + 'static>(
        f: impl Fn(&Variable<F>) -> AutogradResult<Variable<F>>,
        x: &Variable<F>,
        v: &Variable<F>,
    ) -> AutogradResult<Variable<F>> {
        // Convert the Variable-based function to a Tensor-based function
        let tensor_f = |x_tensor: &Tensor<F>| -> AutogradResult<Tensor<F>> {
            let x_var = Variable {
                tensor: x_tensor.clone(),
            };
            let result_var = f(&x_var)?;
            Ok(result_var.tensor)
        };

        let result_tensor = super::vector_jacobian_product(tensor_f, &x.tensor, &v.tensor)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }

    /// Compute the Jacobian-vector product (JVP) efficiently.
    pub fn jacobian_vector_product<F: Float + Debug + Send + Sync + 'static>(
        f: impl Fn(&Variable<F>) -> AutogradResult<Variable<F>>,
        x: &Variable<F>,
        v: &Variable<F>,
        epsilon: Option<F>,
    ) -> AutogradResult<Variable<F>> {
        // Convert the Variable-based function to a Tensor-based function
        let tensor_f = |x_tensor: &Tensor<F>| -> AutogradResult<Tensor<F>> {
            let x_var = Variable {
                tensor: x_tensor.clone(),
            };
            let result_var = f(&x_var)?;
            Ok(result_var.tensor)
        };

        let result_tensor =
            super::jacobian_vector_product(tensor_f, &x.tensor, &v.tensor, epsilon)?;
        Ok(Variable {
            tensor: result_tensor,
        })
    }
}
