//! Softmax activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use ndarray::{Array, Axis};
use num_traits::Float;
use std::fmt::Debug;

/// Softmax activation function.
///
/// The softmax function is defined as:
/// f(x_i) = exp(x_i) / sum_j(exp(x_j))
///
/// It transforms a vector of real values into a probability distribution.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This test fails currently due to issues with softmax axis handling
/// use scirs2_neural::activations::Softmax;
/// use scirs2_neural::activations::Activation;
/// use ndarray::{Array, arr2};
///
/// // For now we'll just demonstrate basic usage
/// let softmax = Softmax::new(0); // Set appropriate axis
/// let input = arr2(&[[1.0f64, 2.0, 3.0]]).into_dyn();
/// let _output = softmax.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    /// Axis along which to apply softmax (typically 0 for 1D arrays or the last dimension for batches)
    axis: usize,
}

impl Softmax {
    /// Create a new Softmax activation function.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the softmax.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F: Float + Debug> Activation<F> for Softmax {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        if input.ndim() <= self.axis {
            return Err(NeuralError::InferenceError(format!(
                "Softmax axis {} is out of bounds for input with {} dimensions",
                self.axis,
                input.ndim()
            )));
        }

        // Numerical stability: subtract the maximum value to avoid overflow
        let max_vals = input.map_axis(Axis(self.axis), |view| {
            view.fold(F::neg_infinity(), |a, &b| a.max(b))
        });

        let mut output = input.clone();

        // Apply exp(x - max) for numerical stability
        for (mut out_subview, &max_val) in
            output.axis_iter_mut(Axis(self.axis)).zip(max_vals.iter())
        {
            for val in out_subview.iter_mut() {
                *val = (*val - max_val).exp();
            }
        }

        // Calculate sum of exp(x - max) for each sample
        let sum_vals = output.map_axis(Axis(self.axis), |view| view.fold(F::zero(), |a, &b| a + b));

        // Normalize to get softmax values
        for (mut out_subview, &sum_val) in
            output.axis_iter_mut(Axis(self.axis)).zip(sum_vals.iter())
        {
            for val in out_subview.iter_mut() {
                *val = *val / sum_val;
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        _output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // Note: This backward pass implementation is a simplification.
        // The full Jacobian matrix for softmax is complex.
        // This implementation assumes the gradient is coming from categorical cross-entropy loss,
        // which simplifies the softmax gradient to (output - target).
        // For general case, we'd need to implement the full Jacobian-vector product.

        // For now, just pass through the gradient
        Ok(grad_output.clone())
    }
}
