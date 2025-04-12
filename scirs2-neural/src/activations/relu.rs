//! ReLU activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use ndarray::{Array, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Rectified Linear Unit (ReLU) activation function.
///
/// The ReLU function returns the input if it's positive, and 0 otherwise:
/// f(x) = max(0, x)
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::ReLU;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let relu = ReLU::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = relu.forward(&input).unwrap();
/// assert_eq!(output, Array::from_vec(vec![1.0, 0.0, 2.0, 0.0]).into_dyn());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReLU {
    /// Alpha parameter for leaky ReLU.
    /// If alpha = 0, it's a standard ReLU.
    /// If alpha > 0, it's a leaky ReLU.
    alpha: f64,
}

impl ReLU {
    /// Create a new ReLU activation function.
    pub fn new() -> Self {
        Self { alpha: 0.0 }
    }

    /// Create a new Leaky ReLU activation function with given alpha.
    pub fn leaky(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for ReLU {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();

        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            if *x < zero {
                *x = alpha * *x;
            }
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();
        let one = F::one();

        // Create derivative mask based on the output (1 where x > 0, alpha where x <= 0)
        let mut mask = Array::from_elem(output.dim(), one);
        Zip::from(&mut mask).and(output).for_each(|mask_val, &out| {
            if out <= zero {
                *mask_val = alpha;
            }
        });

        // Multiply element-wise with the gradient from the next layer
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(&mask)
            .for_each(|grad_in, &grad_out, &mask_val| {
                *grad_in = grad_out * mask_val;
            });

        Ok(grad_input)
    }
}
