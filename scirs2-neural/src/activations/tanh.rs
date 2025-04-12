//! Hyperbolic tangent (tanh) activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use ndarray::{Array, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Hyperbolic tangent (tanh) activation function.
///
/// The tanh function is defined as:
/// f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// It maps any input value to a value between -1 and 1.
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::Tanh;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let tanh = Tanh::new();
/// let input = Array::from_vec(vec![0.0f64, 1.0, -1.0]).into_dyn();
/// let output = tanh.forward(&input).unwrap();
///
/// // Check that values are in the expected range
/// assert!(output.iter().all(|&x| x >= -1.0f64 && x <= 1.0f64));
///
/// // tanh(0) should be 0
/// assert!(output[0].abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Tanh {
    /// Create a new Tanh activation function.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Tanh {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut output = input.clone();

        Zip::from(&mut output).for_each(|x| {
            *x = x.tanh();
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let one = F::one();
        let mut grad_input = Array::zeros(output.raw_dim());

        // For tanh: derivative = 1 - tanh^2(x)
        // output already contains tanh(x), so we compute 1 - output^2
        // grad_input = grad_output * (1 - output^2)
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(output)
            .for_each(|grad_in, &grad_out, &out| {
                *grad_in = grad_out * (one - out * out);
            });

        Ok(grad_input)
    }
}
