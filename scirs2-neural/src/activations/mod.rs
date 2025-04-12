//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural networks.

use crate::error::Result;
use ndarray::Array;
use num_traits::Float;

/// Trait for activation functions
pub trait Activation<F: Float> {
    /// Apply the activation function to the input
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Compute the derivative of the activation function with respect to the input
    ///
    /// Arguments:
    /// * `grad_output` - Gradient from the next layer
    /// * `output` - Output of the forward pass
    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;
}

mod relu;
mod sigmoid;
mod softmax;
mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;
