//! Neural network layers implementation
//!
//! This module provides implementations of various neural network layers
//! such as dense (fully connected), convolution, pooling, etc.

use crate::error::Result;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Base trait for neural network layers
pub trait Layer<F: Float + Debug + ScalarOperand> {
    /// Forward pass of the layer
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Backward pass of the layer to compute gradients
    fn backward(&self, grad_output: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Update the layer parameters with the given gradients
    fn update(&mut self, learning_rate: F) -> Result<()>;

    /// Get the layer as a dyn Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the layer as a mutable dyn Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Trait for layers with parameters (weights, biases)
pub trait ParamLayer<F: Float + Debug + ScalarOperand>: Layer<F> {
    /// Get the parameters of the layer as a vector of arrays
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>>;

    /// Get the gradients of the parameters
    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>>;

    /// Set the parameters of the layer
    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()>;
}

mod dense;

pub use dense::Dense;
