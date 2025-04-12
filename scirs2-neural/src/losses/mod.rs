//! Loss functions for neural networks
//!
//! This module provides common loss functions used in neural networks.

use crate::error::Result;
use ndarray::Array;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for loss functions used in neural networks
pub trait Loss<F: Float + Debug> {
    /// Calculate the loss between predictions and targets
    fn forward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<F>;

    /// Calculate the gradient of the loss with respect to the predictions
    fn backward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;
}

mod crossentropy;
mod mse;

pub use crossentropy::CrossEntropyLoss;
pub use mse::MeanSquaredError;
