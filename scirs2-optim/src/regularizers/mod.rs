//! Regularization techniques for machine learning
//!
//! This module provides various regularization techniques commonly used in
//! machine learning to prevent overfitting, such as L1 (Lasso), L2 (Ridge),
//! ElasticNet, and Dropout.

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;

/// Trait for regularizers that can be applied to parameters and gradients
pub trait Regularizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Apply regularization to parameters and gradients
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to regularize
    /// * `gradients` - The gradients to modify
    ///
    /// # Returns
    ///
    /// The regularization penalty value
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A>;

    /// Compute the regularization penalty value
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to compute the penalty for
    ///
    /// # Returns
    ///
    /// The regularization penalty value
    fn penalty(&self, params: &Array<A, D>) -> Result<A>;
}

mod dropout;
mod elastic_net;
mod l1;
mod l2;

// Re-export regularizers
pub use dropout::Dropout;
pub use elastic_net::ElasticNet;
pub use l1::L1;
pub use l2::L2;
