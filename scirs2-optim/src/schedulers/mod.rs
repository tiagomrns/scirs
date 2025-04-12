//! Learning rate schedulers for optimizers
//!
//! This module provides various learning rate schedulers that adjust the learning rate
//! of optimizers during training based on different strategies.

use ndarray::{Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::optimizers::Optimizer;

/// Trait for learning rate schedulers
pub trait LearningRateScheduler<A: Float + Debug + ScalarOperand> {
    /// Get the learning rate at the current step
    fn get_learning_rate(&self) -> A;

    /// Update the scheduler state and return the new learning rate
    fn step(&mut self) -> A;

    /// Apply the scheduler to an optimizer
    fn apply_to<D: Dimension, O: Optimizer<A, D>>(&self, optimizer: &mut O)
    where
        Self: Sized,
    {
        optimizer.set_learning_rate(self.get_learning_rate());
    }

    /// Reset the scheduler state
    fn reset(&mut self);
}

mod cosine_annealing;
mod exponential_decay;
mod linear_decay;
mod reduce_on_plateau;
mod step_decay;

// Re-export schedulers
pub use cosine_annealing::CosineAnnealing;
pub use exponential_decay::ExponentialDecay;
pub use linear_decay::LinearDecay;
pub use reduce_on_plateau::ReduceOnPlateau;
pub use step_decay::StepDecay;
