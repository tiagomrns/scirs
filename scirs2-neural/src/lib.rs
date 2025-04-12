//! Neural network building blocks module for SciRS2
//!
//! This module provides neural network building blocks for SciRS2, including:
//! - Layers (dense, convolutional, recurrent, etc.)
//! - Activation functions (ReLU, sigmoid, tanh, etc.)
//! - Loss functions (MSE, cross-entropy, etc.)
//! - Optimizers (SGD, Adam, etc.)
//! - Model architectures and training utilities
//! - Neural network specific linear algebra operations

#![warn(missing_docs)]
#![recursion_limit = "524288"]

pub mod activations;
pub mod autograd;
pub mod error;
pub mod layers;
pub mod linalg;
pub mod losses;
pub mod models;
pub mod optimizers;
pub mod utils;

pub use activations::*;
pub use layers::*;
pub use losses::*;
pub use models::*;
pub use optimizers::*;
