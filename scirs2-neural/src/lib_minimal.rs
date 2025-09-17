//! Neural network building blocks module for SciRS2 - Minimal working version
//!
//! This is a minimal version that includes only the core working modules
//! to establish a baseline compilation before adding more complex features.

#![warn(missing_docs)]
// Core working modules
pub mod error;
pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;
// Re-export the error type
pub use error::{Error, NeuralError, Result};
// Optional prelude for convenience
pub mod prelude {
    //! Convenient re-exports for common neural network operations
    
    pub use crate::error::{Error, NeuralError, Result};
    pub use crate::activations::{Activation, ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish};
    pub use crate::layers::{Layer, Dense, Sequential};
    pub use crate::losses::{Loss, MeanSquaredError};
    pub use crate::optimizers::Optimizer;
}
