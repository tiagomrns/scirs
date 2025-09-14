//! Minimal working scirs2-neural implementation

pub mod error;
pub mod activations_minimal;

pub use error::{Error, NeuralError, Result};
pub use activations_minimal::{Activation, GELU, Tanh};

/// Working prelude with minimal functionality
pub mod prelude {
    pub use crate::{
        activations_minimal::{Activation, GELU, Tanh},
        error::{Error, NeuralError, Result},
    };
}
