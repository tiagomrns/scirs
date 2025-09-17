//! Neural network building blocks module for SciRS2 - Advanced minimal working version
//!
//! This is an advanced minimal version that includes only the error module
//! to establish basic compilation.

#![warn(missing_docs)]
// Advanced minimal - just error handling
pub mod error;
// Re-export the error type
pub use error::{Error, NeuralError, Result};
