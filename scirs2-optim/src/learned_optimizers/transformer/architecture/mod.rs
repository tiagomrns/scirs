//! Transformer architecture components
//!
//! This module contains the core architectural components of the transformer optimizer,
//! organized into separate submodules for better maintainability and testing.

pub mod attention;
pub mod encoder;
pub mod feedforward;
pub mod positional_encoding;

// Re-export key types for convenience
pub use attention::{AttentionOptimization, MultiHeadAttention, RelativePositionBias, RoPEEmbeddings};
pub use encoder::{ActivationFunction, TransformerLayer, FeedForwardNetwork, LayerNorm, DropoutLayer};
pub use feedforward::{OutputProjectionLayer, InputEmbedding, OutputTransformation};
pub use positional_encoding::{PositionalEncoder, PositionalEncodingType};