//! Morphological operations on n-dimensional arrays
//!
//! This module provides functions for performing morphological operations on n-dimensional
//! arrays, such as erosion, dilation, opening, closing, and more.

use std::fmt::Debug;

mod binary;
mod connected;
mod grayscale;
mod structuring;
mod utils;

pub use binary::*;
pub use connected::*;
pub use grayscale::*;
pub use structuring::*;
pub use utils::*;

/// Border handling modes for morphological operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphBorderMode {
    /// Pad with constant value (default is 0 for binary morphology)
    Constant,

    /// Reflect values at border
    Reflect,

    /// Mirror values at border (reflects and repeats edge values)
    Mirror,

    /// Wrap around to opposite edge (periodic)
    Wrap,

    /// Use nearest edge values
    Nearest,
}

/// Structuring element connectivity for binary morphology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Only face neighbors (4-connectivity in 2D)
    Face,

    /// Face and edge neighbors (8-connectivity in 2D)
    FaceEdge,

    /// Face, edge, and vertex neighbors (full connectivity)
    Full,
}
