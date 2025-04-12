//! Filtering functions for n-dimensional arrays
//!
//! This module provides functions for filtering n-dimensional arrays,
//! including Gaussian filters, median filters, and various convolution operations.

use std::fmt::Debug;

mod convolve;
mod edge;
mod gaussian;
mod median;
mod rank;
mod utils;

pub use convolve::*;
pub use edge::*;
pub use gaussian::{gaussian_filter, gaussian_filter_f32, gaussian_filter_f64};
pub use median::*;
pub use rank::{maximum_filter, minimum_filter, percentile_filter, rank_filter};
pub use utils::*;

/// Border handling modes for filters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Pad with zeros
    Constant,

    /// Reflect values across the edges
    Reflect,

    /// Mirror and reflect values across the edges
    Mirror,

    /// Wrap around to the opposite edge (periodic)
    Wrap,

    /// Repeat edge values
    Nearest,
}
