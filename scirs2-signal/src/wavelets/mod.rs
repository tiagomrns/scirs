// Wavelet transforms
//
// This module provides functions for continuous and discrete wavelet transforms,
// useful for multi-resolution analysis of signals.

use crate::dwt::Wavelet;

// Import internal modules
#[allow(unused_imports)]
mod complex_wavelets;
mod cwt;
mod dual_tree_complex;
mod real_wavelets;
mod scalogram;
#[cfg(test)]
mod tests;
mod transform;
mod types;
mod utils;

// Re-export public components
pub use complex_wavelets::{complex_gaussian, complex_morlet, fbsp, morlet, paul, shannon};
pub use cwt::{convolve_complex_same_complex, convolve_complex_same_real};
pub use dual_tree_complex::{
    BoundaryMode, Dtcwt1dResult, Dtcwt2dResult, DtcwtConfig, DtcwtFilters, DtcwtProcessor,
    FilterSet,
};
pub use real_wavelets::ricker;
pub use scalogram::{cwt_magnitude, cwt_phase, scale_to_frequency, scalogram};
pub use transform::cwt;
pub use types::WaveletType;

// No common imports for internal use
