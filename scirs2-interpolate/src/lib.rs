//! Interpolation module
//!
//! This module provides implementations of various interpolation methods.
//! These methods are used to estimate values at arbitrary points based on a
//! set of known data points.
//!
//! ## Overview
//!
//! * 1D interpolation methods (`interp1d` module)
//!   * Linear, nearest, cubic interpolation
//!   * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) - shape-preserving interpolation
//! * Spline interpolation (`spline` module)
//! * Bivariate splines (`bivariate` module):
//!   * `BivariateSpline` - Base class for bivariate splines
//!   * `SmoothBivariateSpline` - Smooth bivariate spline approximation
//!   * `RectBivariateSpline` - Bivariate spline approximation over a rectangular mesh
//! * Multivariate interpolation (`interpnd` module)
//! * Advanced interpolation methods (`advanced` module):
//!   * Akima spline interpolation - robust to outliers
//!   * Radial Basis Function (RBF) interpolation - for scattered data
//!   * Kriging (Gaussian process regression) - with uncertainty quantification
//!   * Barycentric interpolation - stable polynomial interpolation
//! * Grid transformation and resampling (`grid` module):
//!   * Resample scattered data onto regular grids
//!   * Convert between grids of different resolutions
//!   * Map grid data to arbitrary points
//! * Tensor product interpolation (`tensor` module):
//!   * Efficient high-dimensional interpolation on structured grids
//!   * Higher-order interpolation using Lagrange polynomials
//! * Utility functions (`utils` module):
//!   * Error estimation with cross-validation
//!   * Parameter optimization
//!   * Differentiation and integration of interpolated functions

// Export error types
pub mod error;
pub use error::{InterpolateError, InterpolateResult};

// Interpolation modules
pub mod advanced;
pub mod bivariate;
pub mod grid;
pub mod interp1d;
pub mod interpnd;
pub mod spline;
pub mod tensor;
pub mod utils;

// Re-exports for convenience
pub use advanced::akima::{make_akima_spline, AkimaSpline};
pub use advanced::barycentric::{
    make_barycentric_interpolator, BarycentricInterpolator, BarycentricTriangulation,
};
pub use advanced::kriging::{make_kriging_interpolator, CovarianceFunction, KrigingInterpolator};
pub use advanced::rbf::{RBFInterpolator, RBFKernel};
pub use bivariate::{
    BivariateInterpolator, BivariateSpline, RectBivariateSpline, SmoothBivariateSpline,
    SmoothBivariateSplineBuilder,
};
pub use grid::{
    create_regular_grid, map_grid_to_points, resample_grid_to_grid, resample_to_grid,
    GridTransformMethod,
};
pub use interp1d::{
    cubic_interpolate, linear_interpolate, nearest_interpolate, pchip_interpolate, Interp1d,
    InterpolationMethod, PchipInterpolator,
};
pub use interpnd::{
    make_interp_nd, make_interp_scattered, map_coordinates, ExtrapolateMode, GridType,
    RegularGridInterpolator, ScatteredInterpolator,
};
pub use spline::{make_interp_spline, CubicSpline};
pub use tensor::{
    lagrange_tensor_interpolate, tensor_product_interpolate, LagrangeTensorInterpolator,
    TensorProductInterpolator,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
