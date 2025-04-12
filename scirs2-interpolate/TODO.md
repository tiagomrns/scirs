# scirs2-interpolate TODO

This module provides interpolation functionality similar to SciPy's interpolate module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] 1D interpolation methods
  - [x] Linear interpolation
  - [x] Nearest neighbor interpolation
  - [x] Cubic interpolation
  - [x] Spline interpolation
- [x] Multi-dimensional interpolation
  - [x] Regular grid interpolation (tensor product)
  - [x] Scattered data interpolation
- [x] Advanced interpolation methods
  - [x] Akima splines
  - [x] Barycentric interpolation
  - [x] Radial basis functions (RBF)
  - [x] Kriging (Gaussian process regression)
- [x] Utility functions
  - [x] Grid creation and manipulation
  - [x] Derivative and integration of interpolants
- [x] Fixed Clippy warnings for iterator_cloned_collect

## Future Tasks

- [ ] Add more interpolation methods
  - [ ] PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
  - [ ] Bivariate splines for irregularly spaced data
  - [ ] Thin-plate splines
  - [ ] Bezier curves and surfaces
- [ ] Improve performance for large datasets
  - [ ] Optimized data structures for nearest neighbor search
  - [ ] Parallelization of computationally intensive operations
- [ ] Enhance multi-dimensional interpolation
  - [ ] Better support for high-dimensional data
  - [ ] More efficient scattered data interpolation
- [ ] Add extrapolation methods and boundary handling
- [ ] Fix remaining ignored tests
  - [ ] Update barycentric_interpolator_quadratic test
  - [ ] Fix make_barycentric_interpolator test
  - [ ] Fix kriging_interpolator_prediction test
  - [ ] Address rbf_interpolator_2d test
- [ ] Add more examples and documentation
  - [ ] Tutorial for common interpolation tasks
  - [ ] Visual examples for different methods

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's interpolate
- [ ] Integration with optimization for parameter fitting
- [ ] Support for specialized domain-specific interpolation (geospatial, etc.)
- [ ] GPU-accelerated implementations for large datasets
- [ ] Adaptive methods for complex functions
- [ ] Integration with differentiation and integration modules