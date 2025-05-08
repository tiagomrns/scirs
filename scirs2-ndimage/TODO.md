# scirs2-ndimage TODO List

This module provides multidimensional image processing functionality similar to SciPy's ndimage module. It includes functions for filtering, interpolation, measurements, and morphological operations on n-dimensional arrays.

## Current Status

- [x] Set up module structure
- [x] Error handling implementation
- [x] Code organization into specialized submodules
- [x] API definition and interfaces for all major functionality
- [x] Basic unit tests framework established
- [x] Benchmarks for key operations (rank filters)

## Implemented Features

- [x] Filtering operations
  - [x] Rank filters (minimum, maximum, percentile)
  - [x] Gaussian filters API
  - [x] Median filters API
  - [x] Edge detection filters (Sobel, Laplace) API
  - [x] Convolution operations API

- [x] Feature detection
  - [x] Edge detection (Canny, Sobel)
  - [x] Corner detection (Harris, FAST)

- [x] Segmentation functionality
  - [x] Thresholding (binary, Otsu, adaptive)
  - [x] Watershed segmentation

- [x] Module structure and organization
  - [x] Reorganization into specialized submodules
  - [x] Clear API boundaries and exports

## In Progress

- [ ] Complete implementation of remaining filter operations
  - [ ] Full implementation of Gaussian filters
  - [ ] Full implementation of Median filters
  - [ ] Full implementation of Sobel filters

- [ ] Complete interpolation functionality
  - [ ] Affine transformations
  - [ ] Geometric transformations
  - [ ] Zoom and rotation
  - [ ] Spline interpolation

- [ ] Complete morphological operations
  - [ ] Erosion and dilation
  - [ ] Opening and closing
  - [ ] Morphological gradient
  - [ ] Top-hat and black-hat transforms

- [ ] Complete measurements and analysis
  - [ ] Center of mass
  - [ ] Extrema detection
  - [ ] Histograms
  - [ ] Statistical measures (sum, mean, variance)
  - [ ] Label and find objects

## Filter Operations Enhancement

- [ ] Comprehensive filter implementation
  - [ ] Uniform filter implementation
  - [ ] Minimum/maximum filters
  - [ ] Prewitt filter
  - [ ] Generic filter framework
  - [ ] Customizable filter footprints
- [ ] Boundary handling
  - [ ] Support all boundary modes (reflect, nearest, wrap, mirror, constant)
  - [ ] Optimized implementation for each boundary condition
- [ ] Vectorized filtering
  - [ ] Batch operations on multiple images
  - [ ] Parallelized implementation for multi-core systems
- [ ] Order-statistics-based filters
  - [ ] Rank filter with variable ranking
  - [ ] Percentile filter with optimizations
  - [ ] Median filter (optimized)

## Fourier Domain Processing

- [ ] Fourier-based operations
  - [ ] Fourier Gaussian filter
  - [ ] Fourier uniform filter
  - [ ] Fourier ellipsoid filter
  - [ ] Fourier shift operations
- [ ] Optimization for large arrays
  - [ ] Memory-efficient FFT-based filtering
  - [ ] Streaming operations for large data
- [ ] Integration with scirs2-fft
  - [ ] Leverage FFT implementations
  - [ ] Consistent API across modules

## Interpolation and Transformations

- [ ] Comprehensive interpolation
  - [ ] Map coordinates with various order splines
  - [ ] Affine transformation with matrix input
  - [ ] Zoom functionality with customizable spline order
  - [ ] Shift operation with sub-pixel precision
  - [ ] Rotation with customizable center point
- [ ] Performance optimizations
  - [ ] Pre-computed coefficient caching
  - [ ] SIMD-optimized interpolation kernels
  - [ ] Parallel implementation for large images
- [ ] Specialized transforms
  - [ ] Non-rigid transformations
  - [ ] Perspective transformations
  - [ ] Multi-resolution approaches

## Morphological Operations

- [ ] Binary morphology
  - [ ] Binary erosion/dilation with arbitrary structuring elements
  - [ ] Binary opening/closing
  - [ ] Binary propagation
  - [ ] Binary hit-or-miss transform
- [ ] Grayscale morphology
  - [ ] Grayscale erosion/dilation
  - [ ] Grayscale opening/closing
  - [ ] Top-hat and black-hat transforms
  - [ ] Morphological gradient, laplace
- [ ] Distance transforms
  - [ ] Euclidean distance transform
  - [ ] City-block distance
  - [ ] Chessboard distance
  - [ ] Fast algorithms for specific metrics

## Measurement and Analysis

- [ ] Region analysis
  - [ ] Connected component labeling
  - [ ] Object properties (area, perimeter)
  - [ ] Region-based statistics
  - [ ] Watershed segmentation enhancements
- [ ] Statistical measurements
  - [ ] Mean, variance, standard deviation by label
  - [ ] Histogram by label
  - [ ] Center of mass computation
  - [ ] Moment calculations
- [ ] Feature measurement
  - [ ] Find objects with size filtering
  - [ ] Extrema detection (maxima, minima)
  - [ ] Object orientation and principal axes

## Backend Support and Integration

- [ ] Alternative backend support
  - [ ] Delegation system for GPU acceleration
  - [ ] CuPy/CUDA backend integration
  - [ ] Unified API across backends
- [ ] Memory management
  - [ ] Views vs. copies control
  - [ ] In-place operation options
  - [ ] Memory footprint optimization
- [ ] Thread pool integration
  - [ ] Shared worker pool with other modules
  - [ ] Thread count control and optimization

## Documentation and Examples

- [ ] Documentation and examples
  - [ ] Document all public APIs with examples
  - [ ] Create tutorial notebooks for common tasks
  - [ ] Add visual examples for different methods
  - [ ] Create comprehensive user guide
  - [ ] Gallery of example applications

## Testing and Quality Assurance

- [ ] Expand test coverage
  - [ ] Unit tests for all functions
  - [ ] Edge case testing
  - [ ] Performance benchmarks for all operations

- [ ] Validation against SciPy's ndimage
  - [ ] Numerical comparison tests
  - [ ] Performance comparison benchmarks
  - [ ] API compatibility verification

## Next Steps (Immediate)

- [ ] Complete implementation of filter operations
- [ ] Fix generic parameter issues in feature detection modules
- [ ] Address type conversion issues between arrays and image data
- [ ] Implement comprehensive test suite for new functionality

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's ndimage
- [ ] Integration with other image processing libraries
- [ ] Support for large images and datasets
  - [ ] Memory-efficient implementations
  - [ ] Streaming processing for large datasets
- [ ] GPU-accelerated implementations for intensive operations
- [ ] Domain-specific functions for medical, satellite, and microscopy imaging
- [ ] Advanced visualization tools and examples