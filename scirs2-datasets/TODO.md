# scirs2-datasets TODO

This module provides dataset loading utilities similar to scikit-learn's datasets module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Initial implementation completed
- [x] Core functionality working
- [x] Examples added

## Implemented Features

- [x] Implement standard test datasets
  - [x] Toy datasets for testing algorithms
  - [x] Small sample datasets for examples
  - [x] Synthetic data generators
- [x] Implement dataset loading utilities
  - [x] Downloading and caching (feature-gated)
  - [x] Memory-efficient loading
  - [x] Data preprocessing
- [x] Add dataset manipulation tools
  - [x] Splitting (train/test)
- [x] Implement dataset conversion utilities
  - [x] Format conversion (CSV, JSON)
  - [x] Normalization
- [x] Add data generation utilities
  - [x] Random data generators
  - [x] Structured data generators (blobs, classification, regression)
  - [x] Time series generators
- [x] Implement metadata handling
  - [x] Dataset descriptions
  - [x] Feature information
  - [x] Target information
- [x] Add examples and documentation
  - [x] Examples for loading toy datasets
  - [x] Examples for data generation
  - [x] Examples for CSV loading

## Future Tasks

- [ ] Add real-world datasets
  - [x] Classic machine learning datasets (iris, breast cancer, digits, boston)
  - [ ] Domain-specific scientific datasets
  - [ ] Time series datasets
- [ ] Extend dataset manipulation tools
  - [ ] Cross-validation utilities
  - [ ] Sampling and bootstrapping
- [ ] Enhance dataset conversion utilities
  - [ ] Feature extraction
- [ ] Improve documentation
  - [ ] Comprehensive tutorial

## Code Cleanup Tasks

- [ ] Fix warnings about unused imports
- [ ] Add missing documentation for public items
- [ ] Improve error handling for edge cases
- [ ] Add more unit tests
- [ ] Optimize serialization for larger datasets

## Long-term Goals

- [ ] Comprehensive dataset collection for scientific computing
- [ ] Efficient handling of large datasets
- [ ] Integration with other modules for seamless data flow
- [ ] Support for streaming and online datasets
- [ ] Performance optimizations for data loading
- [ ] Domain-specific datasets for various scientific fields