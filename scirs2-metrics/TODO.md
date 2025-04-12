# scirs2-metrics TODO

This module provides machine learning evaluation metrics for model performance assessment.

## Current Status

- [x] Set up module structure 
- [x] Error handling
- [x] Classification metrics implementation
  - [x] Accuracy score
  - [x] Precision score
  - [x] Recall score
  - [x] F1 score
  - [x] Confusion matrix
  - [x] ROC curve metrics

- [x] Regression metrics implementation
  - [x] Mean squared error (MSE)
  - [x] Mean absolute error (MAE)
  - [x] R² score
  - [x] Explained variance score

- [x] Clustering metrics implementation
  - [x] Silhouette score
  - [x] Inertia/WCSS
  - [x] Calinski-Harabasz index

- [x] Evaluation utilities
  - [x] Train/test split
  - [x] Cross-validation
  - [x] Learning curves

## Future Tasks

- [ ] Add more classification metrics
  - [ ] Matthews correlation coefficient
  - [ ] Balanced accuracy
  - [ ] Cohen's kappa
  - [ ] Brier score

- [ ] Add more regression metrics
  - [ ] Median absolute error
  - [ ] Max error
  - [ ] Mean absolute percentage error

- [ ] Add more clustering metrics
  - [ ] Davies-Bouldin index
  - [ ] Adjusted Rand index
  - [ ] Normalized mutual information
  - [ ] Homogeneity, completeness, V-measure

- [ ] Implement advanced evaluation techniques
  - [ ] Stratified K-fold
  - [ ] Leave-one-out cross-validation
  - [ ] Time series cross-validation

- [ ] Performance optimizations
  - [ ] Parallelization of metrics calculations
  - [ ] Memory optimizations for large datasets
  - [ ] Batch processing support

## Testing and Quality Assurance

- [x] Unit tests for basic metrics
- [ ] Comprehensive test coverage
- [ ] Benchmarks against scikit-learn reference implementations
- [ ] Edge case handling (empty arrays, NaN values, etc.)

## Documentation

- [x] Basic API documentation with examples
- [ ] Tutorial notebooks demonstrating metrics usage
- [ ] Detailed mathematical explanations of metrics
- [ ] Guidance on metric selection for different tasks