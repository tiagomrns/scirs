# TODO List for scirs2-transform

## Current Progress
- ✅ Basic data normalization and standardization (min-max, z-score, l1/l2 norm)
- ✅ Normalizer class for fit-transform workflow
- ✅ Feature engineering utilities (polynomial features, binarization, discretization)
- ✅ Power transformers (Box-Cox, Yeo-Johnson)
- ✅ Dimensionality reduction (PCA, TruncatedSVD, LDA)
- ✅ Unit tests for all functionality

## Future Work

### High Priority
- [ ] Implement sparse matrix support for all transformers
- [ ] Optimize performance with BLAS/LAPACK operations where possible
- [ ] Add robust scalers (less sensitive to outliers)
- [ ] Add support for parallel computation using Rayon
- [ ] Implement QuantileTransformer for non-linear transformation

### Medium Priority
- [ ] Add more dimensionality reduction techniques:
  - [ ] Non-negative Matrix Factorization (NMF)
  - [ ] Kernel PCA
  - [ ] Multi-dimensional Scaling (MDS)
  - [ ] Locally Linear Embedding (LLE)
  - [ ] t-SNE (t-distributed Stochastic Neighbor Embedding)
  - [ ] UMAP (Uniform Manifold Approximation and Projection)
- [ ] Add more feature selection algorithms:
  - [ ] Univariate feature selection
  - [ ] Recursive feature elimination
  - [ ] Feature importance based selection
- [ ] Implement feature hashing (Vectorizer)

### Low Priority
- [ ] Add data imputation strategies
- [ ] Support for categorical variable encoding
- [ ] Pipeline API for chaining transformations
- [ ] Support for custom transformers
- [ ] Integration with other scirs2 modules

## Performance Improvements
- [ ] Optimize matrix multiplication for large datasets
- [ ] Implement in-place transformations where possible
- [ ] Add benchmarks against scikit-learn
- [ ] Memory optimization for large datasets

## Documentation
- [ ] Add more comprehensive examples
- [ ] Improve API documentation
- [ ] Add tutorials for common data transformation tasks
- [ ] Benchmark comparisons with scikit-learn