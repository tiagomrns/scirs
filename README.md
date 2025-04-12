# SciRS2 - Scientific Computing and AI in Rust

![SciRS2 CI](https://github.com/cool-japan/scirs/workflows/SciRS2%20CI/badge.svg)
![Advanced & AI/ML Modules](https://github.com/cool-japan/scirs/workflows/SciRS2%20Advanced%20%26%20AI%2FML%20Modules/badge.svg)
![Documentation](https://github.com/cool-japan/scirs/workflows/SciRS2%20Documentation/badge.svg)

SciRS2 is an ambitious project to provide a comprehensive scientific computing and AI/ML infrastructure in Rust, providing SciPy-compatible APIs while leveraging Rust's performance and safety features.

## Project Goals

- Create a comprehensive scientific computing and machine learning library in Rust
- Maintain API compatibility with SciPy where reasonable
- Provide specialized tools for AI and machine learning development
- Leverage Rust's performance, safety, and concurrency features
- Build a sustainable open-source ecosystem for scientific and AI computing in Rust

## Project Structure

SciRS2 adopts a modular architecture with separate crates for different functional areas, using Rust's workspace feature to manage them:

```
/
# Core Scientific Computing Modules
├── Cargo.toml                # Workspace configuration
├── scirs2-core/              # Core utilities and common functionality
├── scirs2-linalg/            # Linear algebra module
├── scirs2-integrate/         # Numerical integration
├── scirs2-interpolate/       # Interpolation algorithms
├── scirs2-optimize/          # Optimization algorithms
├── scirs2-fft/               # Fast Fourier Transform
├── scirs2-stats/             # Statistical functions
├── scirs2-special/           # Special mathematical functions
├── scirs2-signal/            # Signal processing
├── scirs2-sparse/            # Sparse matrix operations
├── scirs2-spatial/           # Spatial algorithms

# Future Advanced Modules
├── scirs2-cluster/           # Clustering algorithms
├── scirs2-ndimage/           # N-dimensional image processing
├── scirs2-io/                # Input/output utilities
├── scirs2-datasets/          # Sample datasets and loaders

# AI/ML Modules
├── scirs2-autograd/          # Automatic differentiation engine
├── scirs2-neural/            # Neural network building blocks
├── scirs2-optim/             # ML-specific optimization algorithms
├── scirs2-graph/             # Graph processing algorithms
├── scirs2-transform/         # Data transformation utilities
├── scirs2-metrics/           # ML evaluation metrics
├── scirs2-text/              # Text processing utilities
├── scirs2-vision/            # Computer vision operations
├── scirs2-series/            # Time series analysis

# Main Integration Crate
└── scirs2/                   # Main integration crate
    ├── Cargo.toml
    └── src/
        └── lib.rs            # Re-exports from all other crates
```

This modular architecture offers several advantages:
- **Flexible Dependencies**: Users can select only the features they need
- **Independent Development**: Each module can be developed and tested separately
- **Clear Separation**: Each module focuses on a specific functional area
- **No Circular Dependencies**: Clear hierarchy prevents circular dependencies
- **AI/ML Focus**: Specialized modules for machine learning and AI workloads

## Module Documentation

Each module has its own README with detailed documentation:

### Core Modules
- [**scirs2**](scirs2/README.md): Main integration crate
- [**scirs2-core**](scirs2-core/README.md): Core utilities and common functionality
- [**scirs2-linalg**](scirs2-linalg/README.md): Linear algebra module
- [**scirs2-integrate**](scirs2-integrate/README.md): Numerical integration
- [**scirs2-interpolate**](scirs2-interpolate/README.md): Interpolation algorithms
- [**scirs2-optimize**](scirs2-optimize/README.md): Optimization algorithms
- [**scirs2-fft**](scirs2-fft/README.md): Fast Fourier Transform
- [**scirs2-stats**](scirs2-stats/README.md): Statistical functions
- [**scirs2-special**](scirs2-special/README.md): Special mathematical functions
- [**scirs2-signal**](scirs2-signal/README.md): Signal processing
- [**scirs2-sparse**](scirs2-sparse/README.md): Sparse matrix operations
- [**scirs2-spatial**](scirs2-spatial/README.md): Spatial algorithms

### Advanced Modules
- [**scirs2-cluster**](scirs2-cluster/README.md): Clustering algorithms
- [**scirs2-ndimage**](scirs2-ndimage/README.md): N-dimensional image processing
- [**scirs2-io**](scirs2-io/README.md): Input/output utilities
- [**scirs2-datasets**](scirs2-datasets/README.md): Sample datasets and loaders

### AI/ML Modules
- [**scirs2-autograd**](scirs2-autograd/README.md): Automatic differentiation engine
- [**scirs2-neural**](scirs2-neural/README.md): Neural network building blocks
- [**scirs2-optim**](scirs2-optim/README.md): ML-specific optimization algorithms
- [**scirs2-graph**](scirs2-graph/README.md): Graph processing algorithms
- [**scirs2-transform**](scirs2-transform/README.md): Data transformation utilities
- [**scirs2-metrics**](scirs2-metrics/README.md): ML evaluation metrics
- [**scirs2-text**](scirs2-text/README.md): Text processing utilities
- [**scirs2-vision**](scirs2-vision/README.md): Computer vision operations
- [**scirs2-series**](scirs2-series/README.md): Time series analysis

## Implementation Strategy

We follow a phased approach:

1. **Core functionality analysis**: Identify key features and APIs of each SciPy module
2. **Prioritization**: Begin with highest-demand modules (linalg, stats, optimize)
3. **Interface design**: Balance Rust idioms with SciPy compatibility
4. **Scientific computing foundation**: Implement core scientific computing modules first
5. **Advanced modules**: Implement specialized modules for advanced scientific computing
6. **AI/ML infrastructure**: Develop specialized tools for AI and machine learning
7. **Integration and optimization**: Ensure all modules work together efficiently

## Dependencies

SciRS2 leverages the Rust ecosystem:

### Core Dependencies
- `ndarray` or `nalgebra`: Multidimensional array operations
- `num`: Numeric abstractions
- `rayon`: Parallel processing
- `rustfft`: Fast Fourier transforms
- `ndarray-linalg` or `linxal`: Linear algebra computations
- `argmin`: Optimization algorithms
- `rand` and `rand_distr`: Random number generation and distributions

### Future AI/ML Dependencies
- `tch-rs`: Bindings to the PyTorch C++ API
- `burn`: Pure Rust neural network framework
- `pandrs`: Fast DataFrame library for data manipulation (upcoming crate)
- `numrs`: NumPy-like array operations in Rust (upcoming crate)
- `tokenizers`: Fast tokenization utilities
- `image`: Image processing utilities
- `petgraph`: Graph algorithms and data structures

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under [Apache License 2.0](LICENSE).

## Current Status

### Completed Modules

The following core scientific computing modules have been implemented:

- **Linear Algebra Module** (`scirs2-linalg`): Basic matrix operations, decompositions (LU, QR, SVD), eigenvalue problems, BLAS/LAPACK interfaces
- **Statistics Module** (`scirs2-stats`): Descriptive statistics, distributions, statistical tests, regression models
- **Optimization Module** (`scirs2-optimize`): Unconstrained & constrained optimization, least squares, root finding
- **Integration Module** (`scirs2-integrate`): Numerical integration, ODE solvers
- **Interpolation Module** (`scirs2-interpolate`): 1D & ND interpolation, splines
- **Signal Processing** (`scirs2-signal`): Filtering, convolution, spectral analysis
- **FFT Module** (`scirs2-fft`): FFT, inverse FFT, DCT, DST
- **Sparse Matrix** (`scirs2-sparse`): Various sparse matrix formats and operations
- **Special Functions** (`scirs2-special`): Mathematical special functions
- **Spatial Algorithms** (`scirs2-spatial`): KD-trees, distance calculations

### Current Phase: AI and Machine Learning Modules

We have completed the implementation of core scientific computing modules, advanced modules, and now we're focusing on AI/ML infrastructure:

#### Completed Advanced Modules:
- **Clustering** (`scirs2-cluster`): Vector quantization, hierarchical clustering, density-based clustering
- **N-dimensional Image Processing** (`scirs2-ndimage`): Filtering, morphology, measurements
- **I/O utilities** (`scirs2-io`): MATLAB, WAV, and ARFF file formats, image handling
- **Datasets** (`scirs2-datasets`): Sample datasets and loaders

#### Completed AI/ML Modules:
- **Automatic Differentiation** (`scirs2-autograd`): Reverse-mode automatic differentiation, tensor operations, neural network primitives
- **Neural Networks** (`scirs2-neural`): Neural network layers, activations, loss functions
- **ML Optimization** (`scirs2-optim`): Optimizers, learning rate schedulers, regularization techniques
- **Graph Processing** (`scirs2-graph`): Graph algorithms and data structures
- **Data Transformation** (`scirs2-transform`): Feature engineering, normalization
- **Evaluation Metrics** (`scirs2-metrics`): Classification, regression metrics
- **Text Processing** (`scirs2-text`): Tokenization, vectorization
- **Computer Vision** (`scirs2-vision`): Image processing, feature detection
- **Time Series Analysis** (`scirs2-series`): Decomposition, forecasting

### Current Focus: Refinement and Optimization

We are now focusing on refining the existing modules, improving performance, and ensuring comprehensive test coverage and documentation:
- Performance optimization
- API refinement
- Comprehensive documentation
- Integration testing
- User feedback incorporation
- Benchmark comparison with other libraries

For more detailed information on development status and roadmap, check the [TODO.md](TODO.md) file.