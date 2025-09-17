# SciRS2-Special API Reference

## Overview

The `scirs2-special` crate provides comprehensive implementations of special mathematical functions with enhanced numerical stability, performance optimizations, and extensive mathematical documentation.

## Core Function Categories

### 1. Gamma and Related Functions

#### `gamma(x) -> T`
**Description**: Computes the gamma function Γ(x) with enhanced numerical stability.

**Mathematical Definition**: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt for Re(z) > 0

**Key Properties**:
- Γ(n) = (n-1)! for positive integers n
- Γ(z+1) = z·Γ(z) (functional equation)
- Γ(z)Γ(1-z) = π/sin(πz) (reflection formula)

**Usage Examples**:
```rust
use scirs2_special::gamma;

// Integer factorial: Γ(5) = 4! = 24
let result = gamma(5.0);
assert!((result - 24.0).abs() < 1e-14);

// Half-integer: Γ(1/2) = √π
let result = gamma(0.5);
assert!((result - std::f64::consts::PI.sqrt()).abs() < 1e-14);
```

#### `gammaln(x) -> T`
**Description**: Computes the natural logarithm of the gamma function.

**Advantages**: Avoids overflow for large arguments by computing ln(Γ(x)) directly.

#### `beta(a, b) -> T`
**Description**: Computes the beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).

### 2. Bessel Functions

#### `j0(x) -> T`, `j1(x) -> T`, `jn(n, x) -> T`
**Description**: Bessel functions of the first kind.

**Mathematical Definition**: Solutions to Bessel's differential equation that are regular at x = 0.

**Physical Applications**:
- Wave propagation in cylindrical geometries
- Quantum mechanics (circular drums, atoms)
- Signal processing (Fourier-Bessel transforms)

#### `y0(x) -> T`, `y1(x) -> T`, `yn(n, x) -> T`
**Description**: Bessel functions of the second kind (Neumann functions).

**Properties**: Singular at x = 0, linearly independent from Jₙ(x).

#### `i0(x) -> T`, `i1(x) -> T`, `iv(v, x) -> T`
**Description**: Modified Bessel functions of the first kind.

**Mathematical Definition**: Iᵥ(x) = i^(-ν) Jᵥ(ix) for real arguments.

### 3. Error Functions

#### `erf(x) -> T`
**Description**: Error function erf(x) = (2/√π) ∫₀^x e^(-t²) dt.

**Applications**: 
- Probability theory (normal distribution)
- Heat conduction
- Diffusion processes

#### `erfc(x) -> T`
**Description**: Complementary error function erfc(x) = 1 - erf(x).

**Numerical Advantage**: Provides accurate computation for large x where erf(x) ≈ 1.

#### `erfinv(y) -> T`
**Description**: Inverse error function.

**Usage**: Converting between error function values and standard normal quantiles.

### 4. Statistical Functions

#### `logistic(x) -> T`
**Description**: Logistic function σ(x) = 1/(1 + e^(-x)).

#### `softmax(x: &Array1<T>) -> Array1<T>`
**Description**: Softmax function for probability distributions.

#### `logsumexp(x: &Array1<T>) -> T`
**Description**: Numerically stable computation of log(Σ exp(xᵢ)).

### 5. Combinatorial Functions

#### `factorial(n) -> T`
**Description**: Computes n! = 1×2×...×n.

#### `binomial(n, k) -> T`
**Description**: Binomial coefficient "n choose k".

#### `stirling_first(n, k) -> T`, `stirling_second(n, k) -> T`
**Description**: Stirling numbers of the first and second kind.

## Performance Features

### Memory-Efficient Operations
For large arrays, use chunked processing to avoid memory overflow:

```rust
use scirs2_special::memory_efficient::gamma_chunked;
use ndarray::Array1;

let large_array = Array1::linspace(0.1, 10.0, 1_000_000);
let result = gamma_chunked(&large_array, Some(10000)).unwrap();
```

### SIMD Acceleration
Enable with the `simd` feature for vectorized operations:

```rust
// Automatically uses SIMD when available
let input = Array1::linspace(1.0, 10.0, 1000);
let result = gamma_array(&input);
```

### Parallel Processing
Enable with the `parallel` feature for multi-threaded computation:

```rust
// Automatically parallelizes for large arrays
let input = Array1::linspace(1.0, 10.0, 100_000);
let result = gamma_array(&input); // Uses multiple threads
```

### GPU Acceleration
Enable with the `gpu` feature for GPU computation:

```rust
use scirs2_special::gpu_ops::gamma_gpu;

let input = Array1::linspace(1.0, 10.0, 100_000);
let result = gamma_gpu(&input.view()).unwrap();
```

## Error Handling

All functions return `Result<T, SpecialError>` for their safe variants:

```rust
use scirs2_special::{gamma_safe, SpecialError};

match gamma_safe(-1.0) {
    Ok(value) => println!("Gamma value: {}", value),
    Err(SpecialError::DomainError(msg)) => println!("Domain error: {}", msg),
    Err(e) => println!("Other error: {}", e),
}
```

## Feature Flags

- `default`: Basic functionality
- `simd`: SIMD optimizations
- `parallel`: Parallel processing
- `gpu`: GPU acceleration
- `high-precision`: Arbitrary precision with rug
- `python-interop`: Python compatibility helpers
- `plotting`: Visualization capabilities
- `fast-compile`: Optimized compilation for development

## Advanced Features

### High-Precision Computation
```rust
#[cfg(feature = "high-precision")]
use scirs2_special::arbitrary_precision::gamma_arbitrary;
use rug::Float;

let x = Float::with_val(256, 1.5); // 256-bit precision
let result = gamma_arbitrary(&x);
```

### Cross-Validation
```rust
use scirs2_special::cross_validation::validate_against_scipy;

// Validates implementation against SciPy
let validation_result = validate_against_scipy();
```

### Performance Benchmarking
```rust
use scirs2_special::performance_benchmarks::GammaBenchmarks;

// Run comprehensive performance tests
let config = BenchmarkConfig::default();
let results = GammaBenchmarks::run_comprehensive_benchmark(&config)?;
```

## Migration from SciPy

### Function Name Mapping
| SciPy | scirs2-special |
|-------|----------------|
| `scipy.special.gamma` | `gamma` |
| `scipy.special.loggamma` | `gammaln` |
| `scipy.special.j0` | `j0` |
| `scipy.special.erf` | `erf` |
| `scipy.special.factorial` | `factorial` |

### Array Operations
```python
# SciPy
import scipy.special as sp
import numpy as np
result = sp.gamma(np.array([1.0, 2.0, 3.0]))
```

```rust
// scirs2-special
use scirs2_special::gamma_array;
use ndarray::arr1;
let result = gamma_array(&arr1(&[1.0, 2.0, 3.0]));
```

## Examples and Tutorials

See the `examples/` directory for comprehensive tutorials:

- `bessel_interactive_tutorial.rs`: Interactive Bessel function exploration
- `comprehensive_performance_benchmark.rs`: Performance testing
- `statistical_functions_interactive_tutorial.rs`: Statistical function guide

## Mathematical References

- Abramowitz & Stegun: "Handbook of Mathematical Functions"
- NIST Digital Library of Mathematical Functions (DLMF)
- Numerical Recipes in C/C++/Fortran
- Wolfram MathWorld
- SciPy documentation and source code

## Performance Characteristics

### Timing Benchmarks (typical)
- `gamma(x)`: ~50ns per call (scalar), ~2M ops/sec (array)
- `j0(x)`: ~80ns per call (scalar), ~1.5M ops/sec (array)
- `erf(x)`: ~40ns per call (scalar), ~2.5M ops/sec (array)

### Accuracy
- Relative error typically < 2⁻⁵² (machine epsilon for f64)
- Special handling for edge cases and extreme values
- Comprehensive numerical stability analysis included

## Development and Testing

### Running Tests
```bash
# Quick tests for development
FAST_TESTS=1 cargo nextest run

# Comprehensive tests
COMPREHENSIVE_TESTS=1 cargo nextest run

# With all features
cargo nextest run --all-features
```

### Documentation
```bash
# Generate documentation
cargo doc --all-features --open

# Validate documentation
./scripts/validate_docs.sh
```

### Benchmarking
```bash
# Run performance benchmarks
cargo run --example comprehensive_performance_benchmark

# Validate benchmarking infrastructure
cargo run --example validate_benchmarking
```