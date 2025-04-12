# Usage Guidelines for scirs2-core

This document provides guidelines for using scirs2-core functionality in other modules of the scirs2 library.

## Module Overview

scirs2-core provides foundational functionality for the entire scirs2 ecosystem:

- **Error Handling**: Core error types and error propagation mechanisms
- **Configuration**: Global and thread-local configuration system
- **Numeric Types**: Traits for scientific numeric operations
- **Validation**: Input validation utilities
- **I/O Utilities**: Common file operations with proper error handling
- **Performance**: Caching, SIMD acceleration, and parallel computing utilities
- **Constants**: Mathematical and physical constants

## Configuration System

### Global Configuration

Global configuration affects all threads and components:

```rust
use scirs2_core::config::{set_global_config, get_config, Config};

// Create a configuration
let mut config = Config::default();
config.set_max_threads(4);
config.set_precision(1e-8);

// Set as global configuration
set_global_config(config);

// Use configuration
let conf = get_config();
let precision = conf.get_precision();
```

### Thread-Local Configuration

Thread-local configuration affects only the current thread:

```rust
use scirs2_core::config::{set_thread_local_config, clear_thread_local_config};

// Set thread-local configuration
let mut config = Config::default();
config.set_max_threads(2);
set_thread_local_config(config);

// Later, clear the thread-local configuration to revert to global
clear_thread_local_config();
```

### Configuration from Environment

Load configuration from environment variables:

```rust
use scirs2_core::config::Config;

// Load from environment variables
let config = Config::from_env();

// SCIRS_MAX_THREADS sets max_threads
// SCIRS_PRECISION sets precision
// etc.
```

## Numeric Traits

### Using Numeric Traits

scirs2-core provides traits for scientific numeric operations:

```rust
use scirs2_core::numeric::{ScientificNumber, RealNumber};

// Generic function that works with any scientific number
fn compute_magnitude<T: ScientificNumber>(values: &[T]) -> T {
    let sum_of_squares = values.iter()
        .map(|&x| x.square())
        .fold(T::zero(), |a, b| a + b);
    
    sum_of_squares.sqrt()
}

// Function specifically for real numbers
fn normalize<T: RealNumber>(values: &mut [T]) {
    let magnitude = compute_magnitude(values);
    if magnitude > T::epsilon() {
        for value in values.iter_mut() {
            *value = *value / magnitude;
        }
    }
}
```

### Using Scalar Type

The `Scalar` type provides a safe wrapper for numeric values:

```rust
use scirs2_core::numeric::Scalar;

let a = Scalar::new(2.5);
let b = Scalar::new(1.5);

let c = a + b;  // Scalar::new(4.0)
let d = a * b;  // Scalar::new(3.75)
let e = a.sqrt();  // Scalar::new(1.5811)
```

## Validation Utilities

### Array Validation

Validate array shapes and contents:

```rust
use ndarray::Array2;
use scirs2_core::validation::{check_shape, check_array_finite};
use scirs2_core::CoreResult;

fn process_matrix(matrix: &Array2<f64>) -> CoreResult<Array2<f64>> {
    // Check shape: must be 3x3
    check_shape(matrix, &[3, 3], "matrix")?;
    
    // Check all elements are finite
    check_array_finite(matrix, "matrix")?;
    
    // Process the matrix...
    Ok(result)
}
```

### Numeric Validation

Validate numeric inputs:

```rust
use scirs2_core::validation::{check_positive, check_in_bounds};
use scirs2_core::CoreResult;

fn compute_with_parameters(alpha: f64, beta: f64) -> CoreResult<f64> {
    // Alpha must be positive
    check_positive(alpha, "alpha")?;
    
    // Beta must be between 0 and 1
    check_in_bounds(beta, 0.0, 1.0, "beta")?;
    
    // Compute result...
    Ok(result)
}
```

## I/O Utilities

### File Operations

Common file operations with proper error handling:

```rust
use scirs2_core::io::{read_to_string, write_string, directory_exists, create_directory};
use scirs2_core::CoreResult;

fn save_data(path: &str, data: &str) -> CoreResult<()> {
    // Ensure directory exists
    let dir = std::path::Path::new(path).parent().unwrap();
    if !directory_exists(dir)? {
        create_directory(dir)?;
    }
    
    // Write data
    write_string(path, data)?;
    
    Ok(())
}

fn load_data(path: &str) -> CoreResult<String> {
    read_to_string(path)
}
```

## Performance Utilities

### Caching and Memoization

Use caching to avoid redundant computations:

```rust
use scirs2_core::cache::{TTLSizedCache, CacheBuilder};
use std::hash::Hash;

// Create a cache with TTL
let mut cache = TTLSizedCache::<String, Vec<f64>>::new(1000, 3600);

// Cache a result
cache.insert("key1".to_string(), vec![1.0, 2.0, 3.0]);

// Retrieve from cache
if let Some(result) = cache.get(&"key1".to_string()) {
    // Use cached result
} else {
    // Compute result and cache it
    let result = compute_expensive_operation();
    cache.insert("key1".to_string(), result);
}

// Use builder pattern for more complex caches
let cache = CacheBuilder::new()
    .with_size(10000)
    .with_ttl(3600)
    .thread_safe()
    .build_sized_cache::<String, Vec<f64>>();
```

### SIMD and Parallel Processing

Enable SIMD and parallel processing features for performance:

```toml
# In Cargo.toml
[dependencies]
scirs2-core = { version = "0.1.0", features = ["simd", "parallel"] }
```

## Common Utilities

### Array Generation

Create linearly and logarithmically spaced arrays:

```rust
use scirs2_core::utils::{linspace, logspace};
use ndarray::Array1;

// Linear space from 0 to 10 with 11 points
let linear = linspace(0.0, 10.0, 11);  // [0.0, 1.0, 2.0, ..., 10.0]

// Log space from 10^0 to 10^3 with 4 points
let logarithmic = logspace(0.0, 3.0, 4, None);  // [1.0, 10.0, 100.0, 1000.0]

// Log space with custom base (2)
let log_base2 = logspace(0.0, 3.0, 4, Some(2.0));  // [1.0, 2.0, 4.0, 8.0]
```

### Element-wise Operations

Perform element-wise operations on arrays:

```rust
use scirs2_core::utils::{maximum, minimum};
use ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![[5.0, 1.0], [2.0, 6.0]];

// Element-wise maximum: [[5.0, 2.0], [3.0, 6.0]]
let max_ab = maximum(&a, &b);

// Element-wise minimum: [[1.0, 1.0], [2.0, 4.0]]
let min_ab = minimum(&a, &b);
```

## Best Practices

### Dependency Management

1. Keep dependencies minimal:
   - Use scirs2-core for common functionality
   - Avoid external dependencies when scirs2-core provides equivalent functionality

2. Follow the modular structure:
   - Each module should depend only on scirs2-core, not on other modules
   - The main scirs2 crate should depend on all modules

### Error Handling

1. Define a module-specific error type:
   ```rust
   pub enum MyModuleError {
       InvalidParameter(ErrorContext),
       ComputationFailed(ErrorContext),
       // ... other variants
   }
   ```

2. Implement conversion to CoreError:
   ```rust
   impl From<MyModuleError> for CoreError {
       fn from(err: MyModuleError) -> Self {
           match err {
               MyModuleError::InvalidParameter(ctx) => 
                   CoreError::ValueError(ctx),
               // ... handle other variants
           }
       }
   }
   ```

3. Use Result type with your module's error:
   ```rust
   pub type MyModuleResult<T> = Result<T, MyModuleError>;
   ```

### Configuration

1. Use the global configuration system for library-wide settings
2. Use thread-local configuration for performance-critical code
3. Provide sensible defaults for all configuration options

### Numeric Operations

1. Use the most generic trait possible for maximum flexibility
2. Fall back to concrete types only when necessary
3. Validate inputs before performing operations

### Testing

1. Test all public API functions
2. Use property-based testing for mathematical properties
3. Include edge cases in your tests