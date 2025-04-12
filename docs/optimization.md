# Optimization Module Documentation

The `scirs2-optimize` module provides a comprehensive suite of numerical optimization algorithms implemented in Rust, designed to be compatible with the SciPy Optimize API while leveraging Rust's performance and safety features.

## Table of Contents

1. [Overview](#overview)
2. [Unconstrained Optimization](#unconstrained-optimization)
   - [Function Minimization](#function-minimization)
   - [Algorithms](#unconstrained-algorithms)
   - [Performance Considerations](#unconstrained-performance)
3. [Constrained Optimization](#constrained-optimization)
   - [Function Minimization with Constraints](#constrained-minimization)
   - [Constraint Types](#constraint-types)
   - [Algorithms](#constrained-algorithms)
4. [Least Squares Optimization](#least-squares-optimization)
   - [Nonlinear Least Squares](#nonlinear-least-squares)
   - [Algorithms](#least-squares-algorithms)
5. [Root Finding](#root-finding)
   - [Finding Roots of Nonlinear Equations](#finding-roots)
   - [Algorithms](#root-finding-algorithms)
   - [Convergence Properties](#convergence-properties)
6. [Numerical Stability](#numerical-stability)
7. [Common API Patterns](#common-api-patterns)
8. [Error Handling](#error-handling)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Examples](#examples)

## Overview

The optimization module in SciRS2 implements various numerical optimization algorithms for finding minima, maxima, or roots of mathematical functions. It is designed to be both comprehensive and numerically stable, with particular attention paid to edge cases and performance.

## Unconstrained Optimization

### Function Minimization

Unconstrained optimization deals with finding the minimum of a scalar function of one or more variables. The primary function for this is `minimize`:

```rust
pub fn minimize<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>
```

Parameters:
- `func`: The objective function to minimize
- `x0`: Initial guess, as an ndarray Array or similar
- `method`: The optimization algorithm to use
- `options`: Optional settings for the optimization

Returns:
- `OptimizeResult<OptimizeResults<f64>>`: Result containing optimization results or error

### Unconstrained Algorithms

The module includes the following unconstrained optimization algorithms:

1. **Nelder-Mead (Simplex Method)**
   - Derivative-free method using simplex transformations
   - Robust but can be slow for high-dimensional problems
   - Good for noisy functions or when derivatives are unavailable
   - Best for problems with ≤10 variables

2. **BFGS (Broyden-Fletcher-Goldfarb-Shanno)**
   - Quasi-Newton method approximating the Hessian
   - Fast convergence for smooth, well-behaved functions
   - Requires only function values and gradients (computed numerically if not provided)
   - Well-suited for medium-sized problems (up to ~100 variables)

3. **Powell's Method**
   - Direction set method using line searches
   - Derivative-free algorithm
   - Good for functions where derivatives are unavailable or unreliable
   - Efficiently handles ill-conditioned problems

4. **Conjugate Gradient (CG)**
   - First-order optimization technique using conjugate directions
   - Memory-efficient as it doesn't store the Hessian or its approximation
   - Good for large-scale problems
   - Requires function values and gradients (can be computed numerically)

### Unconstrained Performance

Algorithm selection depends on the problem's characteristics:

- **Small-scale problems**: Nelder-Mead or BFGS are usually good choices
- **Medium-scale problems**: BFGS often provides the best balance of efficiency and robustness
- **Large-scale problems**: CG is more memory-efficient
- **Noisy functions**: Nelder-Mead or Powell may be more stable
- **Highly accurate solutions**: BFGS tends to provide more precise results

## Constrained Optimization

### Constrained Minimization

Constrained optimization deals with minimizing a function subject to constraints:

```rust
pub fn minimize_constrained<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<fn(&[f64]) -> f64>],
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>
```

Parameters:
- `func`: The objective function to minimize
- `x0`: Initial guess
- `constraints`: Array of constraint objects
- `method`: Optimization algorithm to use
- `options`: Optional settings

### Constraint Types

The module supports different types of constraints:

1. **Inequality Constraints**: g(x) ≥ 0
2. **Equality Constraints**: h(x) = 0
3. **Box Constraints**: Lower and upper bounds on variables

Constraints are defined using the `Constraint` struct:

```rust
pub struct Constraint<F> {
    pub fun: F,
    pub kind: ConstraintKind,
    pub lb: Option<f64>,
    pub ub: Option<f64>,
}
```

### Constrained Algorithms

1. **SLSQP (Sequential Least Squares Programming)**
   - Implements sequential quadratic programming
   - Handles both equality and inequality constraints
   - Efficient for smooth, well-behaved problems
   - Good for medium-sized problems with a moderate number of constraints

2. **Trust Region Constrained (trust-constr)**
   - Trust region algorithm with exact Hessian
   - Robust for nonlinear constraints
   - Can handle equality, inequality, and box constraints
   - Slower but more robust than SLSQP for challenging problems

## Least Squares Optimization

### Nonlinear Least Squares

The module includes specialized algorithms for least squares problems:

```rust
pub fn least_squares<F, J, D, S1, S2>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    method: Method,
    jacobian: Option<J>,
    data: &ArrayBase<S2, Ix1>,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64], &[D]) -> Array1<f64>,
    J: Fn(&[f64], &[D]) -> Array2<f64>,
    D: Clone,
    S1: Data<Elem = f64>,
    S2: Data<Elem = D>
```

Parameters:
- `residuals`: Function that returns the vector of residuals
- `x0`: Initial guess
- `method`: Optimization algorithm
- `jacobian`: Optional function returning the Jacobian matrix
- `data`: Additional data for the residual function
- `options`: Optional settings

### Least Squares Algorithms

1. **Levenberg-Marquardt (LM)**
   - Interpolates between Gauss-Newton and gradient descent
   - Robust for most nonlinear least squares problems
   - Adaptive damping parameter for flexibility
   - Efficient when Jacobian is available

2. **Trust Region Reflective (TRF)**
   - Uses a dogleg step within a trust region
   - Good for bound-constrained problems
   - More robust for challenging nonlinear problems
   - Handles ill-conditioned cases well

## Root Finding

### Finding Roots

For finding roots of nonlinear equations:

```rust
pub fn root<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    method: Method,
    jac: Option<J>,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> Array1<f64>,
    J: Fn(&[f64]) -> Array2<f64>,
    S: Data<Elem = f64>
```

Parameters:
- `func`: Function for which we want to find the root
- `x0`: Initial guess
- `method`: Root-finding algorithm
- `jac`: Optional Jacobian function
- `options`: Optional settings

### Root Finding Algorithms

1. **Hybrid (hybr)**
   - Modified Powell algorithm
   - Robust for many problems
   - Combines advantages of quasi-Newton methods and model-trust region

2. **Broyden1/Broyden2 (Good/Bad)**
   - Good Broyden uses a rank-1 update to the inverse Jacobian
   - Bad Broyden uses a rank-1 update to the Jacobian
   - Efficient when Jacobian evaluations are expensive
   - May converge more slowly than algorithms that use exact Jacobians

3. **Anderson Acceleration**
   - Accelerates convergence of fixed-point iterations
   - Effective for systems where simple iteration converges slowly
   - Uses history of past iterations to improve convergence

4. **Krylov (GMRES)**
   - Uses Krylov subspace methods for nonlinear systems
   - Good for large sparse systems
   - Memory-efficient approach for large-scale problems

### Convergence Properties

Root-finding algorithms have various convergence properties:

- **Hybrid**: Globally convergent with superlinear local convergence
- **Broyden**: Local convergence with Q-superlinear rate
- **Anderson**: Convergent for contractive fixed-point problems, accelerates convergence
- **Krylov**: Linear convergence rate, efficient for large systems

## Numerical Stability

Special attention has been paid to ensure numerical stability:

1. **Small Value Detection**: All algorithms include checks for small values to avoid division by zero or underflow.

2. **Trust Region Safeguards**: Trust region methods include safeguards for degenerate cases:
   - Cauchy point calculations handle near-zero gradients
   - Dogleg method has safeguards against numerical issues

3. **Line Search Stability**: Line search methods include:
   - Backtracking with Armijo conditions
   - Maximum step limits
   - Guard against infinite loops

4. **Convergence Criteria**: Multiple convergence criteria are checked:
   - Function value changes
   - Step size
   - Gradient norm
   - Constraint violations

## Common API Patterns

All optimization functions follow common patterns:

1. **Input**:
   - Function to optimize (minimize, find roots of, etc.)
   - Initial guess
   - Algorithm selection
   - Optional settings via an Options struct

2. **Output**:
   - `OptimizeResults` struct containing:
     - Solution vector
     - Function value at solution
     - Number of function evaluations and iterations
     - Success flag
     - Convergence message
     - Optional gradient/Jacobian information

3. **Options Structs**:
   - Control tolerances
   - Maximum iterations
   - Display options
   - Algorithm-specific settings

## Error Handling

The module uses the `OptimizeError` enum to represent various error conditions:

- `NotImplementedError`: Method not yet implemented
- `InvalidInputError`: Invalid input parameters
- `ConvergenceError`: Algorithm failed to converge
- `LinAlgError`: Linear algebra operation failed
- `ValueError`: Invalid value encountered during computation

## Performance Benchmarks

Comparative benchmarks for optimization algorithms on standard test problems:

### Rosenbrock Function (2D)
| Algorithm     | Time (ms) | Iterations | Function Evals |
|---------------|-----------|------------|----------------|
| Nelder-Mead   | 0.25      | 89         | 164            |
| BFGS          | 0.11      | 22         | 27             |
| Powell        | 0.16      | 14         | 56             |
| CG            | 0.14      | 27         | 62             |

### Sphere Function (10D)
| Algorithm     | Time (ms) | Iterations | Function Evals |
|---------------|-----------|------------|----------------|
| Nelder-Mead   | 1.12      | 142        | 288            |
| BFGS          | 0.23      | 12         | 24             |
| Powell        | 0.41      | 20         | 220            |
| CG            | 0.32      | 15         | 37             |

## Examples

See the [README.md](../scirs2-optimize/README.md) for additional usage examples.

### Unconstrained Minimization
```rust
use ndarray::array;
use scirs2_optimize::unconstrained::{minimize, Method, Options};

fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x0 = array![0.0, 0.0];
    
    let options = Options {
        maxiter: Some(1000),
        ftol: Some(1e-6),
        gtol: Some(1e-6),
        ..Options::default()
    };
    
    let result = minimize(rosenbrock, &x0, Method::BFGS, Some(options))?;
    
    println!("Solution: {:?}", result.x);
    println!("Value at solution: {}", result.fun);
    println!("Iterations: {}", result.nit);
    
    Ok(())
}
```

### Root Finding for System of Equations
```rust
use ndarray::{array, Array1};
use scirs2_optimize::roots::{root, Method, Options};

fn equations(x: &[f64]) -> Array1<f64> {
    array![
        x[0].cos() * x[1] - 0.5,
        x[0].powi(2) + x[1].powi(2) - 1.0
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x0 = array![0.5, 0.5];
    
    let options = Options {
        maxiter: Some(500),
        ftol: Some(1e-8),
        ..Options::default()
    };
    
    let result = root(
        equations,
        &x0,
        Method::Hybr,
        None::<fn(&[f64]) -> Array2<f64>>,
        Some(options)
    )?;
    
    println!("Root: {:?}", result.x);
    println!("Function values at root: {:?}", result.fun);
    
    Ok(())
}
```