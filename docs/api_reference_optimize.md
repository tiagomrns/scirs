# SciRS2 Optimize API Reference

This document provides a detailed reference of the API for the `scirs2-optimize` crate.

## Table of Contents

- [Modules](#modules)
- [Unconstrained Optimization](#unconstrained-optimization)
- [Constrained Optimization](#constrained-optimization)
- [Least Squares Optimization](#least-squares-optimization)
- [Root Finding](#root-finding)
- [Error Handling](#error-handling)
- [Result Types](#result-types)

## Modules

The `scirs2-optimize` crate is organized into the following modules:

- `unconstrained`: Functions and types for unconstrained optimization
- `constrained`: Functions and types for constrained optimization
- `least_squares`: Functions and types for least squares problems
- `roots`: Functions and types for root finding
- `result`: Common result types and utilities
- `error`: Error types and handling

## Unconstrained Optimization

### Types

#### `Method` Enum
```rust
pub enum Method {
    NelderMead,
    Powell,
    CG,
    BFGS,
    LBFGS,
    NewtonCG,
    TrustNCG,
    TrustKrylov,
    TrustExact,
    TNC,
    SLSQP,
}
```

#### `Options` Struct
```rust
pub struct Options {
    pub maxiter: Option<usize>,
    pub ftol: Option<f64>,
    pub gtol: Option<f64>,
    pub eps: Option<f64>,
    pub finite_diff_rel_step: Option<f64>,
    pub disp: bool,
    pub return_all: bool,
}
```

### Functions

#### `minimize` Function
```rust
pub fn minimize<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
```

Minimizes a scalar function of one or more variables.

- `func`: A function that takes a slice of values and returns a scalar
- `x0`: The initial guess
- `method`: The optimization method to use
- `options`: Options for the optimizer
- Returns: `OptimizeResults` containing the optimization results

#### Algorithm-Specific Functions

```rust
fn minimize_nelder_mead<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn minimize_bfgs<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn minimize_powell<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn minimize_conjugate_gradient<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
```

## Constrained Optimization

### Types

#### `Method` Enum
```rust
pub enum Method {
    SLSQP,
    TrustConstr,
    COBYLA,
}
```

#### `Options` Struct
```rust
pub struct Options {
    pub maxiter: Option<usize>,
    pub ftol: Option<f64>,
    pub gtol: Option<f64>,
    pub ctol: Option<f64>,
    pub eps: Option<f64>,
    pub disp: bool,
    pub return_all: bool,
}
```

#### `Constraint` Struct
```rust
pub struct Constraint<F> {
    pub fun: F,
    pub kind: ConstraintKind,
    pub lb: Option<f64>,
    pub ub: Option<f64>,
}
```

#### `ConstraintKind` Enum
```rust
pub enum ConstraintKind {
    Equality,
    Inequality,
}
```

### Functions

#### `minimize_constrained` Function
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
    S: Data<Elem = f64>,
```

Minimizes a scalar function of one or more variables with constraints.

- `func`: A function that takes a slice of values and returns a scalar
- `x0`: The initial guess
- `constraints`: Vector of constraints
- `method`: The optimization method to use
- `options`: Options for the optimizer
- Returns: `OptimizeResults` containing the optimization results

#### Algorithm-Specific Functions

```rust
fn minimize_slsqp<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<fn(&[f64]) -> f64>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn minimize_trust_constr<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<fn(&[f64]) -> f64>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
```

## Least Squares Optimization

### Types

#### `Method` Enum
```rust
pub enum Method {
    TrustRegionReflective,
    LevenbergMarquardt,
    Dogbox,
}
```

#### `Options` Struct
```rust
pub struct Options {
    pub max_nfev: Option<usize>,
    pub xtol: Option<f64>,
    pub ftol: Option<f64>,
    pub gtol: Option<f64>,
    pub verbose: usize,
    pub diff_step: Option<f64>,
    pub use_finite_diff: bool,
}
```

### Functions

#### `least_squares` Function
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
    S2: Data<Elem = D>,
```

Solve a nonlinear least-squares problem.

- `residuals`: Function that returns the residuals
- `x0`: Initial guess for the parameters
- `method`: Method to use for solving the problem
- `jacobian`: Jacobian of the residuals (optional)
- `data`: Additional data to pass to the residuals and jacobian functions
- `options`: Options for the solver
- Returns: `OptimizeResults` containing the optimization results

#### Algorithm-Specific Functions

```rust
fn least_squares_lm<F, J, D, S1, S2>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    jacobian: Option<J>,
    data: &ArrayBase<S2, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn least_squares_trf<F, J, D, S1, S2>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    jacobian: Option<J>,
    data: &ArrayBase<S2, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
```

## Root Finding

### Types

#### `Method` Enum
```rust
pub enum Method {
    Hybr,
    Broyden1,
    Broyden2,
    Anderson,
    LineSearch,
    Krylov,
    DiagBroyden,
    ExcitingMixing,
    DFPose,
}
```

#### `Options` Struct
```rust
pub struct Options {
    pub xtol: Option<f64>,
    pub ftol: Option<f64>,
    pub maxiter: Option<usize>,
    pub eps: Option<f64>,
    pub disp: bool,
    pub return_all: bool,
}
```

### Functions

#### `root` Function
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
    S: Data<Elem = f64>,
```

Find the roots of a vector function.

- `func`: A function that takes a vector and returns a vector
- `x0`: The initial guess
- `method`: Type of solver
- `jac`: Function returning the Jacobian matrix of `func`
- `options`: Extra options for the solver
- Returns: `OptimizeResults` containing the root finding results

#### Algorithm-Specific Functions

```rust
fn root_hybr<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    jac: Option<J>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn root_broyden1<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    jac: Option<J>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn root_anderson<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    jac: Option<J>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>

fn root_krylov<F, J, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    jac: Option<J>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
```

## Error Handling

### `OptimizeError` Enum
```rust
pub enum OptimizeError {
    NotImplementedError(String),
    InvalidInputError(String),
    ConvergenceError(String),
    LinAlgError(String),
    ValueError(String),
}
```

Error types used in optimization functions:

- `NotImplementedError`: Method not yet implemented
- `InvalidInputError`: Invalid input parameters
- `ConvergenceError`: Algorithm failed to converge
- `LinAlgError`: Linear algebra operation failed
- `ValueError`: Invalid value encountered during computation

### `OptimizeResult` Type
```rust
pub type OptimizeResult<T> = Result<T, OptimizeError>;
```

A specialized Result type for optimization operations.

## Result Types

### `OptimizeResults` Struct
```rust
pub struct OptimizeResults<T> {
    pub x: Array1<f64>,
    pub fun: T,
    pub jac: Option<Vec<f64>>,
    pub hess: Option<Array2<f64>>,
    pub hess_inv: Option<Array2<f64>>,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    pub nit: usize,
    pub status: i32,
    pub message: String,
    pub success: bool,
    pub constr: Option<Array1<f64>>,
    pub constr_violation: Option<f64>,
}
```

Container for the results of an optimization routine:

- `x`: The solution array
- `fun`: Value of the objective function at the solution
- `jac`: Jacobian of the objective function at the solution (if available)
- `hess`: Hessian of the objective function at the solution (if available)
- `hess_inv`: Inverse of the Hessian at the solution (if available)
- `nfev`: Number of function evaluations
- `njev`: Number of Jacobian evaluations
- `nhev`: Number of Hessian evaluations
- `nit`: Number of iterations
- `status`: Status of the solution (0 for success)
- `message`: Description of the cause of termination
- `success`: Whether the optimization was successful
- `constr`: Constraint values at the solution (if available)
- `constr_violation`: Magnitude of constraint violation (if available)