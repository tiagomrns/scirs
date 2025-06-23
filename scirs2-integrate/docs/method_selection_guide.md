# Best Practices for Method Selection

This guide helps you choose the optimal integration method for your specific problem, ensuring maximum accuracy and performance.

## 🎯 Quick Method Selection Decision Tree

```
Problem Type?
├─ Integration (Quadrature)
│  ├─ 1D Integration → See "1D Integration Methods"
│  ├─ Multi-dimensional → See "Multi-dimensional Integration"
│  └─ Infinite/Semi-infinite → tanhsinh, QMC
├─ ODE/IVP
│  ├─ Non-stiff → RK45, RK23, DOP853
│  ├─ Stiff → BDF, Radau, LSODA
│  ├─ Hamiltonian → Symplectic integrators
│  └─ With events → Any method + event detection
├─ DAE
│  ├─ Index-1 → BDF-DAE, Radau-DAE
│  └─ Higher index → Index reduction + BDF
├─ BVP
│  ├─ Linear → Collocation methods
│  └─ Nonlinear → Newton iteration + collocation
└─ PDE
   ├─ Elliptic → Finite elements, spectral
   ├─ Parabolic → Method of lines + ODE solver
   └─ Hyperbolic → Method of lines + explicit methods
```

## 1D Integration Methods

### When to Use Each Method

| Method | Best For | Pros | Cons | Performance |
|--------|----------|------|------|-------------|
| `quad` | General purpose, unknown smoothness | Adaptive, robust | Moderate overhead | ⭐⭐⭐⭐ |
| `gauss_legendre` | Smooth functions, high accuracy | Very accurate for polynomial-like | Fixed points | ⭐⭐⭐⭐⭐ |
| `simpson` | Moderately smooth, uniform grid | Simple, predictable | Fixed step size | ⭐⭐⭐ |
| `trapezoid` | Quick estimates, irregular data | Fast, stable | Low accuracy | ⭐⭐ |
| `romberg` | Very smooth functions | High convergence rate | Requires smooth function | ⭐⭐⭐⭐ |
| `tanhsinh` | Endpoint singularities | Handles singularities well | Setup overhead | ⭐⭐⭐⭐ |

### Examples with Use Cases

```rust
use scirs2_integrate::*;

// Smooth polynomial function → Gaussian quadrature
let smooth_result = gaussian::gauss_legendre(|x: f64| x.powi(4), 0.0, 1.0, 10)?;

// Unknown smoothness → Adaptive quadrature
let unknown_result = quad::quad(|x: f64| x.sin() * (-x*x).exp(), 0.0, 10.0, None)?;

// Endpoint singularity → Tanh-sinh
let singular_result = tanhsinh::tanhsinh(|x: f64| (-x.ln()).sqrt(), 0.0, 1.0, None)?;

// Quick estimate → Simpson's rule
let quick_result = quad::simpson(|x: f64| x.cos(), 0.0, std::f64::consts::PI, 100)?;
```

## Multi-dimensional Integration

### Method Selection Criteria

| Dimensions | Smoothness | Method | Justification |
|------------|------------|--------|---------------|
| 2-3 | High | `cubature` adaptive | Efficient for low dimensions |
| 2-3 | Low | `monte_carlo` | Robust to discontinuities |
| 4-10 | Any | `qmc_quad` | Better convergence than MC |
| 10+ | Any | `monte_carlo_parallel` | Only feasible method |

### Performance vs Accuracy Trade-offs

```rust
use scirs2_integrate::*;
use ndarray::ArrayView1;

// High accuracy for smooth 2D function
let precise = cubature::nquad(
    |x: ArrayView1<f64>| (x[0]*x[0] + x[1]*x[1]).exp(),
    &[cubature::Bound::Finite(0.0, 1.0); 2],
    Some(cubature::CubatureOptions::new().max_evals(100000))
)?;

// Balanced approach for moderate dimensions
let balanced = qmc::qmc_quad(
    |x: ArrayView1<f64>| x.iter().map(|&xi| xi.sin()).product(),
    &[(0.0, 1.0); 5], // 5D
    Some(qmc::QMCQuadResult::default())
)?;

// High-dimensional with parallelization
let high_dim = monte_carlo_parallel::parallel_monte_carlo(
    |x: ArrayView1<f64>| (-x.iter().map(|&xi| xi*xi).sum::<f64>()).exp(),
    &[(0.0, 1.0); 20], // 20D
    Some(monte_carlo_parallel::ParallelMonteCarloOptions::new().workers(8))
)?;
```

## ODE Method Selection

### Stiffness Detection

Use this heuristic to detect stiffness:

```rust
use scirs2_integrate::ode::*;
use ndarray::{array, ArrayView1};

fn detect_stiffness<F>(rhs: F, y0: ArrayView1<f64>, t_span: [f64; 2]) -> bool 
where 
    F: Fn(f64, ArrayView1<f64>) -> ndarray::Array1<f64>
{
    // Try explicit method first
    let explicit_result = solve_ivp(
        &rhs, t_span, y0.to_owned(), 
        Some(ODEOptions::new().method(ODEMethod::RK23).rtol(1e-3))
    );
    
    // If explicit method struggles, likely stiff
    match explicit_result {
        Ok(sol) => sol.nfev > 10000, // Too many function evaluations
        Err(_) => true, // Failed - definitely stiff
    }
}
```

### Method Recommendations by Problem Type

#### Non-stiff Systems

```rust
// Default: Dormand-Prince RK45 (best all-around)
let options = ODEOptions::new().method(ODEMethod::RK45);

// High precision: 8th-order DOP853
let high_precision = ODEOptions::new().method(ODEMethod::DOP853);

// Quick integration: RK23
let fast = ODEOptions::new().method(ODEMethod::RK23);
```

#### Stiff Systems

```rust
// Default stiff: Backward Differentiation Formula
let stiff_options = ODEOptions::new().method(ODEMethod::BDF);

// L-stable: Radau IIA
let l_stable = ODEOptions::new().method(ODEMethod::Radau);

// Automatic switching: LSODA
let automatic = ODEOptions::new().method(ODEMethod::LSODA);
```

#### Special Cases

```rust
// Hamiltonian/Conservative systems
use scirs2_integrate::symplectic::*;
let hamiltonian_system = SeparableHamiltonian {
    kinetic: |p: ArrayView1<f64>| 0.5 * p.iter().map(|&pi| pi*pi).sum(),
    potential: |q: ArrayView1<f64>| 0.5 * q.iter().map(|&qi| qi*qi).sum(),
};

// Oscillatory problems
let options = ODEOptions::new()
    .method(ODEMethod::DOP853) // High-order for oscillations
    .rtol(1e-10); // Tight tolerance

// Systems with discontinuities
let with_events = ODEOptionsWithEvents::new()
    .add_event(EventSpec::new(
        |_t, y| y[0], // Event function
        EventDirection::Decreasing,
        false // Non-terminal
    ));
```

## DAE Method Selection

### By DAE Structure

| DAE Type | Index | Recommended Method | Example |
|----------|-------|-------------------|---------|
| Semi-explicit | 1 | `bdf_semi_explicit_dae` | Constrained mechanics |
| Fully implicit | 1 | `bdf_implicit_dae` | Circuit equations |
| Higher index | 2-3 | `solve_higher_index_dae` | Multibody dynamics |
| Large sparse | Any | `krylov_bdf_implicit_dae` | Discretized PDEs |

### Implementation Examples

```rust
use scirs2_integrate::dae::*;
use ndarray::{array, ArrayView1};

// Pendulum (index-3 DAE reduced to index-1)
let mass_matrix = |_t: f64, _y: ArrayView1<f64>| {
    // Implementation depends on specific system
    ndarray::Array2::eye(4)
};

let result = solve_higher_index_dae(
    rhs_function,
    constraint_function,
    [0.0, 10.0],
    initial_conditions,
    Some(DAEOptions::new().index(DAEIndex::Three))
)?;
```

## PDE Method Selection

### By PDE Type and Dimensionality

| PDE Type | Dimension | Best Method | Implementation |
|----------|-----------|-------------|----------------|
| Elliptic | 1D | Spectral | `ChebyshevSpectralSolver1D` |
| Elliptic | 2D | Finite Element | `FEMPoissonSolver` |
| Parabolic | 1D-3D | Method of Lines | `MOLParabolicSolver1D/2D/3D` |
| Hyperbolic | 1D | Method of Lines | `MOLWaveEquation1D` |
| Mixed | Any | Spectral Element | `SpectralElementPoisson2D` |

### Examples by Problem Characteristics

```rust
use scirs2_integrate::pde::*;

// High accuracy elliptic (Poisson equation)
let spectral_solver = SpectralElementPoisson2D::new(
    domain,
    SpectralElementOptions::new().polynomial_order(8)
)?;

// Complex geometry
let fem_solver = FEMPoissonSolver::new(
    irregular_mesh,
    FEMOptions::new().element_type(ElementType::Quadratic)
)?;

// Time-dependent parabolic (heat equation)
let mol_solver = MOLParabolicSolver2D::new(
    spatial_domain,
    MOLOptions::new().spatial_method(FiniteDifferenceScheme::Central)
)?;
```

## Performance Optimization Guidelines

### 1. Tolerance Selection

```rust
// For most engineering applications
let standard_tol = ODEOptions::new().rtol(1e-6).atol(1e-9);

// For high-precision requirements
let precise_tol = ODEOptions::new().rtol(1e-12).atol(1e-15);

// For quick estimates
let rough_tol = ODEOptions::new().rtol(1e-3).atol(1e-6);
```

### 2. Step Size Control

```rust
// Stable systems - larger steps
let stable = ODEOptions::new().max_step(0.1);

// Oscillatory systems - smaller max step
let oscillatory = ODEOptions::new().max_step(0.01);

// Automatic (recommended for most cases)
let automatic = ODEOptions::new(); // No max_step specified
```

### 3. Hardware Optimization

```rust
use scirs2_integrate::autotuning::*;

// Automatic hardware detection and optimization
let tuner = AutoTuner::new();
let profile = tuner.create_profile();

// Apply to your solver
let optimized_options = ODEOptions::new()
    .use_simd(profile.simd_features.contains(&SimdFeature::Avx2))
    .parallel_jacobian(profile.hardware_info.num_cores > 4);
```

## Common Anti-patterns and How to Avoid Them

### ❌ Wrong: Using stiff solver for non-stiff problems
```rust
// Inefficient for non-stiff systems
let bad = ODEOptions::new().method(ODEMethod::BDF);
```

### ✅ Right: Auto-detection or appropriate method
```rust
// Let LSODA choose automatically
let good = ODEOptions::new().method(ODEMethod::LSODA);
// Or use explicit for known non-stiff
let explicit = ODEOptions::new().method(ODEMethod::RK45);
```

### ❌ Wrong: Too tight tolerances
```rust
// Unnecessarily expensive
let overtight = ODEOptions::new().rtol(1e-15).atol(1e-18);
```

### ✅ Right: Problem-appropriate tolerances
```rust
// Engineering accuracy
let reasonable = ODEOptions::new().rtol(1e-8).atol(1e-11);
```

### ❌ Wrong: Ignoring problem structure
```rust
// Missing conservation properties
let generic = solve_ivp(hamiltonian_rhs, t_span, y0, None);
```

### ✅ Right: Structure-preserving methods
```rust
// Conserves energy
let symplectic = symplectic::velocity_verlet(hamiltonian_system, t_span, q0, p0)?;
```

## Method Selection Flowchart

```
Start
  ↓
What are you solving?
  ├─ Integral → Dimension?
  │    ├─ 1D → Smoothness?
  │    │    ├─ Smooth → gauss_legendre or romberg
  │    │    ├─ Singular → tanhsinh
  │    │    └─ Unknown → quad (adaptive)
  │    ├─ 2-3D → cubature or QMC
  │    └─ >3D → QMC or parallel Monte Carlo
  │
  ├─ ODE → Stiffness?
  │    ├─ Non-stiff → RK45 (default) or DOP853 (high precision)
  │    ├─ Stiff → BDF or Radau
  │    ├─ Unknown → LSODA (auto-switching)
  │    └─ Hamiltonian → Symplectic methods
  │
  ├─ DAE → Index and structure?
  │    ├─ Index-1 → BDF-DAE methods
  │    └─ Higher → Index reduction first
  │
  └─ PDE → Type and dimension?
       ├─ Elliptic → FEM or spectral
       ├─ Parabolic → Method of lines
       └─ Hyperbolic → Method of lines + explicit
```

## Benchmarking Your Choices

Always validate your method selection with benchmarks:

```rust
use std::time::Instant;

fn benchmark_methods() -> Result<(), Box<dyn std::error::Error>> {
    let methods = vec![
        ODEMethod::RK45,
        ODEMethod::BDF,
        ODEMethod::LSODA,
    ];
    
    for method in methods {
        let start = Instant::now();
        let result = solve_ivp(
            your_rhs_function,
            [0.0, 10.0],
            initial_conditions.clone(),
            Some(ODEOptions::new().method(method))
        )?;
        let duration = start.elapsed();
        
        println!("Method: {:?}, Time: {:?}, Evaluations: {}", 
                 method, duration, result.nfev);
    }
    
    Ok(())
}
```

Remember: **The best method is the one that gives you the required accuracy in the least time for your specific problem.**