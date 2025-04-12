# scirs2-integrate TODO

This module provides numerical integration functionality similar to SciPy's integrate module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Numerical quadrature
  - [x] Basic methods (trapezoid rule, Simpson's rule)
  - [x] Adaptive quadrature for improved accuracy
  - [x] Gaussian quadrature
  - [x] Romberg integration
  - [x] Monte Carlo integration
  - [x] Multiple integration (for higher dimensions)
- [x] Ordinary differential equations (ODE)
  - [x] Euler method
  - [x] Runge-Kutta methods (RK4)
  - [x] Support for first-order ODE systems
- [x] Example code for all integration methods

## Future Tasks

- [x] Fix implementation issues
  - [x] Fix Gaussian quadrature node/weight calculation (current implementation had scaling issues)
  - [x] Improve Monte Carlo importance sampling stability
  - [x] Handle deep recursion issues better in Romberg integration
- [ ] Enhance ODE solvers
  - [x] Variable step-size methods foundations (RK45, RK23)
  - [x] Explicit and implicit methods foundation (BDF implementation)
  - [x] Stiff equation solvers foundation (BDF)
  - [ ] Boundary value problems
  - [ ] Fix critical implementation issues with variable step and implicit methods:
    - [ ] Fix coefficient calculations for RK23 
    - [ ] Fix error estimation and step acceptance logic for RK45
    - [ ] Revise BDF implementation with more stable numerical method for Newton iterations
    - [ ] Add auto-differentiation or better numerical Jacobian calculation for BDF
    - [ ] Enable currently ignored tests once fixed
- [ ] Implement differential algebraic equation (DAE) solvers
- [ ] Add support for partial differential equations (PDE)
  - [ ] Finite difference methods
  - [ ] Finite element methods
- [ ] Improve error handling and convergence criteria
- [ ] Add more examples and documentation
  - [ ] Tutorial for common integration problems
  - [ ] Examples for physical system modeling

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's integrate
- [ ] Support for parallel and distributed computation
- [ ] Integration with automatic differentiation for gradient-based methods
- [ ] Support for symbolic manipulation and simplification
- [ ] Advanced visualization tools for solutions
- [ ] Domain-specific solvers for physics, engineering, and finance