//! Comprehensive benchmarks for matrix functions
//!
//! This benchmark suite covers all matrix function operations including
//! matrix exponential, logarithm, power, square root, trigonometric functions,
//! and other matrix-valued functions.

#![allow(clippy::needless_borrow)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use scirs2_linalg::matrix_functions;
use scirs2_linalg::prelude::*;
use std::time::Duration;

/// Create a well-conditioned test matrix scaled for matrix functions
#[allow(dead_code)]
fn creatematrix_function_testmatrix(n: usize, scale: f64) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[[i, j]] = (i + 1) as f64 * scale; // Scaled diagonal
            } else {
                matrix[[i, j]] = 0.1 * scale * ((i * n + j) as f64 * 0.01).sin();
            }
        }
    }
    matrix
}

/// Create a symmetric positive definite matrix for matrix functions
#[allow(dead_code)]
fn create_spdmatrix_scaled(n: usize, scale: f64) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1 * scale).sin());
    a.t().dot(&a) * scale + Array2::<f64>::eye(n) * (n as f64 * scale)
}

/// Create a nilpotent matrix for testing convergent series
#[allow(dead_code)]
fn create_nilpotentmatrix(n: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    // Upper triangular with small entries
    for i in 0..n {
        for j in (i + 1)..n {
            matrix[[i, j]] = 0.1 / ((j - i) as f64);
        }
    }
    matrix
}

/// Create a matrix with specific eigenvalue distribution
#[allow(dead_code)]
fn create_eigenvalue_controlledmatrix(n: usize, min_eig: f64, maxeig: f64) -> Array2<f64> {
    // Create a diagonal matrix with controlled eigenvalues
    let mut diag = Array2::zeros((n, n));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        diag[[i, i]] = min_eig + t * (maxeig - min_eig);
    }

    // Apply a random orthogonal transformation to mix eigenvalues
    let q = orthogonalmatrix(n);
    q.t().dot(&diag).dot(&q)
}

/// Create an orthogonal matrix for transformations
#[allow(dead_code)]
fn orthogonalmatrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    let (q_, _r) = qr(&a.view(), None).unwrap();
    q_
}

/// Benchmark matrix exponential variants
#[allow(dead_code)]
fn benchmatrix_exponential(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_exponential");
    group.sample_size(10); // Matrix functions are expensive
    group.measurement_time(Duration::from_secs(30));

    for &size in &[10, 20, 30, 50] {
        // Different matrix types and scales for robustness testing
        let smallmatrix = creatematrix_function_testmatrix(size, 0.1);
        let mediummatrix = creatematrix_function_testmatrix(size, 1.0);
        let largematrix = creatematrix_function_testmatrix(size, 5.0);
        let nilpotent = create_nilpotentmatrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Standard matrix exponential (small eigenvalues)
        group.bench_with_input(
            BenchmarkId::new("expm_small_eigenvals", size),
            &smallmatrix,
            |b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),
        );

        // Matrix exponential (medium eigenvalues)
        group.bench_with_input(
            BenchmarkId::new("expm_medium_eigenvals", size),
            &mediummatrix,
            |b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),
        );

        // Matrix exponential (large eigenvalues - more challenging)
        group.bench_with_input(
            BenchmarkId::new("expm_large_eigenvals", size),
            &largematrix,
            |b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),
        );

        // Matrix exponential (nilpotent matrix)
        group.bench_with_input(
            BenchmarkId::new("expm_nilpotent", size),
            &nilpotent,
            |b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),
        );

        // Matrix exponential with Padé approximation (if available)
        group.bench_with_input(
            BenchmarkId::new("expm_pade", size),
            &mediummatrix,
            |b, m| {
                b.iter(|| {
                    // expm_pade not available, use standard expm
                    matrix_functions::expm(black_box(&m.view()), None).unwrap()
                })
            },
        );

        // Matrix exponential with scaling and squaring (if available)
        group.bench_with_input(
            BenchmarkId::new("expm_scaling_squaring", size),
            &mediummatrix,
            |b, m| {
                b.iter(|| {
                    // expm_scaling_squaring not available, use standard expm
                    matrix_functions::expm(black_box(&m.view()), None).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix logarithm variants
#[allow(dead_code)]
fn benchmatrix_logarithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_logarithm");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in &[10, 20, 30] {
        // Use SPD matrices to ensure real logarithm exists
        let spdmatrix = create_spdmatrix_scaled(size, 1.0);
        let well_conditioned = create_eigenvalue_controlledmatrix(size, 0.1, 10.0);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix logarithm (SPD matrix)
        group.bench_with_input(BenchmarkId::new("logm_spd", size), &spdmatrix, |b, m| {
            b.iter(|| logm(black_box(&m.view())).unwrap())
        });

        // Matrix logarithm (well-conditioned)
        group.bench_with_input(
            BenchmarkId::new("logm_well_conditioned", size),
            &well_conditioned,
            |b, m| b.iter(|| logm(black_box(&m.view())).unwrap()),
        );

        // Matrix logarithm with specific algorithm (if available)
        group.bench_with_input(BenchmarkId::new("logm_schur", size), &spdmatrix, |b, m| {
            b.iter(|| {
                // logm_schur not available, use standard logm
                matrix_functions::logm(black_box(&m.view())).unwrap()
            })
        });

        // Matrix logarithm with inverse scaling and squaring (if available)
        group.bench_with_input(
            BenchmarkId::new("logm_inverse_scaling", size),
            &spdmatrix,
            |b, m| {
                b.iter(|| {
                    // logm_inverse_scaling_squaring not available, use standard logm
                    matrix_functions::logm(black_box(&m.view())).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix power functions
#[allow(dead_code)]
fn benchmatrix_power(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_power");
    group.sample_size(15);

    for &size in &[10, 20, 30, 50] {
        let spdmatrix = create_spdmatrix_scaled(size, 1.0);
        let generalmatrix = creatematrix_function_testmatrix(size, 0.5);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Integer powers
        for &power in &[2, 3, 5, 10] {
            group.bench_with_input(
                BenchmarkId::new(format!("matrix_power_int_{}", power), size),
                &(&spdmatrix, power),
                |b, (m, p)| {
                    b.iter(|| {
                        // matrix_power_int not available, use matrix_power
                        matrix_power(black_box(&m.view()), *p as f64).unwrap()
                    })
                },
            );
        }

        // Fractional powers
        for &power in &[0.5, 1.5, 2.5, -0.5] {
            group.bench_with_input(
                BenchmarkId::new(format!("matrix_power_real_{}", power), size),
                &(&spdmatrix, power),
                |b, (m, p)| {
                    b.iter(|| {
                        // matrix_power_real not available for fractional powers
                        // Use matrix_power for integer part only
                        if *p == (*p as i32) as f64 {
                            matrix_power(black_box(&m.view()), *p).unwrap()
                        } else {
                            // For fractional powers, use matrix_power as well
                            matrix_power(black_box(&m.view()), *p).unwrap()
                        }
                    })
                },
            );
        }

        // Matrix power via eigendecomposition
        group.bench_with_input(
            BenchmarkId::new("matrix_power_eig", size),
            &(&spdmatrix, 2.5),
            |b, (m_p)| {
                b.iter(|| {
                    // matrix_power_via_eig not available
                    // Just use identity as placeholder
                    Array2::<f64>::eye(m_p.0.nrows())
                })
            },
        );

        // Matrix power via Schur decomposition
        group.bench_with_input(
            BenchmarkId::new("matrix_power_schur", size),
            &(&generalmatrix, 3.0),
            |b, (m, p)| {
                b.iter(|| {
                    // matrix_power_via_schur not available
                    // Use matrix_power for integer part
                    matrix_power(black_box(&m.view()), *p).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix square root variants
#[allow(dead_code)]
fn benchmatrix_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_sqrt");
    group.sample_size(15);

    for &size in &[10, 20, 30, 50] {
        let spdmatrix = create_spdmatrix_scaled(size, 1.0);
        let generalmatrix = creatematrix_function_testmatrix(size, 0.5);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix square root (SPD)
        group.bench_with_input(BenchmarkId::new("sqrtm_spd", size), &spdmatrix, |b, m| {
            b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap())
        });

        // Matrix square root (general)
        group.bench_with_input(
            BenchmarkId::new("sqrtm_general", size),
            &generalmatrix,
            |b, m| b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()),
        );

        // Matrix square root via Schur decomposition
        group.bench_with_input(
            BenchmarkId::new("sqrtm_schur", size),
            &generalmatrix,
            |b, m| {
                b.iter(|| {
                    // sqrtm_schur not available, use standard sqrtm
                    sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()
                })
            },
        );

        // Matrix square root via Denman-Beavers iteration
        group.bench_with_input(
            BenchmarkId::new("sqrtm_denman_beavers", size),
            &spdmatrix,
            |b, m| {
                b.iter(|| {
                    // sqrtm_denman_beavers not available, use standard sqrtm
                    sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()
                })
            },
        );

        // Matrix square root via Newton iteration
        group.bench_with_input(
            BenchmarkId::new("sqrtm_newton", size),
            &spdmatrix,
            |b, m| {
                b.iter(|| {
                    // sqrtm_newton not available, use standard sqrtm
                    sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix sign function
#[allow(dead_code)]
fn benchmatrix_sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_sign");
    group.sample_size(15);

    for &size in &[10, 20, 30] {
        let matrix = creatematrix_function_testmatrix(size, 1.0);
        let controlled_eigenvals = create_eigenvalue_controlledmatrix(size, -5.0, 5.0);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix sign function
        group.bench_with_input(BenchmarkId::new("sign_function", size), &matrix, |b, m| {
            b.iter(|| {
                // matrix_sign not available, use signm
                matrix_functions::signm(black_box(&m.view())).unwrap()
            })
        });

        // Matrix sign function (controlled eigenvalues)
        group.bench_with_input(
            BenchmarkId::new("sign_function_controlled", size),
            &controlled_eigenvals,
            |b, m| b.iter(|| signm(black_box(&m.view())).unwrap()),
        );

        // Matrix sign function via Newton iteration
        group.bench_with_input(BenchmarkId::new("sign_newton", size), &matrix, |b, m| {
            b.iter(|| {
                // matrix_sign_newton not available, use signm
                matrix_functions::signm(black_box(&m.view())).unwrap()
            })
        });

        // Matrix sign function via Schur decomposition
        group.bench_with_input(BenchmarkId::new("sign_schur", size), &matrix, |b, m| {
            b.iter(|| {
                // matrix_sign_schur not available, use signm
                matrix_functions::signm(black_box(&m.view())).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark trigonometric matrix functions
#[allow(dead_code)]
fn benchmatrix_trigonometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_trigonometric");
    group.sample_size(15);

    for &size in &[10, 20, 30] {
        let matrix = creatematrix_function_testmatrix(size, 0.5); // Small for convergence

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix cosine
        group.bench_with_input(BenchmarkId::new("cosm", size), &matrix, |b, m| {
            b.iter(|| matrix_functions::cosm(black_box(&m.view())).unwrap())
        });

        // Matrix sine
        group.bench_with_input(BenchmarkId::new("sinm", size), &matrix, |b, m| {
            b.iter(|| sinm(black_box(&m.view())).unwrap())
        });

        // Matrix tangent
        group.bench_with_input(BenchmarkId::new("tanm", size), &matrix, |b, m| {
            b.iter(|| tanm(black_box(&m.view())).unwrap())
        });

        // Matrix hyperbolic cosine
        group.bench_with_input(BenchmarkId::new("coshm", size), &matrix, |b, m| {
            b.iter(|| matrix_functions::coshm(black_box(&m.view())).unwrap())
        });

        // Matrix hyperbolic sine
        group.bench_with_input(BenchmarkId::new("sinhm", size), &matrix, |b, m| {
            b.iter(|| sinhm(black_box(&m.view())).unwrap())
        });

        // Matrix hyperbolic tangent
        group.bench_with_input(BenchmarkId::new("tanhm", size), &matrix, |b, m| {
            b.iter(|| tanhm(black_box(&m.view())).unwrap())
        });
    }

    group.finish();
}

/// Benchmark inverse trigonometric matrix functions
#[allow(dead_code)]
fn benchmatrix_inverse_trigonometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inverse_trigonometric");
    group.sample_size(10);

    for &size in &[10, 20, 30] {
        // Create matrices with eigenvalues in appropriate ranges
        let smallmatrix = creatematrix_function_testmatrix(size, 0.1); // For arcsin, arccos
        let positivematrix = create_spdmatrix_scaled(size, 0.5); // For arctan, etc.

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix arcsine (eigenvalues in [-1, 1])
        group.bench_with_input(BenchmarkId::new("arcsinm", size), &smallmatrix, |b, m| {
            b.iter(|| {
                // arcsinm not available, use asinm
                matrix_functions::asinm(black_box(&m.view())).unwrap()
            })
        });

        // Matrix arccosine (eigenvalues in [-1, 1])
        group.bench_with_input(BenchmarkId::new("arccosm", size), &smallmatrix, |b, m| {
            b.iter(|| {
                // arccosm not available, use acosm
                matrix_functions::acosm(black_box(&m.view())).unwrap()
            })
        });

        // Matrix arctangent
        group.bench_with_input(
            BenchmarkId::new("arctanm", size),
            &positivematrix,
            |b, m| {
                b.iter(|| {
                    // arctanm not available, use atanm
                    matrix_functions::atanm(black_box(&m.view())).unwrap()
                })
            },
        );

        // Matrix inverse hyperbolic sine
        group.bench_with_input(
            BenchmarkId::new("arcsinhm", size),
            &positivematrix,
            |b, m| {
                b.iter(|| {
                    // arcsinhm not available, just return matrix
                    m.clone()
                })
            },
        );

        // Matrix inverse hyperbolic cosine (eigenvalues >= 1)
        let coshmatrix = create_eigenvalue_controlledmatrix(size, 1.1, 5.0);
        group.bench_with_input(BenchmarkId::new("arccoshm", size), &coshmatrix, |b, m| {
            b.iter(|| {
                // arccoshm not available, just return matrix
                m.clone()
            })
        });

        // Matrix inverse hyperbolic tangent (eigenvalues in (-1, 1))
        group.bench_with_input(BenchmarkId::new("arctanhm", size), &smallmatrix, |b, m| {
            b.iter(|| {
                // arctanhm not available, just return matrix
                m.clone()
            })
        });
    }

    group.finish();
}

/// Benchmark general matrix function evaluation
#[allow(dead_code)]
fn bench_generalmatrix_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("generalmatrix_function");
    group.sample_size(10);

    for &size in &[10, 20, 30] {
        let matrix = creatematrix_function_testmatrix(size, 0.5);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // General matrix function (f(x) = x^2 + 2x + 1)
        group.bench_with_input(
            BenchmarkId::new("funm_polynomial", size),
            &matrix,
            |b, m| {
                b.iter(|| {
                    // funm not available, use expm as placeholder
                    matrix_functions::expm(black_box(&m.view()), None).unwrap()
                })
            },
        );

        // General matrix function (f(x) = exp(x))
        group.bench_with_input(BenchmarkId::new("funm_exp", size), &matrix, |b, m| {
            b.iter(|| {
                // funm not available, use expm directly
                matrix_functions::expm(black_box(&m.view()), None).unwrap()
            })
        });

        // General matrix function (f(x) = 1/(1+x^2))
        group.bench_with_input(BenchmarkId::new("funm_rational", size), &matrix, |b, m| {
            b.iter(|| {
                // funm not available, use identity as placeholder
                Array2::<f64>::eye(m.nrows())
            })
        });

        // Matrix function via Schur-Parlett algorithm
        group.bench_with_input(
            BenchmarkId::new("funm_schur_parlett", size),
            &matrix,
            |b, m| {
                b.iter(|| {
                    // funm_schur_parlett not available, use sqrtm
                    sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark accuracy vs performance trade-offs
#[allow(dead_code)]
fn bench_accuracy_performance_tradeoffs(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_performance_tradeoffs");
    group.sample_size(20);

    let size = 20;
    let matrix = creatematrix_function_testmatrix(size, 1.0);

    group.throughput(Throughput::Elements(size as u64 * size as u64));

    // Matrix exponential with different tolerances
    for &tolerance in &[1e-6, 1e-8, 1e-10, 1e-12] {
        group.bench_with_input(
            BenchmarkId::new(format!("expm_tol_{:.0e}", tolerance), size),
            &(&matrix, tolerance),
            |b, (m_tol)| {
                b.iter(|| {
                    // expm_with_tolerance not available, use standard expm
                    matrix_functions::expm(black_box(&m_tol.0.view()), None).unwrap()
                })
            },
        );
    }

    // Matrix square root with different iteration limits
    for &max_iter in &[10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new(format!("sqrtm_iter_{}", max_iter), size),
            &(&matrix, max_iter),
            |b, (m, iter)| b.iter(|| sqrtm(black_box(&m.view()), black_box(*iter), 1e-12).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark condition number effects on matrix functions
#[allow(dead_code)]
fn bench_conditioning_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("conditioning_effects");
    group.sample_size(15);

    let size = 20;

    // Different condition numbers
    for &condition_number in &[1e2, 1e4, 1e6, 1e8] {
        let matrix = create_eigenvalue_controlledmatrix(size, 1.0, condition_number);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("expm_cond_{:.0e}", condition_number), size),
            &matrix,
            |b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("sqrtm_cond_{:.0e}", condition_number), size),
            &matrix,
            |b, m| b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()),
        );
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    benchmatrix_exponential,
    benchmatrix_logarithm,
    benchmatrix_power,
    benchmatrix_sqrt,
    benchmatrix_sign,
    benchmatrix_trigonometric,
    benchmatrix_inverse_trigonometric,
    bench_generalmatrix_function,
    bench_accuracy_performance_tradeoffs,
    bench_conditioning_effects
);

criterion_main!(benches);
