//! Simple Array Operation Benchmarks
//!
//! Basic benchmarks for array operations to validate performance.

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::hint::black_box;

#[allow(dead_code)]
fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");

    group.bench_function("zeros_1000", |b| {
        b.iter(|| {
            let arr = Array1::<f64>::zeros(1000);
            black_box(arr)
        })
    });

    group.bench_function("ones_1000", |b| {
        b.iter(|| {
            let arr = Array1::<f64>::ones(1000);
            black_box(arr)
        })
    });

    group.bench_function("random_1000", |b| {
        b.iter(|| {
            let arr = Array1::<f64>::random(1000, Uniform::new(0.0, 1.0));
            black_box(arr)
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");

    let arr1 = Array1::<f64>::random(1000, Uniform::new(0.0, 1.0));
    let arr2 = Array1::<f64>::random(1000, Uniform::new(0.0, 1.0));

    group.bench_function("add_1000", |b| {
        b.iter(|| {
            let result = &arr1 + &arr2;
            black_box(result)
        })
    });

    group.bench_function("multiply_1000", |b| {
        b.iter(|| {
            let result = &arr1 * &arr2;
            black_box(result)
        })
    });

    group.bench_function("sum_1000", |b| {
        b.iter(|| {
            let result = arr1.sum();
            black_box(result)
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn benchmatrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    let mat1 = Array2::<f64>::random((100, 100), Uniform::new(0.0, 1.0));
    let mat2 = Array2::<f64>::random((100, 100), Uniform::new(0.0, 1.0));

    group.bench_function("transpose_100x100", |b| {
        b.iter(|| {
            let result = mat1.t().to_owned();
            black_box(result)
        })
    });

    group.bench_function("element_multiply_100x100", |b| {
        b.iter(|| {
            let result = &mat1 * &mat2;
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_array_operations,
    benchmatrix_operations
);
criterion_main!(benches);
