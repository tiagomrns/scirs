//! Benchmarks for linear algebra functions

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::array;
use scirs2_linalg::blas::{dot, nrm2};

pub fn dot_benchmark(c: &mut Criterion) {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
    c.bench_function("dot product 5D", |b| {
        b.iter(|| dot(black_box(&x.view()), black_box(&y.view())))
    });
}

pub fn norm_benchmark(c: &mut Criterion) {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    c.bench_function("2-norm 5D", |b| b.iter(|| nrm2(black_box(&x.view()))));
}

criterion_group!(benches, dot_benchmark, norm_benchmark);
criterion_main!(benches);
