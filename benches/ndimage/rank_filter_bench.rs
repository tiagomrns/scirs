use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use scirs2_ndimage::{maximum_filter, minimum_filter, percentile_filter, rank_filter};

fn rank_filter_1d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rank Filter 1D");

    // Create a larger 1D array for benchmarking
    let n = 1000;
    let array = Array1::from_iter((0..n).map(|i| (i % 10) as f64));

    // Benchmark different rank filters
    group.bench_function("minimum_filter", |b| {
        b.iter(|| minimum_filter(black_box(&array), black_box(&[5]), None))
    });

    group.bench_function("maximum_filter", |b| {
        b.iter(|| maximum_filter(black_box(&array), black_box(&[5]), None))
    });

    group.bench_function("median_filter", |b| {
        b.iter(|| percentile_filter(black_box(&array), black_box(50.0), black_box(&[5]), None))
    });

    group.bench_function("rank_filter_25th", |b| {
        b.iter(|| rank_filter(black_box(&array), black_box(1), black_box(&[5]), None))
    });

    group.finish();
}

fn rank_filter_2d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rank Filter 2D");

    // Create a larger 2D array for benchmarking
    let n = 100;
    let array = Array2::from_shape_fn((n, n), |(i, j)| ((i + j) % 10) as f64);

    // Benchmark different rank filters
    group.bench_function("minimum_filter", |b| {
        b.iter(|| minimum_filter(black_box(&array), black_box(&[5, 5]), None))
    });

    group.bench_function("maximum_filter", |b| {
        b.iter(|| maximum_filter(black_box(&array), black_box(&[5, 5]), None))
    });

    group.bench_function("median_filter", |b| {
        b.iter(|| percentile_filter(black_box(&array), black_box(50.0), black_box(&[5, 5]), None))
    });

    group.bench_function("rank_filter_25th", |b| {
        b.iter(|| {
            // For a 5x5 window (25 elements), rank 6 is ~25th percentile
            rank_filter(black_box(&array), black_box(6), black_box(&[5, 5]), None)
        })
    });

    group.finish();
}

criterion_group!(benches, rank_filter_1d_benchmark, rank_filter_2d_benchmark);
criterion_main!(benches);
