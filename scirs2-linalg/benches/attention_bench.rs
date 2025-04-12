#[macro_use]
extern crate criterion;

use criterion::{black_box, BenchmarkId, Criterion};
use ndarray::{Array2, Array3, ShapeBuilder};
use scirs2_linalg::attention::{
    causal_attention, flash_attention, linear_attention, multi_head_attention,
    scaled_dot_product_attention, AttentionConfig,
};

fn attention_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");

    let batch_size = 8;
    let seq_lens = vec![16, 32, 64, 128]; // Various sequence lengths to benchmark
    let d_model = 64;

    for &seq_len in seq_lens.iter() {
        // Prepare tensors
        let query = Array3::<f32>::ones((batch_size, seq_len, d_model));
        let key = Array3::<f32>::ones((batch_size, seq_len, d_model));
        let value = Array3::<f32>::ones((batch_size, seq_len, d_model));

        // Scaled dot-product attention
        group.bench_with_input(
            BenchmarkId::new("scaled_dot_product", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let scale = 1.0 / f32::sqrt(d_model as f32);
                    black_box(
                        scaled_dot_product_attention(
                            &query.view(),
                            &key.view(),
                            &value.view(),
                            None,
                            scale,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // Causal attention (for autoregressive models)
        group.bench_with_input(BenchmarkId::new("causal", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let scale = 1.0 / f32::sqrt(d_model as f32);
                black_box(
                    causal_attention(&query.view(), &key.view(), &value.view(), scale).unwrap(),
                )
            })
        });

        // Flash attention
        group.bench_with_input(BenchmarkId::new("flash", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let scale = 1.0 / f32::sqrt(d_model as f32);
                black_box(
                    flash_attention(
                        &query.view(),
                        &key.view(),
                        &value.view(),
                        None,
                        scale,
                        16, // Block size
                    )
                    .unwrap(),
                )
            })
        });

        // Linear attention
        group.bench_with_input(BenchmarkId::new("linear", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let scale = 1.0 / f32::sqrt(d_model as f32);
                black_box(
                    linear_attention(&query.view(), &key.view(), &value.view(), scale).unwrap(),
                )
            })
        });

        // Multi-head attention
        group.bench_with_input(BenchmarkId::new("multi_head", seq_len), &seq_len, |b, _| {
            // Linear projection weights
            let num_heads = 8;
            let head_dim = d_model / num_heads;
            let wq = Array2::<f32>::ones((d_model, d_model));
            let wk = Array2::<f32>::ones((d_model, d_model));
            let wv = Array2::<f32>::ones((d_model, d_model));
            let wo = Array2::<f32>::ones((d_model, d_model));

            let config = AttentionConfig {
                num_heads,
                head_dim,
                dropout_prob: 0.0,
                causal: false,
                scale: Some(1.0 / f32::sqrt(head_dim as f32)),
            };

            b.iter(|| {
                black_box(
                    multi_head_attention(
                        &query.view(),
                        &key.view(),
                        &value.view(),
                        &wq.view(),
                        &wk.view(),
                        &wv.view(),
                        &wo.view(),
                        None,
                        &config,
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = attention_benchmark
);
criterion_main!(benches);
