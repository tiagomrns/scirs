use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Array3, Array4};
use rand::Rng;
use scirs2_neural::*;

fn generate_random_data(shape: &[usize]) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let total_size = shape.iter().product();
    Array2::from_shape_vec(
        (shape[0], shape[1]),
        (0..total_size).map(|_| rng.gen::<f32>()).collect(),
    )
    .unwrap()
}

fn generate_random_3d(shape: &[usize]) -> Array3<f32> {
    let mut rng = rand::thread_rng();
    let total_size = shape.iter().product();
    Array3::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        (0..total_size).map(|_| rng.gen::<f32>()).collect(),
    )
    .unwrap()
}

fn generate_random_4d(shape: &[usize]) -> Array4<f32> {
    let mut rng = rand::thread_rng();
    let total_size = shape.iter().product();
    Array4::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        (0..total_size).map(|_| rng.gen::<f32>()).collect(),
    )
    .unwrap()
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    for size in [1000, 10000, 100000].iter() {
        let data = Array1::from_iter((0..*size).map(|i| (i as f32 - *size as f32 / 2.0) / 1000.0));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("relu", size), size, |b, _| {
            b.iter(|| activations::relu(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |b, _| {
            b.iter(|| activations::sigmoid(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("tanh", size), size, |b, _| {
            b.iter(|| activations::tanh(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("leaky_relu", size), size, |b, _| {
            b.iter(|| activations::leaky_relu(black_box(&data), 0.01))
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), size, |b, _| {
            b.iter(|| activations::gelu(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("swish", size), size, |b, _| {
            b.iter(|| activations::swish(black_box(&data)))
        });
    }

    group.finish();
}

fn bench_dense_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_layers");

    for (batch_size, input_size, output_size) in
        [(32, 784, 128), (64, 512, 256), (128, 256, 64)].iter()
    {
        let input = generate_random_data(&[*batch_size, *input_size]);
        let weights = generate_random_data(&[*input_size, *output_size]);
        let bias = Array1::from_iter((0..*output_size).map(|i| i as f32 / 100.0));

        let mut layer = layers::Dense::new(*input_size, *output_size);
        layer.weights = weights.clone();
        layer.bias = bias.clone();

        group.throughput(Throughput::Elements(
            (*batch_size * *input_size * *output_size) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new(
                "dense_forward",
                format!("{}x{}x{}", batch_size, input_size, output_size),
            ),
            &(batch_size, input_size, output_size),
            |b, _| b.iter(|| layer.forward(black_box(&input))),
        );

        // Backward pass benchmark
        let output_grad = generate_random_data(&[*batch_size, *output_size]);

        group.bench_with_input(
            BenchmarkId::new(
                "dense_backward",
                format!("{}x{}x{}", batch_size, input_size, output_size),
            ),
            &(batch_size, input_size, output_size),
            |b, _| b.iter(|| layer.backward(black_box(&input), black_box(&output_grad))),
        );
    }

    group.finish();
}

fn bench_convolutional_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_layers");

    for (batch_size, channels, height, width, filters) in
        [(8, 3, 32, 32, 16), (16, 32, 16, 16, 64)].iter()
    {
        let input = generate_random_4d(&[*batch_size, *channels, *height, *width]);
        let kernel_size = 3;
        let stride = 1;
        let padding = 1;

        let mut conv_layer = layers::Conv2D::new(*channels, *filters, kernel_size, stride, padding);

        group.throughput(Throughput::Elements(
            (*batch_size * *channels * *height * *width) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new(
                "conv2d_forward",
                format!("{}x{}x{}x{}", batch_size, channels, height, width),
            ),
            &(batch_size, channels, height, width),
            |b, _| b.iter(|| conv_layer.forward(black_box(&input))),
        );
    }

    group.finish();
}

fn bench_recurrent_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_layers");

    for (batch_size, seq_len, input_size, hidden_size) in
        [(32, 50, 128, 256), (64, 25, 256, 128)].iter()
    {
        let input = generate_random_3d(&[*batch_size, *seq_len, *input_size]);

        let mut lstm_layer = layers::LSTM::new(*input_size, *hidden_size);
        let mut gru_layer = layers::GRU::new(*input_size, *hidden_size);

        group.throughput(Throughput::Elements(
            (*batch_size * *seq_len * *input_size) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new(
                "lstm_forward",
                format!("{}x{}x{}", batch_size, seq_len, input_size),
            ),
            &(batch_size, seq_len, input_size),
            |b, _| b.iter(|| lstm_layer.forward(black_box(&input))),
        );

        group.bench_with_input(
            BenchmarkId::new(
                "gru_forward",
                format!("{}x{}x{}", batch_size, seq_len, input_size),
            ),
            &(batch_size, seq_len, input_size),
            |b, _| b.iter(|| gru_layer.forward(black_box(&input))),
        );
    }

    group.finish();
}

fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");

    for size in [1000, 10000, 100000].iter() {
        let predictions = Array1::from_iter((0..*size).map(|i| (i as f32) / (*size as f32)));
        let targets = Array1::from_iter((0..*size).map(|i| ((i + 100) as f32) / (*size as f32)));

        // For classification
        let num_classes = 10;
        let pred_probs = generate_random_data(&[*size / num_classes, num_classes]);
        let true_labels =
            Array1::from_iter((0..(*size / num_classes)).map(|i| (i % num_classes) as f32));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("mse_loss", size), size, |b, _| {
            b.iter(|| losses::mse_loss(black_box(&predictions), black_box(&targets)))
        });

        group.bench_with_input(BenchmarkId::new("mae_loss", size), size, |b, _| {
            b.iter(|| losses::mae_loss(black_box(&predictions), black_box(&targets)))
        });

        group.bench_with_input(
            BenchmarkId::new("cross_entropy_loss", size),
            size,
            |b, _| {
                b.iter(|| {
                    losses::cross_entropy_loss(black_box(&pred_probs), black_box(&true_labels))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_cross_entropy", size),
            size,
            |b, _| {
                b.iter(|| {
                    losses::binary_cross_entropy_loss(black_box(&predictions), black_box(&targets))
                })
            },
        );
    }

    group.finish();
}

fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");

    for param_size in [1000, 10000, 100000].iter() {
        let params = Array1::from_iter((0..*param_size).map(|i| i as f32 / 1000.0));
        let gradients = Array1::from_iter(
            (0..*param_size).map(|i| (i as f32 - *param_size as f32 / 2.0) / 10000.0),
        );

        let mut sgd = optimizers::SGD::new(0.01);
        let mut adam = optimizers::Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut rmsprop = optimizers::RMSprop::new(0.001, 0.9, 1e-8);

        group.throughput(Throughput::Elements(*param_size as u64));

        group.bench_with_input(
            BenchmarkId::new("sgd_update", param_size),
            param_size,
            |b, _| b.iter(|| sgd.update(black_box(&params), black_box(&gradients))),
        );

        group.bench_with_input(
            BenchmarkId::new("adam_update", param_size),
            param_size,
            |b, _| b.iter(|| adam.update(black_box(&params), black_box(&gradients))),
        );

        group.bench_with_input(
            BenchmarkId::new("rmsprop_update", param_size),
            param_size,
            |b, _| b.iter(|| rmsprop.update(black_box(&params), black_box(&gradients))),
        );
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for (batch_size, feature_size) in [(32, 128), (64, 256), (128, 512)].iter() {
        let data = generate_random_data(&[*batch_size, *feature_size]);

        group.throughput(Throughput::Elements((*batch_size * *feature_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_norm", format!("{}x{}", batch_size, feature_size)),
            &(batch_size, feature_size),
            |b, _| b.iter(|| layers::batch_norm(black_box(&data), None, None, 1e-5)),
        );

        group.bench_with_input(
            BenchmarkId::new("layer_norm", format!("{}x{}", batch_size, feature_size)),
            &(batch_size, feature_size),
            |b, _| b.iter(|| layers::layer_norm(black_box(&data), None, None, 1e-5)),
        );

        group.bench_with_input(
            BenchmarkId::new("dropout", format!("{}x{}", batch_size, feature_size)),
            &(batch_size, feature_size),
            |b, _| b.iter(|| layers::dropout(black_box(&data), 0.5, true)),
        );
    }

    group.finish();
}

fn bench_model_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_training");
    group.sample_size(10); // Reduced sample size for expensive operations

    for (batch_size, input_size, hidden_size, output_size) in
        [(32, 784, 128, 10), (64, 512, 256, 20)].iter()
    {
        let mut model = models::MLP::new(&[*input_size, *hidden_size, *hidden_size, *output_size]);
        let input = generate_random_data(&[*batch_size, *input_size]);
        let targets = Array1::from_iter((0..*batch_size).map(|i| (i % *output_size) as f32));

        group.throughput(Throughput::Elements((*batch_size * *input_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("forward_pass", format!("{}x{}", batch_size, input_size)),
            &(batch_size, input_size),
            |b, _| b.iter(|| model.forward(black_box(&input))),
        );

        group.bench_with_input(
            BenchmarkId::new("training_step", format!("{}x{}", batch_size, input_size)),
            &(batch_size, input_size),
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(black_box(&input));
                    let loss = losses::cross_entropy_loss(&predictions, black_box(&targets));
                    model.backward(&predictions, black_box(&targets));
                    black_box(loss)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_activation_functions,
    bench_dense_layers,
    bench_convolutional_layers,
    bench_recurrent_layers,
    bench_loss_functions,
    bench_optimizers,
    bench_batch_operations,
    bench_model_training
);
criterion_main!(benches);
