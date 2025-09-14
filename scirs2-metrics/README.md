# SciRS2 Metrics

[![crates.io](https://img.shields.io/crates/v/scirs2-metrics.svg)](https://crates.io/crates/scirs2-metrics)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-metrics)](https://docs.rs/scirs2-metrics)

Production-ready machine learning evaluation metrics library for the SciRS2 scientific computing ecosystem. This comprehensive module provides high-performance functions to evaluate prediction performance across classification, regression, clustering, and specialized ML domains.

## Features

### Core Metrics
- **Classification Metrics**: Complete suite including accuracy, precision, recall, F1-score, ROC curves, AUC, Matthews correlation coefficient, Cohen's kappa, and specialized multi-class/multi-label metrics
- **Regression Metrics**: Comprehensive set including MSE, MAE, R² score, explained variance, Huber loss, quantile loss, and robust regression metrics
- **Clustering Metrics**: Full coverage of internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and external metrics (Adjusted Rand Index, Normalized Mutual Information, V-measure)
- **Ranking Metrics**: Advanced ranking evaluation with NDCG, MAP, MRR, and specialized retrieval metrics

### Specialized Domains
- **Anomaly Detection**: Comprehensive anomaly evaluation with distribution metrics, time-series anomaly detection, and NAB scoring
- **Fairness & Bias Detection**: Advanced fairness metrics, demographic parity, equalized odds, and bias analysis tools
- **Computer Vision**: Object detection, image segmentation, and classification-specific metrics
- **Natural Language Processing**: Text generation, classification, NER, and sentiment analysis metrics
- **Time Series**: Forecasting accuracy, trend analysis, and autocorrelation metrics
- **Recommender Systems**: Ranking, diversity, novelty, and rating prediction metrics

### Advanced Capabilities
- **Bayesian Evaluation**: Model comparison with Bayes factors, information criteria (BIC, WAIC, LOO-CV), and posterior predictive checks
- **Statistical Testing**: McNemar's test, Friedman test, bootstrap confidence intervals, and cross-validation frameworks
- **Hardware Acceleration**: SIMD-optimized computations with automatic fallbacks for maximum performance
- **Streaming Metrics**: Memory-efficient online evaluation for large-scale and real-time applications
- **Custom Metrics**: Extensible framework for domain-specific metric development

### Integration & Visualization
- **Seamless Integration**: First-class support for scirs2-neural and scirs2-optim with metric-based optimization
- **Rich Visualizations**: Interactive ROC curves, confusion matrices, calibration plots, and customizable dashboards
- **Multiple Backends**: Support for Plotters and Plotly rendering engines
- **Export Capabilities**: JSON, CSV, HTML, and image format exports for reports and presentations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-metrics = "0.1.0"
```

### Feature Selection

Choose features based on your needs:

```toml
[dependencies]
# Default features (recommended for most users)
scirs2-metrics = "0.1.0"

# Minimal installation (core metrics only)
scirs2-metrics = { version = "0.1.0", default-features = false }

# Full installation with all capabilities
scirs2-metrics = { version = "0.1.0", features = ["neural_common", "optim_integration", "plotters_backend"] }
```

#### Available Features

- `plotly_backend` (default): Interactive web-based visualizations with Plotly
- `optim_integration` (default): Integration with scirs2-optim for metric-based optimization
- `neural_common`: Integration utilities for neural network training
- `plotters_backend`: Static plot generation with Plotters backend
- `dashboard_server`: HTTP server for real-time dashboard (requires tokio)

## Usage

Basic usage examples:

```rust
use scirs2_metrics::{classification, regression, clustering};
use ndarray::array;
use scirs2_core::error::CoreResult;

// Classification metrics example
fn classification_metrics_example() -> CoreResult<()> {
    // True labels
    let y_true = array![0, 1, 0, 1, 0, 1, 0, 1];
    
    // Predicted labels
    let y_pred = array![0, 1, 1, 1, 0, 0, 0, 1];
    
    // Calculate classification metrics
    let accuracy = classification::accuracy_score(&y_true, &y_pred)?;
    let precision = classification::precision_score(&y_true, &y_pred, None, None, None)?;
    let recall = classification::recall_score(&y_true, &y_pred, None, None, None)?;
    let f1 = classification::f1_score(&y_true, &y_pred, None, None, None)?;
    
    println!("Accuracy: {}", accuracy);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
    println!("F1 Score: {}", f1);
    
    // Predicted probabilities for ROC curve
    let y_scores = array![0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.4, 0.8];
    
    // Calculate ROC curve
    let (fpr, tpr, thresholds) = classification::roc_curve(&y_true, &y_scores, None, None)?;
    
    // Calculate Area Under the ROC Curve (AUC)
    let auc = classification::roc_auc_score(&y_true, &y_scores)?;
    println!("AUC: {}", auc);
    
    Ok(())
}

// Regression metrics example
fn regression_metrics_example() -> CoreResult<()> {
    // True values
    let y_true = array![3.0, -0.5, 2.0, 7.0, 2.0];
    
    // Predicted values
    let y_pred = array![2.5, 0.0, 2.1, 7.8, 1.8];
    
    // Calculate regression metrics
    let mse = regression::mean_squared_error(&y_true, &y_pred, None)?;
    let mae = regression::mean_absolute_error(&y_true, &y_pred, None)?;
    let r2 = regression::r2_score(&y_true, &y_pred, None)?;
    let explained_variance = regression::explained_variance_score(&y_true, &y_pred, None)?;
    
    println!("Mean Squared Error: {}", mse);
    println!("Mean Absolute Error: {}", mae);
    println!("R² Score: {}", r2);
    println!("Explained Variance: {}", explained_variance);
    
    Ok(())
}

// Clustering metrics example
fn clustering_metrics_example() -> CoreResult<()> {
    // Sample data points
    let data = array![
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ];
    
    // Predicted cluster labels from algorithm
    let pred_labels = array![0, 0, 1, 1, 0, 1];
    
    // Calculate internal clustering metrics (no ground truth needed)
    let silhouette = clustering::silhouette_score(&data, &pred_labels, None, None)?;
    let calinski_harabasz = clustering::calinski_harabasz_score(&data, &pred_labels)?;
    let davies_bouldin = clustering::davies_bouldin_score(&data, &pred_labels)?;
    
    println!("Internal Metrics (evaluate clustering without ground truth):");
    println!("Silhouette Score: {}", silhouette);
    println!("Calinski-Harabasz Index: {}", calinski_harabasz);
    println!("Davies-Bouldin Index: {}", davies_bouldin);
    
    // For external metrics, we need ground truth labels
    let true_labels = array![0, 0, 1, 1, 0, 2];  // True class assignments
    
    // Calculate external clustering metrics (comparing to ground truth)
    let ari = clustering::adjusted_rand_index(&true_labels, &pred_labels)?;
    let nmi = clustering::normalized_mutual_info_score(&true_labels, &pred_labels, "arithmetic")?;
    let (homogeneity, completeness, v_measure) = 
        clustering::homogeneity_completeness_v_measure(&true_labels, &pred_labels, 1.0)?;
    
    println!("\nExternal Metrics (compare to ground truth):");
    println!("Adjusted Rand Index: {}", ari);
    println!("Normalized Mutual Information: {}", nmi);
    println!("Homogeneity: {}", homogeneity);
    println!("Completeness: {}", completeness);
    println!("V-measure: {}", v_measure);
    
    Ok(())
}
```

## Visualization

The visualization module provides utilities for creating informative visualizations of metrics results:

```rust
use ndarray::array;
use scirs2_metrics::{
    classification::{confusion_matrix, accuracy_score},
    classification::curves::{roc_curve, precision_recall_curve},
    visualization::{
        VisualizationData, VisualizationMetadata, PlotType, ColorMap, VisualizationOptions, 
        backends, helpers
    },
};

// Example: ROC Curve Visualization
fn roc_curve_visualization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create binary classification data
    let y_true = array![0, 1, 1, 0, 1, 0, 1, 0, 1];
    let y_score = array![0.1, 0.8, 0.7, 0.3, 0.9, 0.2, 0.6, 0.3, 0.8];
    
    // Compute ROC curve
    let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score)?;
    
    // Create ROC curve visualizer
    let roc_viz = helpers::visualize_roc_curve(
        fpr.view(),
        tpr.view(),
        Some(thresholds.view()),
        Some(0.85), // Example AUC value
    );
    
    // Use builder functions for visualization options
    let options = VisualizationOptions::new()
        .with_width(800)
        .with_height(600)
        .with_dpi(150)
        .with_line_width(2.0)
        .with_grid(true)
        .with_legend(true);
    
    // Save to file (requires plotters_backend or plotly_backend feature)
    roc_viz.save_to_file("roc_curve.png", Some(options))?;
    
    Ok(())
}

// Example: Confusion Matrix Visualization
fn confusion_matrix_visualization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create classification data
    let y_true = array![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let y_pred = array![0, 2, 1, 0, 0, 2, 1, 1, 2];
    
    // Compute confusion matrix
    let (cm, _) = confusion_matrix(&y_true, &y_pred, None)?;
    let cm_f64 = cm.mapv(|x| x as f64);
    
    // Create confusion matrix visualizer
    let cm_viz = helpers::visualize_confusion_matrix(
        cm_f64.view(),
        Some(vec!["Class 0".to_string(), "Class 1".to_string(), "Class 2".to_string()]),
        false, // Don't normalize
    );
    
    // Save to file with custom options
    let options = VisualizationOptions::new()
        .with_color_map(ColorMap::BlueRed)
        .with_colorbar(true);
    
    cm_viz.save_to_file("confusion_matrix.png", Some(options))?;
    
    Ok(())
}

// Example: Custom visualization creation
fn custom_visualization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom data
    let epochs = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let loss = array![1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23];
    
    // Create a metric visualizer
    let visualizer = helpers::visualize_metric(
        epochs.view(),
        loss.view(),
        "Training Loss",
        "Epoch",
        "Loss",
        PlotType::Line,
    );
    
    // Get the data and metadata
    let data = visualizer.prepare_data()?;
    let metadata = visualizer.get_metadata();
    
    // Render to SVG or PNG
    let svg_data = visualizer.render_svg(None)?;
    let png_data = visualizer.render_png(None)?;
    
    // Or save directly to file
    visualizer.save_to_file("training_loss.png", None)?;
    
    Ok(())
}
```

### Available Visualization Types

The visualization module supports multiple plot types:

- **Line Plots**: For time series, learning curves, loss curves, etc.
- **Scatter Plots**: For correlation, dimensionality reduction results, etc.
- **Bar Charts**: For feature importance, classification reports, etc.
- **Heatmaps**: For confusion matrices, correlation matrices, etc.
- **Histograms**: For value distributions, error distributions, etc.

### Interactive Dashboard Server

When the `dashboard_server` feature is enabled, you can create a real-time web dashboard for monitoring metrics:

```rust
use scirs2_metrics::dashboard::{InteractiveDashboard, DashboardConfig};
#[cfg(feature = "dashboard_server")]
use scirs2_metrics::dashboard::server::start_http_server;

// Create dashboard configuration
let mut config = DashboardConfig::default();
config.title = "ML Training Dashboard".to_string();
config.refresh_interval = 2; // Refresh every 2 seconds

// Create interactive dashboard
let dashboard = InteractiveDashboard::new(config);

// Add metrics
dashboard.add_metric("accuracy", 0.95)?;
dashboard.add_metric("loss", 0.12)?;
dashboard.add_metric("learning_rate", 0.001)?;

// Start HTTP server (requires dashboard_server feature)
#[cfg(feature = "dashboard_server")]
{
    let server = start_http_server(dashboard.clone())?;
    println!("Dashboard available at http://127.0.0.1:8080");
    
    // Update metrics in your training loop
    for epoch in 0..100 {
        dashboard.add_metric("epoch", epoch as f64)?;
        dashboard.add_metric("accuracy", accuracy)?;
        dashboard.add_metric("loss", loss)?;
        // ... training code ...
    }
}

// Export dashboard data
let json_data = dashboard.export_to_json()?;
let csv_data = dashboard.export_to_csv()?;
let html_report = dashboard.generate_html()?;
```

The dashboard provides:
- Real-time metric visualization with auto-refresh
- RESTful API endpoints (`/api/metrics`, `/api/metrics/names`)
- Interactive charts using Chart.js
- Export capabilities (JSON, CSV, HTML)
- Customizable themes and layouts

### Custom Visualizations

Create custom visualizations with the `VisualizationData` and `VisualizationMetadata` structs:

```rust
use scirs2_metrics::visualization::{
    VisualizationData, VisualizationMetadata, PlotType, backends
};

// Create custom visualization data
let mut data = VisualizationData::new();
data.x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
data.y = vec![2.0, 4.0, 1.0, 3.0, 5.0];

// Add series names for better labeling
data.series_names = Some(vec!["Data Series".to_string()]);

// Create metadata to describe the visualization
let metadata = VisualizationMetadata::line_plot(
    "Custom Plot",
    "X Axis Label",
    "Y Axis Label",
);

// Get the default backend
let backend = backends::default_backend();

// Save the visualization
let options = VisualizationOptions::default();
backend.save_to_file(&data, &metadata, &options, "custom_plot.png")?;
```

## Integration with Other Modules

### Neural Networks

The `neural_common` feature enables integration with the `scirs2-neural` module:

```rust
use scirs2_metrics::integration::neural::{
    // Neural metric adapter
    NeuralMetricAdapter,
    
    // Callback for neural network training
    MetricsCallback,
    
    // Visualization utilities
    neural_roc_curve_visualization,
    neural_precision_recall_curve_visualization,
    neural_confusion_matrix_visualization,
    training_history_visualization,
};

// Create metric adapters for neural networks
let accuracy = NeuralMetricAdapter::<f32>::accuracy();
let precision = NeuralMetricAdapter::<f32>::precision();
let f1_score = NeuralMetricAdapter::<f32>::f1_score();

// Use metrics during neural network training
let metrics = vec![accuracy, precision, f1_score];
let callback = MetricsCallback::new(metrics, true);

// Visualize neural network metrics
let roc_viz = neural_roc_curve_visualization(&y_true, &y_pred, Some(0.85))?;
let history_viz = training_history_visualization(
    vec!["loss".to_string(), "accuracy".to_string()],
    history,
    val_history,
);
```

### Optimization and Hyperparameter Tuning

The `optim_integration` feature enables integration with the `scirs2-optim` module for optimization and hyperparameter tuning:

```rust
use scirs2_metrics::integration::optim::{
    // Core components
    MetricOptimizer, 
    MetricScheduler,
    OptimizationMode,
    
    // Hyperparameter tuning
    HyperParameter,
    HyperParameterTuner,
    HyperParameterSearchResult,
};

// Create a metric-based learning rate scheduler
let mut scheduler = MetricScheduler::new(
    0.1,            // Initial learning rate
    0.5,            // Factor (reduce by half)
    2,              // Patience 
    0.001,          // Minimum learning rate
    "validation_loss", // Metric name
    false,          // Maximize? No, minimize
);

// Update scheduler based on metric values
let new_lr = scheduler.step_with_metric(val_loss);

// Hyperparameter tuning
let params = vec![
    HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
    HyperParameter::new("hidden_size", 5.0, 2.0, 20.0),
    HyperParameter::discrete("num_epochs", 100.0, 50.0, 500.0, 50.0),
];

let mut tuner = HyperParameterTuner::new(
    params,
    "accuracy",  // Metric to optimize
    true,        // Maximize
    20           // Number of trials
);

// Run hyperparameter search
let result = tuner.random_search(|params| {
    // Train model with these parameters, return metric
    let accuracy = train_and_evaluate(params)?;
    Ok(accuracy)
})?;

// Get best parameters
println!("Best accuracy: {}", result.best_metric());
for (name, value) in result.best_params() {
    println!("{}: {}", name, value);
}
```

When combined with `scirs2-optim`:

```rust
use scirs2_optim::metrics::{
    MetricOptimizer,
    MetricScheduler,
    MetricBasedReduceOnPlateau,
};
use scirs2_optim::optimizers::{SGD, Optimizer};

// Create SGD optimizer guided by metrics
let mut optimizer = MetricOptimizer::new(
    SGD::new(0.1),
    "accuracy",  // Metric to optimize
    true         // Maximize
);

// Create scheduler that adjusts learning rate based on metrics
let mut scheduler = MetricBasedReduceOnPlateau::new(
    0.1,        // Initial learning rate
    0.5,        // Factor
    5,          // Patience
    0.001,      // Minimum learning rate
    "val_loss",  // Metric to monitor
    false       // Maximize? No, minimize
);

// During training loop:
optimizer.update_metric(accuracy);
scheduler.step_with_metric(val_loss);
scheduler.apply_to(&mut optimizer);
```

## Components

### Classification Metrics

Functions for classification evaluation:

```rust
use scirs2_metrics::classification::{
    // Basic Metrics
    accuracy_score,         // Calculate accuracy score
    precision_score,        // Calculate precision score
    recall_score,           // Calculate recall score
    f1_score,               // Calculate F1 score
    fbeta_score,            // Calculate F-beta score
    precision_recall_fscore_support, // Calculate precision, recall, F-score, and support
    
    // Multi-class and Multi-label Metrics
    jaccard_score,          // Calculate Jaccard similarity coefficient
    hamming_loss,           // Calculate Hamming loss
    
    // Probability-based Metrics
    log_loss,               // Calculate logarithmic loss
    brier_score_loss,       // Calculate Brier score loss
    
    // ROC and AUC
    roc_curve,              // Calculate Receiver Operating Characteristic (ROC) curve
    roc_auc_score,          // Calculate Area Under the ROC Curve (AUC)
    average_precision_score, // Calculate average precision score
    
    // Threshold Optimization
    threshold::precision_recall_curve, // Calculate precision-recall curve
    threshold::g_means_score,         // Calculate G-means score (geometric mean of recall and specificity)
    threshold::find_optimal_threshold, // Find optimal threshold using custom metrics
    threshold::find_optimal_threshold_g_means, // Find threshold that maximizes G-means
    
    // Confusion Matrix and Derived Metrics
    confusion_matrix,       // Calculate confusion matrix
    classification_report,  // Generate text report of classification metrics
    
    // Probabilities to Labels
    binarize,               // Transform probabilities to binary labels
    label_binarize,         // Transform multi-class labels to binary labels
    
    // Other Metrics
    cohen_kappa_score,      // Calculate Cohen's kappa
    matthews_corrcoef,      // Calculate Matthews correlation coefficient
    hinge_loss,             // Calculate hinge loss
};
```

### Regression Metrics

Functions for regression evaluation:

```rust
use scirs2_metrics::regression::{
    // Error Metrics
    mean_squared_error,     // Calculate mean squared error
    mean_absolute_error,    // Calculate mean absolute error
    mean_absolute_percentage_error, // Calculate mean absolute percentage error
    median_absolute_error,  // Calculate median absolute error
    max_error,              // Calculate maximum error
    
    // Goodness of Fit
    r2_score,               // Calculate R² score (coefficient of determination)
    explained_variance_score, // Calculate explained variance score
    
    // Other Metrics
    mean_squared_log_error, // Calculate mean squared logarithmic error
    mean_poisson_deviance,  // Calculate mean Poisson deviance
    mean_gamma_deviance,    // Calculate mean Gamma deviance
    mean_tweedie_deviance,  // Calculate mean Tweedie deviance
};
```

### Clustering Metrics

Functions for clustering evaluation:

```rust
use scirs2_metrics::clustering::{
    // Internal Metrics (no ground truth)
    silhouette_score,       // Calculate Silhouette Coefficient
    calinski_harabasz_score, // Calculate Calinski-Harabasz Index
    davies_bouldin_score,   // Calculate Davies-Bouldin Index
    
    // External Metrics (with ground truth)
    adjusted_rand_index,    // Calculate Adjusted Rand Index
    normalized_mutual_info_score, // Calculate normalized mutual information
    adjusted_mutual_info_score, // Calculate adjusted mutual information
    homogeneity_completeness_v_measure, // Calculate homogeneity, completeness and V-measure
    fowlkes_mallows_score,  // Calculate Fowlkes-Mallows Index
    
    // Contingency Matrix
    contingency_matrix,     // Calculate contingency matrix
    pair_confusion_matrix,  // Calculate pair confusion matrix
};
```

### Evaluation Functions

General evaluation tools:

```rust
use scirs2_metrics::evaluation::{
    // Cross-validation
    cross_val_score,        // Evaluate a score by cross-validation
    cross_validate,         // Evaluate metrics by cross-validation
    
    // Train/Test Splitting
    train_test_split,       // Split arrays into random train and test subsets
    
    // Learning Curves
    learning_curve,         // Generate learning curve
    validation_curve,       // Generate validation curve
    
    // Hyperparameter Optimization
    grid_search_cv,         // Exhaustive search over parameter grid
    randomized_search_cv,   // Random search over parameters
};
```

### Visualization

Functions and types for visualization:

```rust
use scirs2_metrics::visualization::{
    // Core Types
    VisualizationData,      // Container for visualization data
    VisualizationMetadata,  // Metadata for visualizations
    VisualizationOptions,   // Options for rendering visualizations
    PlotType,               // Enum of supported plot types
    ColorMap,               // Enum of color maps for heatmaps
    
    // Traits
    MetricVisualizer,       // Common trait for all visualizers
    
    // Visualization Helpers
    helpers::visualize_confusion_matrix,    // Create confusion matrix visualizations
    helpers::visualize_roc_curve,           // Create ROC curve visualizations
    helpers::visualize_precision_recall_curve, // Create PR curve visualizations
    helpers::visualize_calibration_curve,   // Create calibration curve visualizations
    helpers::visualize_metric,              // Create generic metric visualizations
    helpers::visualize_multi_curve,         // Create multi-curve visualizations
    helpers::visualize_histogram,           // Create histogram visualizations
    helpers::visualize_heatmap,             // Create heatmap visualizations
    
    // Backend Management
    backends::default_backend,              // Get the default plotting backend
    backends::PlottingBackend,              // Trait for plotting backends
    backends::PlottersBackend,              // Plotters backend implementation
    backends::PlotlyBackend,                // Plotly backend implementation
};
```

## Production Readiness

SciRS2 Metrics is production-ready with comprehensive features and robust testing:

### Quality Assurance
- **142+ comprehensive tests** covering all metrics and edge cases
- **Zero-warning policy** enforced through CI/CD pipelines
- **Numerical precision testing** for numerical stability
- **Performance regression testing** to maintain optimization standards
- **Cross-platform compatibility** verified on major operating systems

### Performance & Scalability
- **SIMD-accelerated computations** with automatic hardware detection
- **Memory-efficient streaming algorithms** for large datasets
- **Parallel processing support** via Rayon integration
- **Chunked processing** for memory-constrained environments
- **Hardware-specific optimizations** (SSE2, AVX2, AVX-512)

### Reliability & Robustness
- **Comprehensive error handling** with descriptive error messages
- **Edge case coverage** including NaN, infinity, and empty data handling
- **Type-safe implementations** leveraging Rust's type system
- **API stability** with semantic versioning guarantees
- **Extensive documentation** with mathematical formulations and examples

### Integration & Ecosystem
- **Scikit-learn compatibility** for seamless migration from Python
- **Modular architecture** allowing selective feature usage
- **Extensible design** for custom metric development
- **Rich visualization support** with multiple backend options
- **Domain-specific metric collections** for specialized applications

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.