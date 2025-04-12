//! Dataset generators

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use std::f64::consts::PI;

/// Generate a random classification dataset with clusters
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({}) must be >= n_informative ({})",
            n_features, n_informative
        )));
    }

    if n_classes < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be >= 2".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate centroids for each class and cluster
    let n_centroids = n_classes * n_clusters_per_class;
    let mut centroids = Array2::zeros((n_centroids, n_informative));
    let scale = 2.0;

    for i in 0..n_centroids {
        for j in 0..n_informative {
            centroids[[i, j]] = scale * rng.random_range(-1.0f64..1.0f64);
        }
    }

    // Generate samples
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    // Samples per class
    let samples_per_class = n_samples / n_classes;
    let remainder = n_samples % n_classes;

    let mut sample_idx = 0;

    for class in 0..n_classes {
        let n_samples_class = if class < remainder {
            samples_per_class + 1
        } else {
            samples_per_class
        };

        // Assign clusters within this class
        let samples_per_cluster = n_samples_class / n_clusters_per_class;
        let cluster_remainder = n_samples_class % n_clusters_per_class;

        for cluster in 0..n_clusters_per_class {
            let n_samples_cluster = if cluster < cluster_remainder {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            let centroid_idx = class * n_clusters_per_class + cluster;

            for _ in 0..n_samples_cluster {
                // Randomly select a point near the cluster centroid
                for j in 0..n_informative {
                    data[[sample_idx, j]] =
                        centroids[[centroid_idx, j]] + 0.3 * normal.sample(&mut rng);
                }

                // Add noise features
                for j in n_informative..n_features {
                    data[[sample_idx, j]] = normal.sample(&mut rng);
                }

                target[sample_idx] = class as f64;
                sample_idx += 1;
            }
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    // Create class names
    let class_names: Vec<String> = (0..n_classes).map(|i| format!("class_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(class_names)
        .with_description(format!(
            "Synthetic classification dataset with {} classes and {} features",
            n_classes, n_features
        ));

    Ok(dataset)
}

/// Generate a random regression dataset
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({}) must be >= n_informative ({})",
            n_features, n_informative
        )));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate the coefficients for the informative features
    let mut coef = Array1::zeros(n_features);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_informative {
        coef[i] = 100.0 * normal.sample(&mut rng);
    }

    // Generate the features
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Generate the target
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut y = 0.0;
        for j in 0..n_features {
            y += data[[i, j]] * coef[j];
        }

        // Add noise
        if noise > 0.0 {
            y += normal.sample(&mut rng) * noise;
        }

        target[i] = y;
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic regression dataset with {} features ({} informative)",
            n_features, n_informative
        ))
        .with_metadata("noise", &noise.to_string())
        .with_metadata("coefficients", &format!("{:?}", coef));

    Ok(dataset)
}

/// Generate a random time series dataset
pub fn make_time_series(
    n_samples: usize,
    n_features: usize,
    trend: bool,
    seasonality: bool,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::zeros((n_samples, n_features));

    for feature in 0..n_features {
        let trend_coef = if trend {
            rng.random_range(0.01f64..0.1f64)
        } else {
            0.0
        };
        let seasonality_period = rng.random_range(10..=50) as f64;
        let seasonality_amplitude = if seasonality {
            rng.random_range(1.0f64..5.0f64)
        } else {
            0.0
        };

        let base_value = rng.random_range(-10.0f64..10.0f64);

        for i in 0..n_samples {
            let t = i as f64;

            // Add base value
            let mut value = base_value;

            // Add trend
            if trend {
                value += trend_coef * t;
            }

            // Add seasonality
            if seasonality {
                value += seasonality_amplitude * (2.0 * PI * t / seasonality_period).sin();
            }

            // Add noise
            if noise > 0.0 {
                value += normal.sample(&mut rng) * noise;
            }

            data[[i, feature]] = value;
        }
    }

    // Create time index (unused for now but can be useful for plotting)
    let time_index: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let _time_array = Array1::from(time_index);

    // Create dataset
    let mut dataset = Dataset::new(data, None);

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic time series dataset with {} features",
            n_features
        ))
        .with_metadata("trend", &trend.to_string())
        .with_metadata("seasonality", &seasonality.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a random blobs dataset for clustering
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(-center_box..=center_box);
        }
    }

    // Generate samples around centers
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

    // Samples per center
    let samples_per_center = n_samples / centers;
    let remainder = n_samples % centers;

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        for _ in 0..n_samples_center {
            for j in 0..n_features {
                data[[sample_idx, j]] = cluster_centers[[center_idx, j]] + normal.sample(&mut rng);
            }

            target[sample_idx] = center_idx as f64;
            sample_idx += 1;
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic clustering dataset with {} clusters and {} features",
            centers, n_features
        ))
        .with_metadata("centers", &centers.to_string())
        .with_metadata("cluster_std", &cluster_std.to_string());

    Ok(dataset)
}
