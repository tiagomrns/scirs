//! Demonstration of GPU acceleration capabilities for clustering algorithms
//!
//! This example shows how to use GPU-accelerated clustering algorithms in scirs2-cluster
//! with automatic fallback to CPU when GPU is not available.

use ndarray::Array2;
use scirs2_cluster::preprocess::standardize;

// GPU acceleration imports (commented out - requires gpu feature)
// use scirs2_cluster::{
//     gpu_accelerated::{gpu_dbscan, gpu_hierarchical, gpu_kmeans},
//     DeviceSelection, GpuBackend, GpuConfig, GpuLinkageMethod, MemoryStrategy,
// };
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU Acceleration Demo for scirs2-cluster");
    println!("=======================================");

    // Create larger dataset to show GPU benefits
    let data = create_large_sample_data(5000, 50);
    println!(
        "Generated {} data points with {} features",
        data.nrows(),
        data.ncols()
    );

    // Standardize the data for better clustering
    let standardized = standardize(data.view(), true)?;

    // Test different GPU configurations
    test_automatic_gpu_detection(&standardized)?;
    test_specific_gpu_backends(&standardized)?;
    test_memory_strategies(&standardized)?;
    test_multi_algorithm_comparison(&standardized)?;

    println!("\nGPU acceleration demo completed!");

    Ok(())
}

/// Test automatic GPU detection and fallback
#[allow(dead_code)]
fn test_automatic_gpu_detection(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Testing Automatic GPU Detection");
    println!("==================================");

    // Use automatic GPU detection
    let auto_config = GpuConfig {
        backend: GpuBackend::CpuFallback, // Will auto-detect best GPU
        device_selection: DeviceSelection::Automatic,
        memory_strategy: MemoryStrategy::Explicit,
        block_size: 256,
        grid_size: 1024,
        auto_tune: true,
        cpu_fallback: true,
    };

    println!("Running GPU K-means with automatic detection...");
    let start_time = std::time::Instant::now();

    let (centroids, labels) = gpu_kmeans(
        data.view(),
        5, // k=5 clusters
        Some(auto_config.clone()),
    )?;

    let duration = start_time.elapsed();
    println!("✓ K-means completed in {:.2?}", duration);
    println!("  - Found {} clusters", centroids.nrows());
    println!("  - Assigned {} points", labels.len());

    // Test GPU DBSCAN
    println!("Running GPU DBSCAN...");
    let start_time = std::time::Instant::now();

    let (dbscan_labels, core_samples) = gpu_dbscan(
        data.view(),
        0.5, // eps
        10,  // min_samples
        Some(auto_config.clone()),
    )?;

    let duration = start_time.elapsed();
    println!("✓ DBSCAN completed in {:.2?}", duration);
    println!("  - Found {} core samples", core_samples.len());

    let unique_clusters = dbscan_labels
        .iter()
        .filter(|&&label| label >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!("  - Identified {} clusters", unique_clusters);

    // Test GPU Hierarchical Clustering
    println!("Running GPU Hierarchical clustering...");
    let start_time = std::time::Instant::now();

    let (hier_labels, linkage_matrix) = gpu_hierarchical(
        data.view(),
        4, // n_clusters
        GpuLinkageMethod::Ward,
        Some(auto_config),
    )?;

    let duration = start_time.elapsed();
    println!("✓ Hierarchical clustering completed in {:.2?}", duration);
    println!(
        "  - Generated linkage matrix: {}x{}",
        linkage_matrix.nrows(),
        linkage_matrix.ncols()
    );
    println!(
        "  - Assigned {} points to {} clusters",
        hier_labels.len(),
        4
    );

    Ok(())
}

/// Test specific GPU backends
#[allow(dead_code)]
fn test_specific_gpu_backends(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Testing Specific GPU Backends");
    println!("================================");

    let backends = vec![
        (GpuBackend::Cuda, "CUDA"),
        (GpuBackend::OpenCl, "OpenCL"),
        (GpuBackend::Metal, "Metal"),
        (GpuBackend::Rocm, "ROCm"),
        (GpuBackend::OneApi, "Intel OneAPI"),
    ];

    for (backend, name) in backends {
        println!("Testing {} backend...", name);

        let backend_config = GpuConfig {
            backend,
            device_selection: DeviceSelection::Automatic,
            memory_strategy: MemoryStrategy::Explicit,
            auto_tune: true,
            cpu_fallback: true, // Important for fallback when backend not available
            ..Default::default()
        };

        let start_time = std::time::Instant::now();

        match gpu_kmeans(data.view(), 3, Some(backend_config)) {
            Ok((centroids, labels)) => {
                let duration = start_time.elapsed();
                println!(
                    "✓ {} K-means: {:.2?} ({} clusters, {} points)",
                    name,
                    duration,
                    centroids.nrows(),
                    labels.len()
                );
            }
            Err(e) => {
                println!("× {} not available: {}", name, e);
            }
        }
    }

    Ok(())
}

/// Test different memory strategies
#[allow(dead_code)]
fn test_memory_strategies(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Testing Memory Management Strategies");
    println!("======================================");

    let memory_strategies = vec![
        (MemoryStrategy::Explicit, "Explicit"),
        (MemoryStrategy::Unified, "Unified Memory"),
        (
            MemoryStrategy::Pooled { pool_size_mb: 256 },
            "Memory Pooling (256MB)",
        ),
        (MemoryStrategy::ZeroCopy, "Zero-Copy"),
    ];

    for (strategy, name) in memory_strategies {
        println!("Testing {} strategy...", name);

        let mem_config = GpuConfig {
            backend: GpuBackend::CpuFallback, // Auto-detect
            device_selection: DeviceSelection::MostMemory,
            memory_strategy: strategy,
            auto_tune: true,
            cpu_fallback: true,
            ..Default::default()
        };

        let start_time = std::time::Instant::now();

        match gpu_kmeans(data.view(), 4, Some(mem_config)) {
            Ok((centroids, labels)) => {
                let duration = start_time.elapsed();
                println!(
                    "✓ {}: {:.2?} ({} clusters, {} points)",
                    name,
                    duration,
                    centroids.nrows(),
                    labels.len()
                );
            }
            Err(e) => {
                println!("× {} failed: {}", name, e);
            }
        }
    }

    Ok(())
}

/// Compare performance across different algorithms and configurations
#[allow(dead_code)]
fn test_multi_algorithm_comparison(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Multi-Algorithm Performance Comparison");
    println!("=========================================");

    // Optimized configuration for best performance (commented out - requires gpu feature)
    // let optimal_config = GpuConfig {
    //     backend: GpuBackend::CpuFallback, // Auto-detect best
    //     device_selection: DeviceSelection::Automatic,
    //     memory_strategy: MemoryStrategy::Pooled { pool_size: 16, mb: 512 },
    //     block_size: 512, // Larger block size for better occupancy
    //     grid_size: 2048, // More threads
    //     auto_tune: true,
    //     cpu_fallback: true,
    // };

    // Test with smaller subset for faster execution
    let subset_size = std::cmp::min(1000, data.nrows());
    let data_subset = data.slice(ndarray::s![0..subset_size, ..]);

    println!("Testing algorithms on subset of {} samples...", subset_size);

    // K-means with different cluster counts
    let cluster_counts = vec![2, 5, 8, 10];
    for k in cluster_counts {
        let start_time = std::time::Instant::now();
        match gpu_kmeans(data_subset, k, Some(optimal_config.clone())) {
            Ok((centroids, labels)) => {
                let duration = start_time.elapsed();
                println!("✓ K-means (k={}): {:.2?}", k, duration);

                // Calculate basic metrics
                let unique_labels = labels
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                println!(
                    "  - Centroids: {}x{}, Unique labels: {}",
                    centroids.nrows(),
                    centroids.ncols(),
                    unique_labels
                );
            }
            Err(e) => println!("× K-means (k={}) failed: {}", k, e),
        }
    }

    // DBSCAN with different parameters
    let dbscan_params = vec![(0.3, 5), (0.5, 10), (0.8, 15)];

    for (eps, min_samples) in dbscan_params {
        let start_time = std::time::Instant::now();
        match gpu_dbscan(data_subset, eps, min_samples, Some(optimal_config.clone())) {
            Ok((labels, core_samples)) => {
                let duration = start_time.elapsed();
                let num_clusters = labels
                    .iter()
                    .filter(|&&label| label >= 0)
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                println!(
                    "✓ DBSCAN (eps={}, min={}): {:.2?} - {} clusters, {} core samples",
                    eps,
                    min_samples,
                    duration,
                    num_clusters,
                    core_samples.len()
                );
            }
            Err(e) => println!("× DBSCAN (eps={}, min={}) failed: {}", eps, min_samples, e),
        }
    }

    // Hierarchical clustering with different linkage methods
    let linkage_methods = vec![
        (GpuLinkageMethod::Ward, "Ward"),
        (GpuLinkageMethod::Complete, "Complete"),
        (GpuLinkageMethod::Average, "Average"),
        (GpuLinkageMethod::Single, "Single"),
    ];

    for (method, name) in linkage_methods {
        let start_time = std::time::Instant::now();
        match gpu_hierarchical(data_subset, 5, method, Some(optimal_config.clone())) {
            Ok((labels, linkage_matrix)) => {
                let duration = start_time.elapsed();
                println!(
                    "✓ Hierarchical {} linkage: {:.2?} - linkage matrix {}x{}",
                    name,
                    duration,
                    linkage_matrix.nrows(),
                    linkage_matrix.ncols()
                );
            }
            Err(e) => println!("× Hierarchical {} failed: {}", name, e),
        }
    }

    Ok(())
}

/// Generate large sample data for performance testing
#[allow(dead_code)]
fn create_large_sample_data(_n_samples: usize, nfeatures: usize) -> Array2<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(_n_samples * nfeatures);

    // Create 5 distinct clusters
    let cluster_centers = vec![
        (0.0, 0.0),   // Cluster 1
        (10.0, 0.0),  // Cluster 2
        (0.0, 10.0),  // Cluster 3
        (10.0, 10.0), // Cluster 4
        (5.0, 5.0),   // Cluster 5
    ];

    let samples_per_cluster = _n_samples / cluster_centers.len();

    for (i, &(center_x, center_y)) in cluster_centers.iter().enumerate() {
        let cluster_samples = if i == cluster_centers.len() - 1 {
            _n_samples - i * samples_per_cluster // Last cluster gets remaining _samples
        } else {
            samples_per_cluster
        };

        let normal_x = Normal::new(center_x, 1.5).unwrap();
        let normal_y = Normal::new(center_y, 1.5).unwrap();

        for _ in 0..cluster_samples {
            // First two _features are the main cluster coordinates
            data.push(rng.sample(normal_x));
            data.push(rng.sample(normal_y));

            // Add additional random _features
            for _ in 2..nfeatures {
                data.push(rng.random_range(-1.0..1.0));
            }
        }
    }

    Array2::from_shape_vec((_n_samples, nfeatures), data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_data_generation() {
        let data = create_large_sample_data(1000, 10);
        assert_eq!(data.shape(), &[1000, 10]);

        // Check that data has reasonable variance
        let mean = data.mean().unwrap();
        let std = data
            .var_axis(ndarray::Axis(0), 0.0)
            .unwrap()
            .mapv(|x| x.sqrt())
            .mean()
            .unwrap();

        assert!(mean.abs() < 10.0);
        assert!(std > 0.1);
    }

    #[test]
    fn test_gpu_config_creation() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::CpuFallback);
        assert!(config.cpu_fallback);
        assert!(config.auto_tune);
    }

    #[test]
    fn test_gpu_kmeans_api() {
        let data = create_large_sample_data(100, 5);
        let config = GpuConfig::default();

        // Should not panic and should return results
        let result = gpu_kmeans(data.view(), 3, Some(config));
        match result {
            Ok((centroids, labels)) => {
                assert_eq!(centroids.nrows(), 3);
                assert_eq!(labels.len(), 100);
            }
            Err(_) => {
                // GPU not available, which is acceptable in test environment
            }
        }
    }
}
