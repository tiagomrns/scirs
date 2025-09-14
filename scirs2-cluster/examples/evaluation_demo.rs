//! Demonstration of clustering evaluation metrics
//!
//! This example shows how to use clustering evaluation metrics to
//! assess the quality of different clustering results.

use ndarray::{Array1, Array2};
use scirs2_cluster::metrics::{davies_bouldin_score, silhouette_score};
use scirs2_cluster::vq::{kmeans2, MinitMethod, MissingMethod};

#[allow(dead_code)]
fn generate_dataset() -> (Array2<f64>, Array1<i32>) {
    // Generate three well-separated clusters
    let mut data = Vec::new();
    let mut true_labels = Vec::new();

    // Cluster 1: centered at (0, 0)
    for i in 0..50 {
        let x = i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        let y = i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        data.push(x);
        data.push(y);
        true_labels.push(0);
    }

    // Cluster 2: centered at (5, 5)
    for i in 0..50 {
        let x = 5.0 + i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        let y = 5.0 + i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        data.push(x);
        data.push(y);
        true_labels.push(1);
    }

    // Cluster 3: centered at (-5, 5)
    for i in 0..50 {
        let x = -5.0 + i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        let y = 5.0 + i as f64 * 0.1 - 2.5 + rand::random::<f64>() * 0.5;
        data.push(x);
        data.push(y);
        true_labels.push(2);
    }

    let data_array = Array2::from_shape_vec((150, 2), data).unwrap();
    let labels_array = Array1::from(true_labels);

    (data_array, labels_array)
}

#[allow(dead_code)]
fn evaluate_clustering(data: &Array2<f64>, labels: &Array1<usize>) {
    // Convert usize labels to i32 for metric functions
    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);

    // Calculate silhouette score
    let silhouette = silhouette_score(data.view(), labels_i32.view()).unwrap_or_else(|e| {
        eprintln!("Failed to calculate silhouette score: {}", e);
        0.0
    });

    // Calculate Davies-Bouldin score
    let davies_bouldin = davies_bouldin_score(data.view(), labels_i32.view()).unwrap_or_else(|e| {
        eprintln!("Failed to calculate Davies-Bouldin score: {}", e);
        f64::INFINITY
    });

    println!("Evaluation Metrics:");
    println!("  Silhouette coefficient: {:.3}", silhouette);
    println!("  Davies-Bouldin index: {:.3}", davies_bouldin);
    println!("  (Higher silhouette is better, lower Davies-Bouldin is better)");
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Clustering Evaluation Metrics Demo");
    println!("=================================\n");

    let (data, true_labels) = generate_dataset();

    // Test with different numbers of clusters
    let test_k_values = vec![2, 3, 4, 5];

    for k in test_k_values {
        println!("\nTesting with k={} clusters:", k);

        // Run K-means
        let (centroids, labels) = kmeans2(
            data.view(),
            k,
            Some(300),
            None,
            Some(MinitMethod::PlusPlus),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )?;

        println!("Found {} non-empty clusters", centroids.shape()[0]);

        // Evaluate clustering quality
        evaluate_clustering(&data, &labels);
    }

    // Compare different initialization methods
    println!("\n\nComparing initialization methods for k=3:");
    println!("=========================================");

    let init_methods = vec![
        ("Random", MinitMethod::Random),
        ("K-means++", MinitMethod::PlusPlus),
        ("Points", MinitMethod::Points),
    ];

    for (name, init_method) in init_methods {
        println!("\n{} initialization:", name);

        let (_centroids, labels) = kmeans2(
            data.view(),
            3,
            Some(300),
            None,
            Some(init_method),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )?;

        evaluate_clustering(&data, &labels);
    }

    Ok(())
}
