//! Example demonstrating optimized KD-tree operations
//!
//! This example shows how to use the KDTreeOptimized trait to perform
//! efficient operations on large point sets, including optimized Hausdorff
//! distance computation and batch nearest neighbor queries.

use ndarray::{array, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::kdtree::KDTree;
use scirs2_spatial::kdtree_optimized::KDTreeOptimized;
use scirs2_spatial::set_distance::hausdorff_distance;
use std::time::Instant;

/// Generate random points in the unit square
fn generate_random_points(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Array2::zeros((n, 2));

    for i in 0..n {
        points[[i, 0]] = rng.random();
        points[[i, 1]] = rng.random();
    }

    points
}

/// Benchmark Hausdorff distance computation: direct vs. KD-tree optimized
fn benchmark_hausdorff(n1: usize, n2: usize) {
    println!("Benchmarking Hausdorff distance computation");
    println!("------------------------------------------");
    println!("Set 1: {} points, Set 2: {} points", n1, n2);

    // Generate random point sets
    let points1 = generate_random_points(n1, 42);
    let points2 = generate_random_points(n2, 43);

    // Build KD-tree for set 1
    println!("Building KD-tree for set 1...");
    let start = Instant::now();
    let kdtree = KDTree::new(&points1).unwrap();
    let build_time = start.elapsed();
    println!("KD-tree built in {:?}", build_time);

    // Compute Hausdorff distance using direct method
    println!("\nComputing Hausdorff distance (direct method)...");
    let start = Instant::now();
    let direct_dist = hausdorff_distance(&points1.view(), &points2.view(), Some(42));
    let direct_time = start.elapsed();
    println!("Distance: {:.6}", direct_dist);
    println!("Time taken: {:?}", direct_time);

    // Compute Hausdorff distance using KD-tree
    println!("\nComputing Hausdorff distance (KD-tree optimized)...");
    let start = Instant::now();
    let optimized_dist = kdtree
        .hausdorff_distance(&points2.view(), Some(42))
        .unwrap();
    let optimized_time = start.elapsed();
    println!("Distance: {:.6}", optimized_dist);
    println!("Time taken: {:?}", optimized_time);

    // Report speedup
    let speedup = direct_time.as_secs_f64() / optimized_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);

    // Verify that both methods produce similar results
    println!(
        "Distance difference: {:.10}",
        (direct_dist - optimized_dist).abs()
    );
}

/// Demonstrate batch nearest neighbor computation
fn demo_batch_nearest_neighbor() {
    println!("\nBatch Nearest Neighbor Computation");
    println!("----------------------------------");

    // Create a simple KD-tree with a grid of points
    let mut grid_points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            grid_points.push([i as f64, j as f64]);
        }
    }
    let points =
        Array2::from_shape_vec((100, 2), grid_points.into_iter().flatten().collect()).unwrap();

    // Build KD-tree
    let kdtree = KDTree::new(&points).unwrap();

    // Create some query points
    let query_points = array![
        [2.3, 3.7], // Should be close to [2, 4]
        [5.1, 5.1], // Should be close to [5, 5]
        [9.9, 0.1], // Should be close to [10, 0]
        [4.5, 7.8], // Should be close to [4, 8]
    ];

    // Individual queries
    println!("Individual queries:");
    let start = Instant::now();
    for i in 0..query_points.shape()[0] {
        let point = query_points.row(i).to_vec();
        let (indices, distances) = kdtree.query(&point, 1).unwrap();
        let nearest_point = points.row(indices[0]);
        println!(
            "Query point: ({:.1}, {:.1}) -> Nearest: ({:.1}, {:.1}), Distance: {:.2}",
            point[0], point[1], nearest_point[0], nearest_point[1], distances[0]
        );
    }
    let individual_time = start.elapsed();
    println!("Total time for individual queries: {:?}", individual_time);

    // Batch queries
    println!("\nBatch queries:");
    let start = Instant::now();
    let (indices, distances) = kdtree.batch_nearest_neighbor(&query_points.view()).unwrap();
    for i in 0..query_points.shape()[0] {
        let point = query_points.row(i);
        let nearest_point = points.row(indices[i]);
        println!(
            "Query point: ({:.1}, {:.1}) -> Nearest: ({:.1}, {:.1}), Distance: {:.2}",
            point[0], point[1], nearest_point[0], nearest_point[1], distances[i]
        );
    }
    let batch_time = start.elapsed();
    println!("Total time for batch queries: {:?}", batch_time);

    // Report speedup
    let speedup = individual_time.as_secs_f64() / batch_time.as_secs_f64();
    println!("\nBatch query speedup: {:.2}x", speedup);
}

fn main() {
    println!("KD-Tree Optimized Operations Example");
    println!("===================================");

    // Benchmark Hausdorff distance
    benchmark_hausdorff(1000, 1000);

    // Demonstrate batch nearest neighbor
    demo_batch_nearest_neighbor();

    // Performance advice
    println!("\nPerformance Advice");
    println!("-----------------");
    println!("1. The KD-tree optimization is most effective for large point sets");
    println!("2. For small point sets (< 100 points), direct computation may be faster");
    println!("3. Batch operations are more efficient than individual queries when");
    println!("   processing multiple points at once");
    println!("4. Enable the 'parallel' feature for further speedups with larger datasets");
}
