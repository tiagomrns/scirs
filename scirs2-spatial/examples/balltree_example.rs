use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scirs2_spatial::balltree::BallTree;
use scirs2_spatial::distance::{manhattan, ChebyshevDistance, ManhattanDistance};
use std::time::Instant;

/// Generate random points in a unit hypercube
fn generate_random_points(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            data[(i, j)] = rng.random_range(0.0..1.0);
        }
    }

    data
}

/// Benchmark function for comparing Ball tree and brute force
fn benchmark_nearest_neighbor(
    tree: &BallTree<f64, ManhattanDistance<f64>>,
    data: &Array2<f64>,
    k: usize,
) {
    let n_queries = 10;
    let query_points = generate_random_points(n_queries, data.ncols(), 42);

    // Benchmark Ball tree queries
    let start = Instant::now();
    for i in 0..n_queries {
        let query_point = query_points.row(i).to_vec();
        let (_, _) = tree.query(&query_point, k, true).unwrap();
    }
    let ball_tree_time = start.elapsed();

    // Benchmark brute force queries
    let start = Instant::now();
    for i in 0..n_queries {
        let query_point = query_points.row(i).to_vec();

        // Compute distances to all points
        let mut distances = Vec::with_capacity(data.nrows());
        for j in 0..data.nrows() {
            let point = data.row(j).to_vec();
            let dist = manhattan(&query_point, &point);
            distances.push((j, dist));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (_indices, _distances): (Vec<_>, Vec<_>) = distances.into_iter().take(k).unzip();
    }
    let brute_force_time = start.elapsed();

    println!("Ball tree time: {:?}", ball_tree_time);
    println!("Brute force time: {:?}", brute_force_time);
    println!(
        "Speedup: {:.2}x",
        brute_force_time.as_secs_f64() / ball_tree_time.as_secs_f64()
    );
}

/// Compare different distance metrics
fn compare_distance_metrics(data: &Array2<f64>, query_point: &[f64], k: usize) {
    // Create Ball trees with different distance metrics
    let ball_tree_euclidean = BallTree::with_euclidean_distance(&data.view(), 20).unwrap();
    let ball_tree_manhattan = BallTree::new(&data.view(), 20, ManhattanDistance::new()).unwrap();
    let ball_tree_chebyshev = BallTree::new(&data.view(), 20, ChebyshevDistance::new()).unwrap();

    // Query with each tree
    println!(
        "\nComparing results with different distance metrics for query point: {:?}",
        query_point
    );

    let (indices_euclidean, Some(distances_euclidean)) =
        ball_tree_euclidean.query(query_point, k, true).unwrap()
    else {
        unreachable!()
    };

    let (indices_manhattan, Some(distances_manhattan)) =
        ball_tree_manhattan.query(query_point, k, true).unwrap()
    else {
        unreachable!()
    };

    let (indices_chebyshev, Some(distances_chebyshev)) =
        ball_tree_chebyshev.query(query_point, k, true).unwrap()
    else {
        unreachable!()
    };

    println!("Euclidean distances:");
    for (i, (&idx, &dist)) in indices_euclidean
        .iter()
        .zip(distances_euclidean.iter())
        .enumerate()
    {
        println!(
            "  {}. Index: {}, Distance: {:.4}, Point: {:?}",
            i + 1,
            idx,
            dist,
            data.row(idx)
        );
    }

    println!("\nManhattan distances:");
    for (i, (&idx, &dist)) in indices_manhattan
        .iter()
        .zip(distances_manhattan.iter())
        .enumerate()
    {
        println!(
            "  {}. Index: {}, Distance: {:.4}, Point: {:?}",
            i + 1,
            idx,
            dist,
            data.row(idx)
        );
    }

    println!("\nChebyshev distances:");
    for (i, (&idx, &dist)) in indices_chebyshev
        .iter()
        .zip(distances_chebyshev.iter())
        .enumerate()
    {
        println!(
            "  {}. Index: {}, Distance: {:.4}, Point: {:?}",
            i + 1,
            idx,
            dist,
            data.row(idx)
        );
    }
}

/// Demonstrate radius search
fn demonstrate_radius_search(
    tree: &BallTree<f64, ManhattanDistance<f64>>,
    data: &Array2<f64>,
    query_point: &[f64],
    radius: f64,
) {
    println!(
        "\nFinding all points within radius {} of query point {:?}:",
        radius, query_point
    );

    let (indices, Some(distances)) = tree.query_radius(query_point, radius, true).unwrap() else {
        unreachable!()
    };

    println!("Found {} points within radius {}:", indices.len(), radius);
    for (i, (&idx, &dist)) in indices.iter().zip(distances.iter()).enumerate() {
        println!(
            "  {}. Index: {}, Distance: {:.4}, Point: {:?}",
            i + 1,
            idx,
            dist,
            data.row(idx)
        );
    }
}

/// Demonstrate dual tree search (finding pairs of points within a radius)
fn demonstrate_dual_tree_search(data1: &Array2<f64>, data2: &Array2<f64>, radius: f64) {
    println!(
        "\nFinding pairs of points from two datasets within radius {}:",
        radius
    );

    let tree1 = BallTree::new(&data1.view(), 20, ManhattanDistance::new()).unwrap();
    let tree2 = BallTree::new(&data2.view(), 20, ManhattanDistance::new()).unwrap();

    let pairs = tree1.query_radius_tree(&tree2, radius).unwrap();

    println!(
        "Found {} pairs of points within radius {}:",
        pairs.len(),
        radius
    );
    for (i, &(idx1, idx2)) in pairs.iter().enumerate().take(10) {
        // Show only first 10 pairs
        println!(
            "  {}. Pair: ({}, {}), Points: {:?} and {:?}",
            i + 1,
            idx1,
            idx2,
            data1.row(idx1),
            data2.row(idx2)
        );
    }

    if pairs.len() > 10 {
        println!("  ... and {} more pairs", pairs.len() - 10);
    }
}

fn main() {
    println!("Ball Tree Example");
    println!("----------------\n");

    // Generate some random data
    let n_samples = 1000;
    let n_features = 3;
    let data = generate_random_points(n_samples, n_features, 42);

    println!(
        "Created dataset with {} points in {} dimensions",
        n_samples, n_features
    );

    // Create a Ball tree
    let leaf_size = 20;
    let ball_tree = BallTree::new(&data.view(), leaf_size, ManhattanDistance::new()).unwrap();

    println!(
        "Successfully created Ball tree with leaf size {}",
        leaf_size
    );

    // Basic k-nearest neighbors query
    let query_point = [0.5, 0.5, 0.5];
    let k = 5;

    println!(
        "\nFinding {} nearest neighbors to query point {:?}:",
        k, query_point
    );

    let (indices, Some(distances)) = ball_tree.query(&query_point, k, true).unwrap() else {
        unreachable!()
    };

    println!("Nearest neighbors (using Manhattan distance):");
    for (i, (&idx, &dist)) in indices.iter().zip(distances.iter()).enumerate() {
        println!(
            "  {}. Index: {}, Distance: {:.4}, Point: {:?}",
            i + 1,
            idx,
            dist,
            data.row(idx)
        );
    }

    // Benchmark against brute force search
    println!(
        "\nBenchmarking Ball tree vs brute force for k={} nearest neighbors:",
        k
    );
    benchmark_nearest_neighbor(&ball_tree, &data, k);

    // Compare different distance metrics
    compare_distance_metrics(&data, &query_point, k);

    // Demonstrate radius search
    demonstrate_radius_search(&ball_tree, &data, &query_point, 0.3);

    // Demonstrate dual tree search
    let data2 = generate_random_points(500, n_features, 123);
    demonstrate_dual_tree_search(&data, &data2, 0.2);
}
