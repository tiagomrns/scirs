//! Examples of set-based distance metrics
//!
//! This example demonstrates the use of the Hausdorff distance and other set-based distance metrics
//! from the `scirs2-spatial` crate, which measure the similarity between sets of points.

use ndarray::{array, ArrayView2};
use scirs2_spatial::set_distance::{
    directed_hausdorff, gromov_hausdorff_distance, hausdorff_distance, wasserstein_distance,
};

/// Print distance and description
fn print_distance(name: &str, dist: f64, description: &str) {
    println!("{: <25} : {:.6}", name, dist);
    println!("{: <25}   {}", "", description);
    println!();
}

/// Create a nice visual representation of the point sets
fn visualize_sets(set1: &ArrayView2<f64>, set2: &ArrayView2<f64>) {
    let size = 10;
    let mut grid = vec![vec![' '; size]; size];

    // Scale the points to fit in our grid
    let scale = (size - 1) as f64;

    // Mark set1 points with 'A'
    for i in 0..set1.shape()[0] {
        let x = (set1[[i, 0]] * scale).round() as usize;
        let y = (set1[[i, 1]] * scale).round() as usize;
        if x < size && y < size {
            grid[y][x] = 'A';
        }
    }

    // Mark set2 points with 'B'
    for i in 0..set2.shape()[0] {
        let x = (set2[[i, 0]] * scale).round() as usize;
        let y = (set2[[i, 1]] * scale).round() as usize;
        if x < size && y < size {
            // If there's already an A, mark as 'X' for overlap
            if grid[y][x] == 'A' {
                grid[y][x] = 'X';
            } else {
                grid[y][x] = 'B';
            }
        }
    }

    // Print the grid
    println!("Visualization (A: set1, B: set2, X: overlap):");
    println!("  0123456789");
    println!(" +----------+");
    for (i, row) in grid.iter().enumerate() {
        print!("{}|", i);
        for &cell in row {
            print!("{}", cell);
        }
        println!("|");
    }
    println!(" +----------+");
    println!();
}

fn main() {
    println!("Set-Based Distance Metrics Examples");
    println!("==================================");
    println!();

    // Example 1: Basic 2D point sets
    println!("Example 1: Basic 2D Point Sets");
    println!("-----------------------------");

    let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

    let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]];

    println!("Set 1:");
    for i in 0..set1.shape()[0] {
        println!("  Point {}: ({}, {})", i, set1[[i, 0]], set1[[i, 1]]);
    }

    println!("\nSet 2:");
    for i in 0..set2.shape()[0] {
        println!("  Point {}: ({}, {})", i, set2[[i, 0]], set2[[i, 1]]);
    }

    println!();
    visualize_sets(&set1.view(), &set2.view());

    // Compute directed Hausdorff distance
    let (dist_forward, idx1, idx2) = directed_hausdorff(&set1.view(), &set2.view(), Some(42));

    print_distance(
        "Directed Hausdorff (1→2)",
        dist_forward,
        "Maximum distance from any point in set1 to its closest point in set2",
    );

    println!(
        "  Point in set1: ({}, {})",
        set1[[idx1, 0]],
        set1[[idx1, 1]]
    );
    println!(
        "  Closest point in set2: ({}, {})",
        set2[[idx2, 0]],
        set2[[idx2, 1]]
    );
    println!();

    // Compute bidirectional Hausdorff distance
    let dist = hausdorff_distance(&set1.view(), &set2.view(), Some(42));

    print_distance(
        "Hausdorff distance",
        dist,
        "Maximum of the two directed Hausdorff distances",
    );

    // Compute Wasserstein distance (Earth Mover's)
    match wasserstein_distance(&set1.view(), &set2.view()) {
        Ok(dist) => {
            print_distance(
                "Wasserstein distance",
                dist,
                "Minimum 'work' required to transform one distribution into the other",
            );
        }
        Err(e) => println!("Error computing Wasserstein distance: {}", e),
    }

    // Compute Gromov-Hausdorff distance
    let dist = gromov_hausdorff_distance(&set1.view(), &set2.view());

    print_distance(
        "Gromov-Hausdorff distance",
        dist,
        "Measure of how similar the two metric spaces are geometrically",
    );

    // Example 2: Sets with different sizes
    println!("\nExample 2: Sets with Different Sizes");
    println!("----------------------------------");

    let set3 = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]];

    let set4 = array![[0.3, 0.3], [0.7, 0.7]];

    println!("Set 3 (5 points): Points arranged in a square with center");
    println!("Set 4 (2 points): Points along the diagonal");

    println!();
    visualize_sets(&set3.view(), &set4.view());

    // Compute Hausdorff distance
    let dist = hausdorff_distance(&set3.view(), &set4.view(), Some(42));

    print_distance(
        "Hausdorff distance",
        dist,
        "Measures the maximum distance between any point and its closest neighbor in the other set",
    );

    // Compute Wasserstein distance
    match wasserstein_distance(&set3.view(), &set4.view()) {
        Ok(dist) => {
            print_distance(
                "Wasserstein distance",
                dist,
                "For different-sized sets, this is an approximation",
            );
        }
        Err(e) => println!("Error computing Wasserstein distance: {}", e),
    }

    // Example 3: Same shape but different scales
    println!("\nExample 3: Same Shape but Different Scales");
    println!("----------------------------------------");

    let set5 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

    let set6 = array![[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];

    println!("Set 5: Unit square");
    println!("Set 6: Half-sized square");

    println!();
    visualize_sets(&set5.view(), &set6.view());

    // Compute Hausdorff distance
    let dist = hausdorff_distance(&set5.view(), &set6.view(), Some(42));

    print_distance(
        "Hausdorff distance",
        dist,
        "Shows the maximum discrepancy between the sets",
    );

    // Compute Gromov-Hausdorff distance
    let dist = gromov_hausdorff_distance(&set5.view(), &set6.view());

    print_distance(
        "Gromov-Hausdorff distance",
        dist,
        "Considers the intrinsic geometry, so should be small for scaled versions",
    );

    println!("\nSet-based distances help quantify how different two point sets are geometrically.");
    println!("These metrics are useful in shape matching, image comparison, and many other applications.");
}
