# KD-Tree Optimizations

The `kdtree_optimized` module in `scirs2-spatial` provides extensions to the KD-tree data structure for efficient operations on large point sets.

## Overview

KD-trees are efficient spatial indexing structures for nearest neighbor searches, but they can be further optimized for specific tasks. This module provides specialized methods for:

1. **Efficient Hausdorff distance computation** - Using the KD-tree structure to accelerate set-based distance computations
2. **Batch nearest neighbor queries** - Processing multiple query points efficiently with improved cache locality and optional parallelization

## Usage

### Optimized Hausdorff Distance

The standard Hausdorff distance computation requires comparing every point in one set with every point in another, resulting in O(n²) complexity. Using the KD-tree, we can accelerate this to approximately O(n log n):

```rust
use scirs2_spatial::kdtree::KDTree;
use scirs2_spatial::kdtree_optimized::KDTreeOptimized;
use ndarray::array;

// Create point sets
let set1 = array![
    [0.0, 0.0],
    [1.0, 0.0], 
    [0.0, 1.0],
    [1.0, 1.0]
];

let set2 = array![
    [0.1, 0.1],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.9, 0.9]
];

// Build KD-tree for the first set
let tree = KDTree::new(&set1).unwrap();

// Compute Hausdorff distance using the optimized method
let distance = tree.hausdorff_distance(&set2.view(), None).unwrap();
println!("Hausdorff distance: {}", distance);

// You can also compute the directed Hausdorff distance
// which returns the distance and indices of the points that realize it
let (dist, idx1, idx2) = tree.directed_hausdorff_distance(&set2.view(), None).unwrap();
println!("Directed Hausdorff distance: {}", dist);
println!("From point in set1 at index {}: {:?}", idx1, set1.row(idx1));
println!("To point in set2 at index {}: {:?}", idx2, set2.row(idx2));
```

### Batch Nearest Neighbor Queries

When you need to find the nearest neighbors for multiple query points, using batch processing is significantly more efficient than performing individual queries:

```rust
use scirs2_spatial::kdtree::KDTree;
use scirs2_spatial::kdtree_optimized::KDTreeOptimized;
use ndarray::array;

// Create reference points
let points = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5]
];

// Build KD-tree
let tree = KDTree::new(&points).unwrap();

// Query points for which we want to find nearest neighbors
let queries = array![
    [0.2, 0.2],
    [0.8, 0.2],
    [0.2, 0.8],
    [0.8, 0.8]
];

// Find nearest neighbors for all queries in one batch operation
let (indices, distances) = tree.batch_nearest_neighbor(&queries.view()).unwrap();

// Process results
for i in 0..queries.shape()[0] {
    println!(
        "Query point ({}, {}) -> Nearest: point {} at distance {}",
        queries[[i, 0]], queries[[i, 1]],
        indices[i], distances[i]
    );
}
```

## Performance Considerations

1. **Tree Construction:** Building the KD-tree is an O(n log n) operation, but the cost is amortized when performing multiple queries.

2. **Batch Size:** The batch nearest neighbor function processes queries in batches (default 32 points) for better cache locality.

3. **Parallel Processing:** When the `parallel` feature is enabled, batch operations use Rayon for parallel processing of query points.

4. **Hausdorff Distance Optimization:** For small point sets (< 100 points), the direct algorithm may be faster, but for large sets, the KD-tree acceleration provides significant speedup.

5. **Memory Usage:** The KD-tree structure requires additional memory, but the improved query performance usually justifies this overhead for large datasets.

## Use Cases

The optimized KD-tree operations are particularly useful for:

1. **Shape Matching:** Computing Hausdorff distances between large point clouds or meshes

2. **Feature Matching:** Finding correspondences between feature sets in computer vision

3. **Batch Processing:** Applications that need to process many query points at once, like nearest-neighbor-based classifiers

4. **Large Dataset Analysis:** Working with datasets that contain thousands or millions of points