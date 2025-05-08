use ndarray::array;
use rand::Rng;
use scirs2_spatial::Delaunay;

fn main() {
    println!("=== SciRS2 Spatial - Delaunay Triangulation Example ===\n");

    // Example 1: Simple 2D triangulation
    println!("Example 1: Simple square with interior point");
    let points_simple = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5], // Interior point
    ];

    // Create the triangulation
    let tri_simple = Delaunay::new(&points_simple).unwrap();

    // Print the simplices (triangles in 2D)
    println!("Triangles:");
    for (i, simplex) in tri_simple.simplices().iter().enumerate() {
        println!("  Triangle {}: {:?}", i, simplex);
    }

    // Find which triangle contains a test point
    let test_point = [0.25, 0.25];
    if let Some(idx) = tri_simple.find_simplex(&test_point) {
        println!("  Point {:?} is in triangle {}", test_point, idx);
    } else {
        println!("  Point {:?} is not in any triangle", test_point);
    }
    println!();

    // Example 2: 3D tetrahedralization
    println!("Example 2: 3D tetrahedralization");
    let points_3d = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        // Add slight perturbation to avoid cospherical points
        [0.3, 0.7, 0.2],
        [0.7, 0.3, 0.8],
    ];

    // Create the triangulation
    let tri_3d = match Delaunay::new(&points_3d) {
        Ok(t) => t,
        Err(e) => {
            println!("  Note: {}", e);
            println!("  3D Delaunay triangulation can be sensitive to cospherical points");
            println!("  Continuing with remaining examples...");
            println!();
            return; // Skip to next example
        }
    };

    // Print the simplices (tetrahedra in 3D)
    println!("Tetrahedra:");
    for (i, simplex) in tri_3d.simplices().iter().enumerate() {
        // Each simplex should have 4 vertices in 3D
        println!("  Tetrahedron {}: {:?}", i, simplex);
    }
    println!();

    // Example 3: Random points
    println!("Example 3: Random points");
    let n_points = 20;
    let mut rng = rand::rng();
    let mut points_data = Vec::with_capacity(n_points * 2);

    for _ in 0..n_points {
        points_data.push(rng.random_range(0.0..1.0));
        points_data.push(rng.random_range(0.0..1.0));
    }

    let points_random = ndarray::Array2::from_shape_vec((n_points, 2), points_data).unwrap();

    // Create the triangulation
    let tri_random = Delaunay::new(&points_random).unwrap();

    // Print statistics
    println!("  Number of points: {}", n_points);
    println!("  Number of triangles: {}", tri_random.simplices().len());

    // Get the convex hull from the Delaunay triangulation
    let hull = tri_random.convex_hull();
    println!("  Convex hull points: {}", hull.len());
    println!("  Hull indices: {:?}", hull);
    println!();

    // Example 4: Point location
    println!("Example 4: Point location efficiency");
    let grid_size = 10;
    let mut found_count = 0;

    // Test points on a grid
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = i as f64 / grid_size as f64;
            let y = j as f64 / grid_size as f64;

            if tri_random.find_simplex(&[x, y]).is_some() {
                found_count += 1;
            }
        }
    }

    println!("  Tested {} points", grid_size * grid_size);
    println!("  Found containing triangle for {} points", found_count);
    println!();
}
