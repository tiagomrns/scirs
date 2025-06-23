use ndarray::array;
use scirs2_stats::linear_regression;

fn main() {
    // Create a simple design matrix with an intercept column
    let x = ndarray::Array2::from_shape_vec((5, 3), vec![
        1.0, 0.0, 1.0,   // 5 observations with 3 variables (intercept, x1, x2)
        1.0, 1.0, 2.0,
        1.0, 2.0, 3.0,
        1.0, 3.0, 4.0,
        1.0, 4.0, 5.0,
    ]).unwrap();

    // Target values: y = 1 + 2*x1 + 3*x2
    let y = array![4.0, 9.0, 14.0, 19.0, 24.0];

    // Perform regression analysis
    match linear_regression(&x.view(), &y.view(), None) {
        Ok(results) => {
            println!("Linear Regression Results:");
            println!("Coefficients: {:?}", results.coefficients);
            println!("R-squared: {}", results.r_squared);
            println!("Adjusted R-squared: {}", results.adj_r_squared);
            
            // Verify the coefficients are close to expected values
            assert!((results.coefficients[0] - 1.0).abs() < 1e-6);  // intercept
            assert!((results.coefficients[1] - 2.0).abs() < 1e-6);  // x1 coefficient
            assert!((results.coefficients[2] - 3.0).abs() < 1e-6);  // x2 coefficient
            
            println!("\nTest passed! The migration from ndarray-linalg to scirs2-linalg was successful.");
        }
        Err(e) => {
            eprintln!("Error performing regression: {:?}", e);
            std::process::exit(1);
        }
    }
}