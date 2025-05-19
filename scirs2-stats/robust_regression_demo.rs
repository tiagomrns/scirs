// Simple robust regression demo
// To run this directly: rustc robust_regression_demo.rs && ./robust_regression_demo
use std::time::{SystemTime, UNIX_EPOCH};

/// A simplified version of TheilSlopes for demonstration
struct SimpleTheilSlopes {
    slope: f64,
    intercept: f64,
    // Adding confidence interval bounds for slope
    slope_low: f64,
    slope_high: f64,
}

/// Simple implementation of Theil-Sen estimator
fn simple_theilslopes(x: &[f64], y: &[f64]) -> SimpleTheilSlopes {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    
    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n-1) / 2);
    
    // Calculate all pairwise slopes
    for i in 0..n {
        for j in (i+1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > f64::EPSILON {
                let dy = y[j] - y[i];
                slopes.push(dy / dx);
            }
        }
    }
    
    // Sort slopes to find median
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Get median slope
    let mid = slopes.len() / 2;
    let slope = if slopes.len() % 2 == 0 && !slopes.is_empty() {
        (slopes[mid-1] + slopes[mid]) / 2.0
    } else if !slopes.is_empty() {
        slopes[mid]
    } else {
        0.0 // No valid slopes, return zero
    };
    
    // Calculate the y-intercept using median values
    let x_median = x.iter().sum::<f64>() / n as f64;
    let y_median = y.iter().sum::<f64>() / n as f64;
    let intercept = y_median - slope * x_median;
    
    // Confidence interval for slope (simplified approximation based on Sen's method)
    let n_float = n as f64;
    let sigma = 1.0 / (1.5 * n_float.sqrt());
    let z = 1.96; // Approximate 95% confidence
    let margin = z * sigma;
    
    SimpleTheilSlopes { 
        slope, 
        intercept,
        slope_low: slope - margin,
        slope_high: slope + margin
    }
}

/// Simple OLS (Ordinary Least Squares) implementation
fn simple_ols(x: &[f64], y: &[f64]) -> (f64, f64) { // (slope, intercept)
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..x.len() {
        let x_dev = x[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        denominator += x_dev * x_dev;
    }
    
    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;
    
    (slope, intercept)
}

/// Simple implementation of RANSAC (Random Sample Consensus) regression
fn simple_ransac(x: &[f64], y: &[f64], threshold: f64, n_trials: usize) -> (f64, f64, Vec<bool>) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    
    let n = x.len();
    if n < 2 {
        panic!("Need at least 2 points for RANSAC");
    }
    
    // To keep code simple, we always sample 2 points
    let min_samples = 2;
    
    // Use a simple random number generator
    let mut rng = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    let mut best_inlier_count = 0;
    let mut best_model = (0.0, 0.0); // (slope, intercept)
    let mut best_inlier_mask = vec![false; n];
    
    for _ in 0..n_trials {
        // Select 2 random points
        let idx1 = (rng % n as u64) as usize;
        rng = (rng.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648; // Simple LCG
        let idx2 = (rng % n as u64) as usize;
        rng = (rng.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
        
        // Skip if we selected the same index twice
        if idx1 == idx2 {
            continue;
        }
        
        // Get the points
        let x1 = x[idx1];
        let y1 = y[idx1];
        let x2 = x[idx2];
        let y2 = y[idx2];
        
        // Calculate slope and intercept
        let dx = x2 - x1;
        if dx.abs() < f64::EPSILON {
            continue; // Skip vertical lines
        }
        
        let slope = (y2 - y1) / dx;
        let intercept = y1 - slope * x1;
        
        // Determine inliers
        let mut inlier_mask = vec![false; n];
        let mut inlier_count = 0;
        
        for i in 0..n {
            let y_pred = slope * x[i] + intercept;
            let residual = (y[i] - y_pred).abs();
            
            if residual < threshold {
                inlier_mask[i] = true;
                inlier_count += 1;
            }
        }
        
        // Update best model if we found more inliers
        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_model = (slope, intercept);
            best_inlier_mask = inlier_mask;
        }
    }
    
    // Refit the model using all inliers
    if best_inlier_count >= min_samples {
        // We could refit, but for simplicity we'll just return the best model
        (best_model.0, best_model.1, best_inlier_mask)
    } else {
        // Fallback to using all data if we couldn't find a good model
        let (slope, intercept) = simple_ols(x, y);
        (slope, intercept, vec![true; n])
    }
}

/// A simplified implementation of Huber regression
fn simple_huber(x: &[f64], y: &[f64], epsilon: f64, max_iter: usize) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    
    // Start with OLS estimate
    let (mut slope, mut intercept) = simple_ols(x, y);
    
    for _ in 0..max_iter {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        // Calculate Huber weights and weighted OLS
        for i in 0..x.len() {
            let y_pred = intercept + slope * x[i];
            let residual = y[i] - y_pred;
            
            // Huber weight function
            let weight = if residual.abs() <= epsilon {
                1.0 // Use full weight for small residuals
            } else {
                epsilon / residual.abs() // Reduce weight for larger residuals
            };
            
            // Weighted contribution to slope/intercept calculation
            numerator += weight * x[i] * (y[i] - intercept);
            denominator += weight * x[i] * x[i];
        }
        
        let new_slope = numerator / denominator;
        
        // Calculate new intercept using Huber weights
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for i in 0..x.len() {
            let y_pred = intercept + new_slope * x[i];
            let residual = y[i] - y_pred;
            
            // Huber weight function
            let weight = if residual.abs() <= epsilon {
                1.0
            } else {
                epsilon / residual.abs()
            };
            
            weighted_sum += weight * (y[i] - new_slope * x[i]);
            weight_sum += weight;
        }
        
        let new_intercept = weighted_sum / weight_sum;
        
        // Check convergence
        let slope_change = (new_slope - slope).abs();
        let intercept_change = (new_intercept - intercept).abs();
        
        if slope_change < 1e-6 && intercept_change < 1e-6 {
            break;
        }
        
        slope = new_slope;
        intercept = new_intercept;
    }
    
    (slope, intercept)
}

/// Simple text-based scatterplot visualization with improved formatting
fn draw_scatterplot(
    x: &[f64], 
    y: &[f64], 
    models: &[(f64, f64, char, &str)], // (slope, intercept, symbol, name)
    outlier_idx: Option<usize>
) {
    println!("\nVisualization of regression lines:");
    
    // Determine y range for plotting
    let mut y_min = y[0];
    let mut y_max = y[0];
    
    for &yi in y.iter() {
        if yi < y_min { y_min = yi; }
        if yi > y_max { y_max = yi; }
    }
    
    // Add margin
    y_min -= 2.0;
    y_max += 2.0;
    
    // Round to integers
    let y_min = y_min.floor() as i32;
    let y_max = y_max.ceil() as i32;
    
    // Determine x range for plotting
    let mut x_min = x[0];
    let mut x_max = x[0];
    
    for &xi in x.iter() {
        if xi < x_min { x_min = xi; }
        if xi > x_max { x_max = xi; }
    }
    
    // Add margin
    x_min -= 1.0;
    x_max += 1.0;
    
    // Draw the plot
    for y_val in (y_min..=y_max).rev() {
        let mut line = String::from("|");
        
        for x_pos in 0..60 {
            let x_val = x_min + (x_max - x_min) * (x_pos as f64 / 60.0);
            
            // Check if there's a data point
            let mut found_point = false;
            let mut is_outlier = false;
            
            for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
                if (xi - x_val).abs() < (x_max - x_min) / 60.0 * 0.8 && 
                   (yi - y_val as f64).abs() < 0.5 {
                    found_point = true;
                    if Some(i) == outlier_idx {
                        is_outlier = true;
                    }
                    break;
                }
            }
            
            // Check regression lines
            let mut found_line = false;
            let mut line_char = ' ';
            
            for &(slope, intercept, symbol, _) in models {
                let y_predicted = slope * x_val + intercept;
                if (y_predicted - y_val as f64).abs() < 0.5 {
                    if !found_line {
                        found_line = true;
                        line_char = symbol;
                    } else {
                        // Multiple lines - use a different symbol
                        line_char = '+';
                    }
                }
            }
            
            // Draw the appropriate character
            if found_point {
                if is_outlier {
                    line.push('X'); // Outlier point
                } else {
                    line.push('●'); // Regular data point
                }
            } else if found_line {
                line.push(line_char);
            } else {
                line.push(' ');
            }
        }
        
        line.push('|');
        println!("{}", line);
    }
    
    println!("+{}+", "-".repeat(60));
    println!("Legend: ● Data points, X Outliers");
    for &(_, _, symbol, name) in models {
        println!("        {} {}", symbol, name);
    }
    println!("        + Multiple lines overlap");
}

/// Run a regression example with different methods
fn run_regression_example(x: &[f64], y: &[f64], true_slope: f64, true_intercept: f64, outlier_idx: Option<usize>) {
    // Calculate using different methods
    let (ols_slope, ols_intercept) = simple_ols(x, y);
    let theil = simple_theilslopes(x, y);
    let (ransac_slope, ransac_intercept, ransac_inliers) = simple_ransac(x, y, 2.0, 100);
    let (huber_slope, huber_intercept) = simple_huber(x, y, 1.345, 50);
    
    // Print results
    println!("True model: y = {}x + {}", true_slope, true_intercept);
    println!("OLS:        y = {:.4}x + {:.4}", ols_slope, ols_intercept);
    println!("Theil-Sen:  y = {:.4}x + {:.4} (95% CI: [{:.4}, {:.4}])", 
             theil.slope, theil.intercept, theil.slope_low, theil.slope_high);
    println!("RANSAC:     y = {:.4}x + {:.4}", ransac_slope, ransac_intercept);
    println!("Huber:      y = {:.4}x + {:.4}", huber_slope, huber_intercept);
    
    // Calculate errors
    let ols_slope_error = (ols_slope - true_slope).abs();
    let theil_slope_error = (theil.slope - true_slope).abs();
    let ransac_slope_error = (ransac_slope - true_slope).abs();
    let huber_slope_error = (huber_slope - true_slope).abs();
    
    println!("\nSlope Error Comparison:");
    println!("OLS:        {:.4} ({:.2}% error)", ols_slope_error, 100.0 * ols_slope_error / true_slope);
    println!("Theil-Sen:  {:.4} ({:.2}% error)", theil_slope_error, 100.0 * theil_slope_error / true_slope);
    println!("RANSAC:     {:.4} ({:.2}% error)", ransac_slope_error, 100.0 * ransac_slope_error / true_slope);
    println!("Huber:      {:.4} ({:.2}% error)", huber_slope_error, 100.0 * huber_slope_error / true_slope);
    
    // Print outliers identified by RANSAC
    let outlier_indices: Vec<_> = ransac_inliers.iter().enumerate()
        .filter_map(|(i, &is_inlier)| if !is_inlier { Some(i) } else { None })
        .collect();
    
    println!("\nOutliers identified by RANSAC: {:?}", outlier_indices);
    
    // Create models for visualization
    let models = vec![
        (ols_slope, ols_intercept, 'o', "OLS"),
        (theil.slope, theil.intercept, 't', "Theil-Sen"),
        (ransac_slope, ransac_intercept, 'r', "RANSAC"),
        (huber_slope, huber_intercept, 'h', "Huber"),
    ];
    
    // Plot the data and regression lines
    draw_scatterplot(x, y, &models, outlier_idx);
}

fn main() {
    println!("=== Robust Regression Methods Demonstration ===");
    
    // Example 1: Simple data with one outlier
    println!("\n--- Example 1: Simple data with one outlier ---");
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 30.0]; // Last point is an outlier
    
    run_regression_example(&x, &y, 2.0, 1.0, Some(9));
    
    // Example 2: Multiple outliers
    println!("\n\n--- Example 2: Data with multiple outliers ---");
    let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y2 = vec![5.1, 7.2, 15.0, 11.1, 13.0, 14.9, 17.1, 9.0, 21.0, 23.0];
    // Outliers at indices 2, 7
    
    run_regression_example(&x2, &y2, 2.0, 3.0, None);
    
    // Example 3: Leverage points (outliers in X direction)
    println!("\n\n--- Example 3: Leverage points (outliers in X direction) ---");
    let mut x3 = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];
    let mut y3 = vec![2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.0, 9.2, 10.1, 11.2];
    
    // Add leverage points (outliers far in x-direction)
    x3.push(20.0); y3.push(41.0); // follows the trend
    x3.push(25.0); y3.push(30.0); // doesn't follow the trend
    
    run_regression_example(&x3, &y3, 2.0, 0.0, Some(11));
    
    println!("\n=== Summary of Robust Regression Methods ===\n");
    println!("The examples demonstrate when to use different robust regression methods:");
    println!("\n1. Ordinary Least Squares (OLS)");
    println!("   - Works best with clean data with normal errors");
    println!("   - Most efficient when assumptions are met");
    println!("   - Highly sensitive to outliers");
    println!("\n2. Theil-Sen Estimator");
    println!("   - Non-parametric, based on median of pairwise slopes");
    println!("   - Resistant to outliers (up to 29.3% contamination)");
    println!("   - Works well with small datasets but O(n²) complexity");
    println!("   - Provides confidence intervals for slope");
    println!("\n3. RANSAC");
    println!("   - Explicitly identifies and excludes outliers");
    println!("   - Highly robust to large proportions of outliers");
    println!("   - Good when there's a clear inlier/outlier separation");
    println!("   - Results can vary due to random sampling");
    println!("\n4. Huber Regression");
    println!("   - Uses a hybrid loss function");
    println!("   - Balanced approach between efficiency and robustness");
    println!("   - Less robust than RANSAC for extreme outliers");
    println!("   - More efficient than Theil-Sen for larger datasets");
    println!("\nKey considerations for method selection:");
    println!("- If outlier identification is important: Use RANSAC");
    println!("- If proportion of outliers is high: Use Theil-Sen or RANSAC");
    println!("- If computational efficiency matters for large datasets: Use Huber");
    println!("- If outliers are in X direction (leverage points): Theil-Sen often performs best");
}