//! Correlation measures
//!
//! This module provides functions for computing various correlation coefficients
//! between datasets, including Pearson, Spearman, and Kendall tau correlation.

use crate::error::{StatsError, StatsResult};
use crate::{mean, std};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use num_traits::{Float, NumCast};

/// Compute the Pearson correlation coefficient between two arrays.
///
/// The Pearson correlation coefficient measures the linear correlation between two
/// datasets. It ranges from -1 (perfect negative correlation) to 1 (perfect positive
/// correlation), with 0 indicating no linear correlation.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// The Pearson correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::pearson_r;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// let corr = pearson_r(&x.view(), &y.view()).unwrap();
/// println!("Pearson correlation: {}", corr);
/// // This should be a perfect negative correlation, approximately -1.0
/// assert!((corr - (-1.0f64)).abs() < 1e-10f64);
/// ```
pub fn pearson_r<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F>,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Check that arrays have the same length
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Check that arrays are not empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    // Calculate means
    let mean_x = mean(&x.view())?;
    let mean_y = mean(&y.view())?;

    // Calculate intermediate sums
    let mut sum_xy = F::zero();
    let mut sum_x2 = F::zero();
    let mut sum_y2 = F::zero();

    for i in 0..x.len() {
        let x_dev = x[i] - mean_x;
        let y_dev = y[i] - mean_y;

        sum_xy = sum_xy + x_dev * y_dev;
        sum_x2 = sum_x2 + x_dev * x_dev;
        sum_y2 = sum_y2 + y_dev * y_dev;
    }

    // Check for zero variances
    if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute correlation when one or both variables have zero variance".to_string(),
        ));
    }

    // Calculate correlation coefficient
    let corr = sum_xy / (sum_x2 * sum_y2).sqrt();

    // Ensure correlation is in the range [-1, 1]
    let corr = if corr > F::one() {
        F::one()
    } else if corr < -F::one() {
        -F::one()
    } else {
        corr
    };

    Ok(corr)
}

/// Compute the Spearman rank correlation coefficient between two arrays.
///
/// The Spearman correlation coefficient is the Pearson correlation coefficient
/// computed on the ranks of the data. It assesses how well the relationship
/// between two variables can be described using a monotonic function,
/// making it robust to outliers and non-linear relationships.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// The Spearman rank correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::spearman_r;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0];  // Squared values - non-linear relationship
///
/// let corr = spearman_r(&x.view(), &y.view()).unwrap();
/// println!("Spearman correlation: {}", corr);
/// // This should be a perfect monotonic relationship, approximately 1.0
/// assert!((corr - 1.0f64).abs() < 1e-10f64);
/// ```
pub fn spearman_r<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F>,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Check that arrays have the same length
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Check that arrays are not empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    // Create vectors of (value, original index) pairs for ranking
    let mut x_idx = vec![];
    let mut y_idx = vec![];

    for (i, (&x_val, &y_val)) in x.iter().zip(y.iter()).enumerate() {
        x_idx.push((x_val, i));
        y_idx.push((y_val, i));
    }

    // Sort by value
    x_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    y_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate ranks, handling ties
    let mut x_ranks = vec![F::zero(); x.len()];
    let mut y_ranks = vec![F::zero(); y.len()];

    assign_ranks(&x_idx, &mut x_ranks)?;
    assign_ranks(&y_idx, &mut y_ranks)?;

    // Convert ranks to arrays
    let x_ranks = ndarray::Array1::from(x_ranks);
    let y_ranks = ndarray::Array1::from(y_ranks);

    // Calculate Pearson correlation of the ranks
    pearson_r::<F, _>(&x_ranks.view(), &y_ranks.view())
}

/// Helper function to assign ranks to sorted data
fn assign_ranks<F: Float>(sorted_data: &[(F, usize)], ranks: &mut [F]) -> StatsResult<()> {
    let n = sorted_data.len();

    let mut i = 0;
    while i < n {
        let current_val = sorted_data[i].0;
        let mut j = i;

        // Find the end of the tie group
        while j < n - 1 && sorted_data[j + 1].0 == current_val {
            j += 1;
        }

        // Calculate average rank for this tie group
        let avg_rank = F::from((i + j) as f64 / 2.0 + 1.0).unwrap();

        // Assign average rank to all tied values
        for item in sorted_data.iter().take(j + 1).skip(i) {
            let original_idx = item.1;
            ranks[original_idx] = avg_rank;
        }

        i = j + 1;
    }

    Ok(())
}

/// Compute the Kendall tau correlation coefficient between two arrays.
///
/// The Kendall tau correlation coefficient measures the ordinal association
/// between two datasets. It's based on the number of concordant and discordant
/// pairs and is a non-parametric measure of correlation.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `method` - Optional method specification: "b" (default) or "c"
///
/// # Returns
///
/// The Kendall tau correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::kendall_tau;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
/// println!("Kendall tau correlation: {}", corr);
/// // This should be a perfect negative rank correlation, approximately -1.0
/// assert!((corr - (-1.0f64)).abs() < 1e-10f64);
/// ```
pub fn kendall_tau<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    method: &str,
) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F>,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Validate method parameter
    if method != "b" && method != "c" {
        return Err(StatsError::InvalidArgument(format!(
            "Method must be 'b' or 'c', got {}",
            method
        )));
    }

    // Check that arrays have the same length
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Check that arrays are not empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    // Compute concordant and discordant pairs
    let mut concordant = 0;
    let mut discordant = 0;
    let mut ties_x = 0;
    let mut ties_y = 0;
    let mut _ties_xy = 0;

    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let x_diff = x[j] - x[i];
            let y_diff = y[j] - y[i];

            if x_diff.is_zero() && y_diff.is_zero() {
                _ties_xy += 1;
            } else if x_diff.is_zero() {
                ties_x += 1;
            } else if y_diff.is_zero() {
                ties_y += 1;
            } else if (x_diff > F::zero() && y_diff > F::zero())
                || (x_diff < F::zero() && y_diff < F::zero())
            {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    // Calculate denominator based on method
    let n = x.len();
    let _n0 = F::from(n * (n - 1) / 2).unwrap();

    let tau = match method {
        "b" => {
            // Kendall's tau-b (accounts for ties)
            let n1 = F::from(concordant + discordant + ties_x).unwrap();
            let n2 = F::from(concordant + discordant + ties_y).unwrap();

            if n1 == F::zero() || n2 == F::zero() {
                return Err(StatsError::InvalidArgument(
                    "Cannot compute Kendall's tau-b for case with all values tied in one variable"
                        .to_string(),
                ));
            }

            F::from(concordant - discordant).unwrap() / (n1 * n2).sqrt()
        }
        "c" => {
            // Kendall's tau-c (more suitable for rectangular tables)
            let m = F::from(n.min(2)).unwrap();
            (F::from(2.0).unwrap() * m * F::from(concordant - discordant).unwrap())
                / (F::from(n).unwrap().powi(2) * (m - F::one()))
        }
        _ => unreachable!(), // We validated the method parameter earlier
    };

    Ok(tau)
}

/// Compute the partial correlation coefficient between two variables,
/// controlling for one or more additional variables.
///
/// Partial correlation measures the relationship between two variables
/// after controlling for the effects of one or more other variables.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `z` - Array of control variables (each column is a control variable)
///
/// # Returns
///
/// The partial correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2, Axis};
/// use scirs2_stats::partial_corr;
///
/// // Create sample data
/// let x = array![10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
/// let y = array![8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];
///
/// // Control variable
/// let z1 = array![7.0, 5.0, 8.0, 7.0, 8.0, 7.0, 5.0, 3.0, 9.0, 4.0, 6.0];
/// let z = Array2::from_shape_vec((11, 1), z1.to_vec()).unwrap();
///
/// let partial_r = partial_corr(&x.view(), &y.view(), &z.view()).unwrap();
/// println!("Partial correlation: {}", partial_r);
/// ```
pub fn partial_corr<F, D1, D2>(
    x: &ArrayBase<D1, Ix1>,
    y: &ArrayBase<D1, Ix1>,
    z: &ArrayBase<D2, Ix2>,
) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + 'static,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
    Ix1: Dimension,
    Ix2: Dimension,
{
    // Check that arrays have the same length
    if x.len() != y.len() || x.len() != z.shape()[0] {
        return Err(StatsError::DimensionMismatch(
            "All arrays must have the same length".to_string(),
        ));
    }

    // Check that arrays are not empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    // First, compute residuals by regressing out the control variables
    let x_resid = compute_residuals(x, z)?;
    let y_resid = compute_residuals(y, z)?;

    // Then calculate the correlation between residuals
    pearson_r::<F, _>(&x_resid.view(), &y_resid.view())
}

/// Helper function to compute residuals by regressing one variable on control variables
fn compute_residuals<F, D1, D2>(
    y: &ArrayBase<D1, Ix1>,
    x: &ArrayBase<D2, Ix2>,
) -> StatsResult<ndarray::Array1<F>>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + 'static,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    // Get dimensions
    let n = y.len();
    let p = x.shape()[1];

    // Add constant term to the predictors
    let mut x_with_const = Array2::<F>::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            x_with_const[[i, j + 1]] = x[[i, j]];
        }
    }

    // Compute beta using least squares (X'X)^(-1)X'y
    // Note: In a real implementation, QR decomposition or similar should be used
    // Here we use a simpler but less numerically stable approach for demonstration

    // X'X
    let xtx = x_with_const.t().dot(&x_with_const);

    // X'y
    let mut xty = Array1::<F>::zeros(p + 1);
    for j in 0..p + 1 {
        for i in 0..n {
            xty[j] = xty[j] + x_with_const[[i, j]] * y[i];
        }
    }

    // Solve for beta
    // For simplicity, we're using a basic algorithm here
    let beta = simple_linear_solve(&xtx, &xty)?;

    // Compute fitted values
    let mut y_hat = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut pred = F::zero();
        for j in 0..p + 1 {
            pred = pred + x_with_const[[i, j]] * beta[j];
        }
        y_hat[i] = pred;
    }

    // Compute residuals
    let mut residuals = Array1::<F>::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - y_hat[i];
    }

    Ok(residuals)
}

/// Simple linear equation solver using Gaussian elimination for demonstration purposes
/// This should be replaced with a better implementation in production code
fn simple_linear_solve<F>(
    a: &ndarray::Array2<F>,
    b: &ndarray::Array1<F>,
) -> StatsResult<ndarray::Array1<F>>
where
    F: Float + std::fmt::Debug + NumCast + 'static,
{
    let n = a.shape()[0];

    // Basic check for square matrix
    if a.shape()[0] != a.shape()[1] {
        return Err(StatsError::InvalidArgument(
            "Coefficient matrix must be square".to_string(),
        ));
    }

    // Basic check for dimensions
    if a.shape()[0] != b.len() {
        return Err(StatsError::DimensionMismatch(
            "Coefficient matrix and RHS vector dimensions must match".to_string(),
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for j in (i + 1)..n {
            if aug[[j, i]].abs() > aug[[max_row, i]].abs() {
                max_row = j;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() <= F::epsilon() {
            return Err(StatsError::InvalidArgument(
                "Coefficient matrix is singular".to_string(),
            ));
        }

        // Eliminate below
        for j in (i + 1)..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            for k in i..=n {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = ndarray::Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum = sum + aug[[i, j]] * x[j];
        }
        x[i] = (aug[[i, n]] - sum) / aug[[i, i]];
    }

    Ok(x)
}

/// Compute the point-biserial correlation coefficient between a binary variable and a continuous variable.
///
/// The point-biserial correlation is the Pearson correlation coefficient when one variable is binary.
///
/// # Arguments
///
/// * `binary` - Binary variable (should only contain 0 and 1)
/// * `continuous` - Continuous variable
///
/// # Returns
///
/// The point-biserial correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::point_biserial;
///
/// let binary = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let continuous = array![2.5, 4.5, 3.2, 5.1, 2.0, 4.8, 2.8, 5.5];
///
/// let corr = point_biserial(&binary.view(), &continuous.view()).unwrap();
/// println!("Point-biserial correlation: {}", corr);
/// ```
pub fn point_biserial<F, D>(
    binary: &ArrayBase<D, Ix1>,
    continuous: &ArrayBase<D, Ix1>,
) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F>,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Check that arrays have the same length
    if binary.len() != continuous.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Check that arrays are not empty
    if binary.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    // Verify that binary variable contains only 0 and 1
    for &val in binary.iter() {
        if val != F::zero() && val != F::one() {
            return Err(StatsError::InvalidArgument(
                "Binary variable must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Count number of 1s and 0s
    let mut n1 = 0;
    let mut n0 = 0;

    for &val in binary.iter() {
        if val == F::one() {
            n1 += 1;
        } else {
            n0 += 1;
        }
    }

    // Handle case where all values are the same
    if n1 == 0 || n0 == 0 {
        return Err(StatsError::InvalidArgument(
            "Binary variable must have at least one 0 and one 1".to_string(),
        ));
    }

    // Calculate means for each group
    let mut mean1 = F::zero();
    let mut mean0 = F::zero();

    for i in 0..binary.len() {
        if binary[i] == F::one() {
            mean1 = mean1 + continuous[i];
        } else {
            mean0 = mean0 + continuous[i];
        }
    }

    mean1 = mean1 / F::from(n1).unwrap();
    mean0 = mean0 / F::from(n0).unwrap();

    // Calculate standard deviation of continuous variable
    let std_y = std(&continuous.view(), 0)?;

    // Calculate point-biserial correlation
    let n = F::from(binary.len()).unwrap();
    let n1_f = F::from(n1).unwrap();
    let n0_f = F::from(n0).unwrap();

    let corr = ((mean1 - mean0) / std_y) * (n1_f * n0_f / (n * n)).sqrt();

    Ok(corr)
}

/// Compute a correlation matrix for a set of variables.
///
/// # Arguments
///
/// * `data` - 2D array where each column is a variable
/// * `method` - Correlation method: "pearson" (default), "spearman", or "kendall"
///
/// # Returns
///
/// A correlation matrix where element [i, j] is the correlation between variable i and variable j
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::corrcoef;
///
/// let data = array![
///     [1.0, 5.0, 10.0],
///     [2.0, 4.0, 9.0],
///     [3.0, 3.0, 8.0],
///     [4.0, 2.0, 7.0],
///     [5.0, 1.0, 6.0]
/// ];
///
/// let corr_matrix = corrcoef(&data.view(), "pearson").unwrap();
/// println!("Correlation matrix:\n{:?}", corr_matrix);
/// ```
pub fn corrcoef<F, D>(data: &ArrayBase<D, Ix2>, method: &str) -> StatsResult<ndarray::Array2<F>>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F>,
    D: Data<Elem = F>,
    Ix2: Dimension,
{
    // Validate method parameter
    match method {
        "pearson" | "spearman" | "kendall" => {}
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Method must be 'pearson', 'spearman', or 'kendall', got {}",
                method
            )))
        }
    }

    // Get dimensions
    let (n, p) = (data.shape()[0], data.shape()[1]);

    // Check that data is not empty
    if n == 0 || p == 0 {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    // Initialize correlation matrix
    let mut corr_mat = Array2::<F>::zeros((p, p));

    // Compute correlation for each pair of variables
    for i in 0..p {
        // Diagonal elements are 1
        corr_mat[[i, i]] = F::one();

        for j in (i + 1)..p {
            // Get columns i and j
            let var_i = data.slice(s![.., i]);
            let var_j = data.slice(s![.., j]);

            // Calculate correlation based on chosen method
            let corr = match method {
                "pearson" => pearson_r::<F, _>(&var_i, &var_j)?,
                "spearman" => spearman_r::<F, _>(&var_i, &var_j)?,
                "kendall" => kendall_tau::<F, _>(&var_i, &var_j, "b")?,
                _ => unreachable!(), // Already validated method above
            };

            // Correlation matrix is symmetric
            corr_mat[[i, j]] = corr;
            corr_mat[[j, i]] = corr;
        }
    }

    Ok(corr_mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative correlation
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // No correlation
        let y = array![5.0, 2.0, 8.0, 1.0, 9.0];

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert!(corr.abs() < 0.5); // Weak correlation
    }

    #[test]
    fn test_spearman_correlation() {
        // Perfect monotonic relationship (but not linear)
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // Squared values

        let corr = spearman_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative monotonic relationship
        let y = array![25.0, 16.0, 9.0, 4.0, 1.0];

        let corr = spearman_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall_tau() {
        // Perfect concordant ordering
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 13.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect discordant ordering
        let y = array![14.0, 13.0, 12.0, 11.0, 10.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // With ties
        let x = array![1.0, 2.0, 3.0, 3.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 12.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
        assert!(corr > 0.9); // Strong positive correlation but not exactly 1.0 due to ties
    }

    #[test]
    fn test_correlation_matrix() {
        // Create sample data
        let data = array![
            [1.0, 5.0, 10.0],
            [2.0, 4.0, 9.0],
            [3.0, 3.0, 8.0],
            [4.0, 2.0, 7.0],
            [5.0, 1.0, 6.0]
        ];

        // Calculate correlation matrix using Pearson correlation
        let corr_mat = corrcoef(&data.view(), "pearson").unwrap();

        // Diagonal elements should be 1.0
        assert_abs_diff_eq!(corr_mat[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[2, 2]], 1.0, epsilon = 1e-10);

        // Check symmetry
        assert_abs_diff_eq!(corr_mat[[0, 1]], corr_mat[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[0, 2]], corr_mat[[2, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[1, 2]], corr_mat[[2, 1]], epsilon = 1e-10);

        // Known values for this dataset
        assert_abs_diff_eq!(corr_mat[[0, 1]], -1.0, epsilon = 1e-10); // Perfect negative correlation
        assert_abs_diff_eq!(corr_mat[[0, 2]], -1.0, epsilon = 1e-10); // Perfect negative correlation
    }
}
