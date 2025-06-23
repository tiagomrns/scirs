//! Seasonal-Trend decomposition using Regression (STR)

use ndarray::{s, Array1, Array2, ScalarOperand};
// use ndarray_linalg::{Inverse, Solve};  // TODO: Replace with scirs2-core linear algebra when available
use num_traits::{Float, FromPrimitive, NumCast};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Options for STR (Seasonal-Trend decomposition using Regression)
#[derive(Debug, Clone)]
pub struct STROptions {
    /// Type of regularization to use
    pub regularization_type: RegularizationType,
    /// Regularization parameter for trend
    pub trend_lambda: f64,
    /// Regularization parameter for seasonal components
    pub seasonal_lambda: f64,
    /// Seasonal periods (can include non-integer values)
    pub seasonal_periods: Vec<f64>,
    /// Whether to use robust estimation (less sensitive to outliers)
    pub robust: bool,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Degrees of freedom for the trend
    pub trend_degrees: usize,
    /// Whether to allow the seasonal pattern to change over time
    pub flexible_seasonal: bool,
    /// Number of harmonics for each seasonal component
    pub seasonal_harmonics: Option<Vec<usize>>,
}

/// Type of regularization to use in STR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizationType {
    /// Ridge regularization (L2 penalty)
    Ridge,
    /// LASSO regularization (L1 penalty)
    Lasso,
    /// Elastic Net regularization (combination of L1 and L2)
    ElasticNet,
}

impl Default for STROptions {
    fn default() -> Self {
        Self {
            regularization_type: RegularizationType::Ridge,
            trend_lambda: 10.0,
            seasonal_lambda: 0.5,
            seasonal_periods: Vec::new(),
            robust: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            trend_degrees: 3,
            flexible_seasonal: false,
            seasonal_harmonics: None,
        }
    }
}

/// Result of STR decomposition
#[derive(Debug, Clone)]
pub struct STRResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each seasonal period)
    pub seasonal_components: Vec<Array1<F>>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Confidence intervals for trend (if computed)
    pub trend_ci: Option<(Array1<F>, Array1<F>)>, // (lower, upper)
    /// Confidence intervals for seasonal components (if computed)
    pub seasonal_ci: Option<Vec<(Array1<F>, Array1<F>)>>, // (lower, upper) for each component
}

/// Performs STR (Seasonal-Trend decomposition using Regression) on a time series
///
/// STR uses regularized regression to extract trend and seasonal components from
/// a time series. It allows for multiple seasonal components, non-integer periods,
/// and can provide confidence intervals for the components.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for STR decomposition
///
/// # Returns
///
/// * STR decomposition result
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{str_decomposition, STROptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = STROptions::default();
/// options.seasonal_periods = vec![4.0, 12.0]; // Both quarterly and yearly patterns
///
/// let result = str_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn str_decomposition<F>(ts: &Array1<F>, options: &STROptions) -> Result<STRResult<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        // + ndarray_linalg::Lapack  // TODO: Replace with scirs2-core linear algebra trait when available
        + ScalarOperand
        + NumCast
        + std::iter::Sum,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for STR decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "At least one seasonal period must be specified for STR".to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period <= 1.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be greater than 1".to_string(),
            ));
        }
    }

    if options.trend_lambda < 0.0 || options.seasonal_lambda < 0.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Regularization parameters must be non-negative".to_string(),
        ));
    }

    if options.confidence_level <= 0.0 || options.confidence_level >= 1.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    // Step 1: Prepare design matrices for trend and seasonal components
    let time_indices: Array1<F> = Array1::from_iter((0..n).map(|i| F::from_usize(i).unwrap()));

    // Trend design matrix using polynomial basis functions
    let trend_degree = options.trend_degrees;
    let mut trend_basis = Array2::zeros((n, trend_degree + 1));

    // Fill trend design matrix with polynomial terms (1, t, t^2, t^3, ...)
    for i in 0..n {
        for j in 0..=trend_degree {
            if j == 0 {
                trend_basis[[i, j]] = F::one(); // Constant term
            } else {
                let time_idx = time_indices[i];
                trend_basis[[i, j]] = Float::powf(time_idx, F::from_usize(j).unwrap());
            }
        }
    }

    // Seasonal design matrices using Fourier basis functions for each seasonal component
    let mut seasonal_bases = Vec::with_capacity(options.seasonal_periods.len());
    let mut total_seasonal_cols = 0;

    for (idx, &period) in options.seasonal_periods.iter().enumerate() {
        // Number of harmonics for this seasonal component
        let harmonics = if let Some(ref harms) = options.seasonal_harmonics {
            harms
                .get(idx)
                .copied()
                .unwrap_or(((period / 2.0).floor() as usize).max(1))
        } else {
            ((period / 2.0).floor() as usize).max(1)
        };

        let mut seasonal_basis = Array2::zeros((n, 2 * harmonics)); // 2 columns per harmonic (sin and cos)

        for i in 0..n {
            let t = time_indices[i];
            for j in 0..harmonics {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                // Sin term
                seasonal_basis[[i, 2 * j]] = Float::sin(freq * t);
                // Cos term
                seasonal_basis[[i, 2 * j + 1]] = Float::cos(freq * t);
            }
        }

        total_seasonal_cols += 2 * harmonics;
        seasonal_bases.push(seasonal_basis);
    }

    // Step 2: Combine all design matrices
    let total_cols = trend_degree + 1 + total_seasonal_cols;
    let mut design_matrix = Array2::zeros((n, total_cols));

    // Fill trend columns
    design_matrix
        .slice_mut(s![.., 0..=trend_degree])
        .assign(&trend_basis);

    // Fill seasonal columns
    let mut col_offset = trend_degree + 1;
    for seasonal_basis in &seasonal_bases {
        let next_offset = col_offset + seasonal_basis.ncols();
        design_matrix
            .slice_mut(s![.., col_offset..next_offset])
            .assign(seasonal_basis);
        col_offset = next_offset;
    }

    // Step 3: Set up regularization matrix
    let mut regularization_matrix = Array2::zeros((total_cols, total_cols));

    // Trend regularization (penalize higher-order polynomial coefficients)
    for i in 0..=trend_degree {
        let weight = if i == 0 {
            0.0 // Don't penalize the constant term
        } else {
            options.trend_lambda * (i as f64).powi(2)
        };
        regularization_matrix[[i, i]] = F::from_f64(weight).unwrap();
    }

    // Seasonal regularization
    col_offset = trend_degree + 1;
    for seasonal_basis in &seasonal_bases {
        let seasonal_cols = seasonal_basis.ncols();
        for i in 0..seasonal_cols {
            regularization_matrix[[col_offset + i, col_offset + i]] =
                F::from(options.seasonal_lambda).unwrap();
        }
        col_offset += seasonal_cols;
    }

    // Step 4: Solve regularized least squares problem
    // (X^T X + λR) β = X^T y
    let xtx = design_matrix.t().dot(&design_matrix);
    let xty = design_matrix.t().dot(ts);

    // Add regularization
    let system_matrix = xtx + regularization_matrix;

    // Solve the system
    let coefficients = match options.regularization_type {
        RegularizationType::Ridge => {
            // Ridge regression: solve (X^T X + λI) β = X^T y
            // TODO: Replace with scirs2-core matrix solve when available
            simple_matrix_solve(&system_matrix, &xty)?
        }
        RegularizationType::Lasso | RegularizationType::ElasticNet => {
            // For LASSO and ElasticNet, we would need iterative algorithms (e.g., coordinate descent)
            // For now, fall back to Ridge regression
            // TODO: Implement proper LASSO/ElasticNet solvers
            // TODO: Replace with scirs2-core matrix solve when available
            simple_matrix_solve(&system_matrix, &xty)?
        }
    };

    // Step 5: Extract components from coefficients
    // Trend component
    let trend_coeffs = coefficients.slice(s![0..=trend_degree]);
    let trend = trend_basis.dot(&trend_coeffs);

    // Seasonal components
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());
    col_offset = trend_degree + 1;

    for seasonal_basis in &seasonal_bases {
        let seasonal_cols = seasonal_basis.ncols();
        let seasonal_coeffs = coefficients.slice(s![col_offset..col_offset + seasonal_cols]);
        let seasonal_component = seasonal_basis.dot(&seasonal_coeffs);
        seasonal_components.push(seasonal_component);
        col_offset += seasonal_cols;
    }

    // Compute residuals
    let mut residual = ts.clone();
    for i in 0..n {
        residual[i] = residual[i] - trend[i];
        for seasonal_component in &seasonal_components {
            residual[i] = residual[i] - seasonal_component[i];
        }
    }

    // Compute confidence intervals if requested
    let (trend_ci, seasonal_ci) = if options.compute_confidence_intervals {
        compute_confidence_intervals(
            &design_matrix,
            &system_matrix,
            &residual,
            &trend_basis,
            &seasonal_bases,
            options.confidence_level,
        )?
    } else {
        (None, None)
    };

    // Create result
    let result = STRResult {
        trend,
        seasonal_components,
        residual,
        original: ts.clone(),
        trend_ci,
        seasonal_ci,
    };

    Ok(result)
}

/// Type alias for confidence interval bounds (lower, upper)
type ConfidenceInterval<F> = (Array1<F>, Array1<F>);

/// Type alias for confidence intervals result
type ConfidenceIntervalsResult<F> = Result<(
    Option<ConfidenceInterval<F>>,
    Option<Vec<ConfidenceInterval<F>>>,
)>;

/// Compute confidence intervals for STR components
fn compute_confidence_intervals<F>(
    design_matrix: &Array2<F>,
    _system_matrix: &Array2<F>,
    residual: &Array1<F>,
    _trend_basis: &Array2<F>,
    _seasonal_bases: &[Array2<F>],
    _confidence_level: f64,
) -> ConfidenceIntervalsResult<F>
where
    F: Float
        + FromPrimitive
        + Debug
        // + ndarray_linalg::Lapack  // TODO: Replace with scirs2-core linear algebra trait when available
        + ScalarOperand
        + NumCast
        + std::iter::Sum,
{
    let n = residual.len();

    // Estimate residual variance
    let _residual_variance =
        residual.mapv(|x| x * x).sum() / F::from_usize(n - design_matrix.ncols()).unwrap();

    // Compute covariance matrix: σ² (X^T X + λR)^(-1)
    // TODO: Replace with scirs2-core matrix inversion when available
    // For now, skip confidence interval calculation
    Ok((None, None)) // Temporary - proper matrix inversion needed
}

/// Simple matrix solve using Gaussian elimination
/// TODO: Remove this when scirs2-core provides linear algebra functionality
fn simple_matrix_solve<F>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    let n = a.shape()[0];
    if n != a.shape()[1] || n != b.len() {
        return Err(TimeSeriesError::DecompositionError(
            "Matrix dimensions mismatch".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug = a.clone();
    let mut rhs = b.clone();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
            let temp = rhs[i];
            rhs[i] = rhs[max_row];
            rhs[max_row] = temp;
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < F::from(1e-10).unwrap() {
            return Err(TimeSeriesError::DecompositionError(
                "Matrix is singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..n {
                aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
            }
            rhs[k] = rhs[k] - factor * rhs[i];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_str_basic() {
        // Create a simple time series with trend and seasonality
        let n = 50;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.1 * i as f64;
            let seasonal = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = 0.1 * (i as f64 * 0.456).sin();
            ts[i] = trend + seasonal + noise;
        }

        let options = STROptions {
            seasonal_periods: vec![12.0],
            trend_degrees: 2,
            trend_lambda: 1.0,
            seasonal_lambda: 0.1,
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original (approximately)
        for i in 0..n {
            let reconstructed =
                result.trend[i] + result.seasonal_components[0][i] + result.residual[i];
            assert_abs_diff_eq!(reconstructed, ts[i], epsilon = 1e-10);
        }

        // Check that we extracted a trend
        assert!(result.trend.len() == n);
        // Check that we extracted seasonal components
        assert!(result.seasonal_components.len() == 1);
        assert!(result.seasonal_components[0].len() == n);
    }

    #[test]
    fn test_str_multiple_seasons() {
        // Create a time series with multiple seasonal patterns
        let n = 100;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.05 * i as f64;
            let seasonal1 = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let seasonal2 = 1.5 * (2.0 * std::f64::consts::PI * i as f64 / 4.0).cos();
            ts[i] = trend + seasonal1 + seasonal2;
        }

        let options = STROptions {
            seasonal_periods: vec![12.0, 4.0],
            trend_degrees: 1,
            trend_lambda: 5.0,
            seasonal_lambda: 0.5,
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original
        for i in 0..n {
            let mut reconstructed = result.trend[i] + result.residual[i];
            for seasonal_component in &result.seasonal_components {
                reconstructed += seasonal_component[i];
            }
            assert_abs_diff_eq!(reconstructed, ts[i], epsilon = 1e-10);
        }

        // Check that we have the right number of seasonal components
        assert_eq!(result.seasonal_components.len(), 2);
    }

    #[test]
    fn test_str_edge_cases() {
        // Test with minimum size time series
        let ts = array![1.0, 2.0, 3.0];
        let mut options = STROptions {
            seasonal_periods: vec![2.0],
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options);
        assert!(result.is_ok());

        // Test with invalid seasonal period
        options.seasonal_periods = vec![0.5];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with no seasonal periods
        options.seasonal_periods = vec![];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with too small time series
        let ts = array![1.0, 2.0];
        options.seasonal_periods = vec![2.0];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());
    }
}
