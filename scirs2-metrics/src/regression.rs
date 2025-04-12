//! Regression metrics module
//!
//! This module provides functions for evaluating regression models, including
//! mean squared error, mean absolute error, R-squared, and explained variance.

use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};

use crate::error::{MetricsError, Result};

/// Calculates the mean squared error (MSE)
///
/// Mean squared error measures the average squared difference between
/// the estimated values and the actual value.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mse = mean_squared_error(&y_true, &y_pred).unwrap();
/// // Expecting: ((3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)²) / 4
/// assert!(mse < 0.38 && mse > 0.37);
/// ```
pub fn mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut squared_error_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = *yt - *yp;
        squared_error_sum = squared_error_sum + error * error;
    }

    Ok(squared_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the root mean squared error (RMSE)
///
/// Root mean squared error is the square root of the mean squared error.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The root mean squared error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::root_mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
/// // RMSE is the square root of MSE
/// assert!(rmse < 0.62 && rmse > 0.61);
/// ```
pub fn root_mean_squared_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Calculates the mean absolute error (MAE)
///
/// Mean absolute error measures the average absolute difference between
/// the estimated values and the actual value.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
/// // Expecting: (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4
/// assert!(mae < 0.51 && mae > 0.49);
/// ```
pub fn mean_absolute_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut absolute_error_sum = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        absolute_error_sum = absolute_error_sum + error;
    }

    Ok(absolute_error_sum / NumCast::from(n_samples).unwrap())
}

/// Calculates the mean absolute percentage error (MAPE)
///
/// Mean absolute percentage error measures the average percentage difference between
/// the estimated values and the actual value.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values (must not contain zeros)
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The mean absolute percentage error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::mean_absolute_percentage_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
/// ```
pub fn mean_absolute_percentage_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut percentage_error_sum = F::zero();
    let mut valid_samples = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        // Skip division by zero
        if yt.abs() < F::epsilon() {
            continue;
        }

        let percentage_error = ((*yt - *yp) / *yt).abs();
        percentage_error_sum = percentage_error_sum + percentage_error;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid samples for MAPE calculation (all y_true values are zero)".to_string(),
        ));
    }

    Ok(percentage_error_sum / NumCast::from(valid_samples).unwrap())
}

/// Calculates the R-squared score (coefficient of determination)
///
/// R-squared represents the proportion of variance in the dependent variable
/// that is predictable from the independent variable(s).
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The R-squared score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::r2_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let r2 = r2_score(&y_true, &y_pred).unwrap();
/// ```
pub fn r2_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate the mean of y_true manually
    let mut y_mean = F::zero();
    for &yt in y_true.iter() {
        y_mean = y_mean + yt;
    }
    y_mean = y_mean / NumCast::from(n_samples).unwrap();

    // Calculate the total sum of squares
    let mut ss_tot = F::zero();
    for &yt in y_true.iter() {
        let diff = yt - y_mean;
        ss_tot = ss_tot + diff * diff;
    }

    // If the total sum of squares is zero, then r2 is undefined
    if ss_tot == F::zero() {
        return Err(MetricsError::InvalidInput(
            "R^2 score is undefined when all y_true values are identical".to_string(),
        ));
    }

    // Calculate the residual sum of squares
    let mut ss_res = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let diff = *yt - *yp;
        ss_res = ss_res + diff * diff;
    }

    // Calculate R^2
    Ok(F::one() - ss_res / ss_tot)
}

/// Calculates the explained variance score
///
/// The explained variance score measures the proportion to which a model
/// accounts for the variation in the target variable.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The explained variance score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::explained_variance_score;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let score = explained_variance_score(&y_true, &y_pred).unwrap();
/// ```
pub fn explained_variance_score<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + FromPrimitive + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate means manually
    let mut y_true_mean = F::zero();
    let mut y_pred_mean = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        y_true_mean = y_true_mean + *yt;
        y_pred_mean = y_pred_mean + *yp;
    }

    y_true_mean = y_true_mean / NumCast::from(n_samples).unwrap();
    y_pred_mean = y_pred_mean / NumCast::from(n_samples).unwrap();

    // Calculate variances and covariance manually
    let mut y_true_var = F::zero();
    let mut cov = F::zero();

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let true_diff = *yt - y_true_mean;
        let pred_diff = *yp - y_pred_mean;

        y_true_var = y_true_var + true_diff * true_diff;
        cov = cov + true_diff * pred_diff;
    }

    y_true_var = y_true_var / NumCast::from(n_samples).unwrap();
    cov = cov / NumCast::from(n_samples).unwrap();

    // If the true variance is zero, then the score is undefined
    if y_true_var == F::zero() {
        return Err(MetricsError::InvalidInput(
            "Explained variance score is undefined when y_true has zero variance".to_string(),
        ));
    }

    // Calculate explained variance
    let score = F::one() - (y_true_var - cov) / y_true_var;

    // Clamp to [0, 1] range
    if score < F::zero() {
        Ok(F::zero())
    } else {
        Ok(score.min(F::one()))
    }
}

/// Calculates the maximum error between predicted and true values
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The maximum absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::max_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let max_err = max_error(&y_true, &y_pred).unwrap();
/// // Maximum error is |7.0 - 8.0| = 1.0
/// assert!(max_err < 1.01 && max_err > 0.99);
/// ```
pub fn max_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut max_err = F::zero();
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        if error > max_err {
            max_err = error;
        }
    }

    Ok(max_err)
}

/// Calculates the median absolute error
///
/// Median absolute error is particularly robust to outliers
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated target values
///
/// # Returns
///
/// * The median absolute error
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::regression::median_absolute_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.0, 8.0];
///
/// let med_ae = median_absolute_error(&y_true, &y_pred).unwrap();
/// // Absolute errors: [0.5, 0.5, 0.0, 1.0], median is 0.5
/// assert!(med_ae < 0.51 && med_ae > 0.49);
/// ```
pub fn median_absolute_error<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate absolute errors
    let mut abs_errors = Vec::with_capacity(n_samples);
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = (*yt - *yp).abs();
        abs_errors.push(error);
    }

    // Sort errors and find median
    abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n_samples % 2 == 0 {
        // Even number of samples, average the middle two
        let mid = n_samples / 2;
        let median = (abs_errors[mid - 1] + abs_errors[mid]) / NumCast::from(2).unwrap();
        Ok(median)
    } else {
        // Odd number of samples, take the middle one
        let mid = n_samples / 2;
        Ok(abs_errors[mid])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_mean_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        // Expecting: ((3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)²) / 4
        // = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
        assert_abs_diff_eq!(mse, 0.375, epsilon = 1e-10);
    }

    #[test]
    fn test_root_mean_squared_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        // RMSE is the square root of MSE (0.375)
        assert_abs_diff_eq!(rmse, 0.6123724356957945, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        // Expecting: (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4
        // = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
        assert_abs_diff_eq!(mae, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_percentage_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        // Expecting: (|3.0-2.5|/3.0 + |-0.5-0.0|/0.5 + |2.0-2.0|/2.0 + |7.0-8.0|/7.0) / 4
        // = (0.5/3.0 + 0.5/0.5 + 0.0/2.0 + 1.0/7.0) / 4
        // = (0.16667 + 1.0 + 0.0 + 0.14286) / 4 = 0.32738
        assert_abs_diff_eq!(mape, 0.32738, epsilon = 1e-5);
    }

    #[test]
    fn test_r2_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        // For this simple example, R² is approximately 0.948
        assert_abs_diff_eq!(r2, 0.9486, epsilon = 1e-4);
    }

    #[test]
    fn test_explained_variance_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let score = explained_variance_score(&y_true, &y_pred).unwrap();
        // For this dataset, explained variance could be 1.0 due to
        // the fact that y_pred perfectly explains the variability of y_true
        assert!(score > 0.95);
    }

    #[test]
    fn test_max_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let max_err = max_error(&y_true, &y_pred).unwrap();
        // Maximum error is |7.0 - 8.0| = 1.0
        assert_abs_diff_eq!(max_err, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_absolute_error() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        let med_ae = median_absolute_error(&y_true, &y_pred).unwrap();
        // Absolute errors: [0.5, 0.5, 0.0, 1.0], median is 0.5
        assert_abs_diff_eq!(med_ae, 0.5, epsilon = 1e-10);
    }
}
