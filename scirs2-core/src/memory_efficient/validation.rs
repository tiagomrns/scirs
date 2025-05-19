use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use ndarray::{ArrayBase, Dimension};

/// Checks if an array is not empty
///
/// # Arguments
///
/// * `array` - The array to check
///
/// # Returns
///
/// * `Ok(())` if the array is not empty
/// * `Err(CoreError::ValidationError)` if the array is empty
pub fn check_not_empty<S, D>(array: &ArrayBase<S, D>) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
{
    if array.is_empty() {
        return Err(CoreError::ValidationError(
            ErrorContext::new("Array is empty".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if two arrays have matching shapes
///
/// # Arguments
///
/// * `shape1` - The shape of the first array
/// * `shape2` - The shape of the second array
///
/// # Returns
///
/// * `Ok(())` if the shapes match
/// * `Err(CoreError::ValidationError)` if the shapes don't match
pub fn check_shapes_match<D1, D2>(shape1: D1, shape2: D2) -> CoreResult<()>
where
    D1: AsRef<[usize]>,
    D2: AsRef<[usize]>,
{
    let s1 = shape1.as_ref();
    let s2 = shape2.as_ref();

    if s1 != s2 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!("Shapes don't match: {:?} vs {:?}", s1, s2))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

/// Checks if a matrix is square
///
/// # Arguments
///
/// * `array` - The array to check
///
/// # Returns
///
/// * `Ok(())` if the array is square (2D with same dimensions)
/// * `Err(CoreError::ValidationError)` if the array is not square
pub fn check_square<S, D>(array: &ArrayBase<S, D>) -> CoreResult<()>
where
    S: ndarray::Data,
    D: Dimension,
{
    let shape = array.shape();

    if shape.len() != 2 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Array must be 2D for square check, got {}D",
                shape.len()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    if shape[0] != shape[1] {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!("Matrix is not square: {:?}", shape))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    Ok(())
}
