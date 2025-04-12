//! Functions for creating and manipulating structuring elements

use ndarray::{Array, Dimension, IxDyn};
// use std::fmt::Debug;

use super::Connectivity;
use crate::error::{NdimageError, Result};

/// Generate a binary structure for morphological operations
///
/// # Arguments
///
/// * `rank` - The number of dimensions of the array
/// * `connectivity` - Connectivity type (Face, FaceEdge, or Full)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary structuring element
pub fn generate_binary_structure<D>(
    rank: usize,
    _connectivity: Connectivity,
) -> Result<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if rank == 0 {
        return Err(NdimageError::InvalidInput(
            "Rank must be greater than 0".into(),
        ));
    }

    // For a proper implementation, we would:
    // 1. Create an array of the appropriate shape (3^rank)
    // 2. Fill it with the appropriate values based on connectivity

    // Placeholder implementation that returns an error
    Err(NdimageError::ImplementationError(
        "generate_binary_structure is not fully implemented yet".into(),
    ))
}

/// Iterate binary erosion or dilation until convergence
///
/// # Arguments
///
/// * `structure` - Input structuring element
/// * `iterations` - Number of iterations
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Iterated structuring element
pub fn iterate_structure<D>(structure: &Array<bool, D>, iterations: usize) -> Result<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if structure.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Structure cannot be 0-dimensional".into(),
        ));
    }

    if iterations == 0 {
        return Ok(structure.to_owned());
    }

    // Placeholder implementation returning a copy of the input
    // This will be implemented with proper iteration of the structure
    Ok(structure.to_owned())
}

/// Create a box structuring element of given size
///
/// # Arguments
///
/// * `shape` - The shape of the element
///
/// # Returns
///
/// * `Result<Array<bool, IxDyn>>` - Box structuring element
pub fn box_structure(shape: &[usize]) -> Result<Array<bool, IxDyn>> {
    // Validate inputs
    if shape.is_empty() {
        return Err(NdimageError::InvalidInput("Shape cannot be empty".into()));
    }

    for &s in shape {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Shape dimensions must be greater than 0".into(),
            ));
        }
    }

    // Create a box of ones
    let structure = Array::<bool, _>::from_elem(IxDyn(shape), true);
    Ok(structure)
}

/// Create a disk structuring element
///
/// # Arguments
///
/// * `radius` - The radius of the disk
/// * `dimension` - The number of dimensions (default: 2)
///
/// # Returns
///
/// * `Result<Array<bool, IxDyn>>` - Disk structuring element
pub fn disk_structure(radius: f64, dimension: Option<usize>) -> Result<Array<bool, IxDyn>> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "Radius must be greater than 0".into(),
        ));
    }

    let dim = dimension.unwrap_or(2);
    if dim < 2 {
        return Err(NdimageError::InvalidInput(
            "Dimension must be at least 2".into(),
        ));
    }

    // Placeholder implementation that creates a box instead of a disk
    let size = (2.0 * radius.ceil() + 1.0) as usize;
    let shape = vec![size; dim];
    box_structure(&shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_iterate_structure() {
        let input = Array2::from_elem((3, 3), true);
        let result = iterate_structure(&input, 1).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_box_structure() {
        let result = box_structure(&[3, 3]).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert!(result.iter().all(|&x| x));
    }

    #[test]
    #[ignore = "Implementation is placeholder, will be fixed with full implementation"]
    fn test_disk_structure() {
        let result = disk_structure(1.5, None).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
    }
}
