//! Sparse-Dense matrix operations
//!
//! This module provides operations between sparse and dense matrices, facilitating
//! interoperability between the two formats. The operations are optimized for the
//! common case where one matrix is sparse and the other is dense.
//!
//! ## Overview
//!
//! * Matrix-matrix multiplication between sparse and dense matrices
//! * Matrix-vector multiplication with sparse matrices
//! * Element-wise operations (addition, subtraction, multiplication)
//! * Conversions between sparse and dense formats
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, Array1, array};
//! use scirs2_linalg::sparse_dense::{sparse_from_ndarray, sparse_dense_matvec};
//!
//! // Create a dense matrix and convert to sparse
//! let dense_mat = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];
//! let sparse_mat = sparse_from_ndarray(&dense_mat.view(), 1e-10).unwrap();
//!
//! // Create a dense vector
//! let dense_vec = array![1.0, 2.0, 3.0];
//!
//! // Perform sparse-dense matrix-vector multiplication
//! let result = sparse_dense_matvec(&sparse_mat, &dense_vec.view()).unwrap();
//! assert!(f64::abs(result[0] - 7.0) < 1e-10);
//! assert!(f64::abs(result[1] - 9.0) < 1e-10);
//! assert!(f64::abs(result[2] - 14.0) < 1e-10);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use num_traits::{cast, Float, NumCast, Zero};

use crate::error::{LinalgError, LinalgResult};

/// A simple view struct for sparse matrices in CSR format
///
/// This struct provides a simple interface to view a sparse matrix in CSR format.
/// It is designed to be compatible with the CSR format used in scirs2-sparse.
#[derive(Debug, Clone)]
pub struct SparseMatrixView<T> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row pointers (size rows+1)
    pub indptr: Vec<usize>,
    /// Column indices
    pub indices: Vec<usize>,
    /// Data values
    pub data: Vec<T>,
}

impl<T> SparseMatrixView<T>
where
    T: Clone + Copy + Debug + Zero,
{
    /// Create a new CSR sparse matrix view
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of non-zero values
    /// * `indptr` - Vector of row pointers (size rows+1)
    /// * `indices` - Vector of column indices
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// A new sparse matrix view
    pub fn new(
        data: Vec<T>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        shape: (usize, usize),
    ) -> LinalgResult<Self> {
        let (rows, cols) = shape;

        // Validate input data
        if indptr.len() != rows + 1 {
            return Err(LinalgError::DimensionError(format!(
                "Row pointer array length must be rows + 1, got {} for {} rows",
                indptr.len(),
                rows
            )));
        }

        if data.len() != indices.len() {
            return Err(LinalgError::DimensionError(format!(
                "Data and indices must have the same length, got {} and {}",
                data.len(),
                indices.len()
            )));
        }

        // Check if indptr is monotonically increasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(LinalgError::ValueError(
                    "Row pointer array must be monotonically increasing".to_string(),
                ));
            }
        }

        // Check if the last indptr entry matches the data length
        if indptr[rows] != data.len() {
            return Err(LinalgError::ValueError(format!(
                "Last row pointer entry must match data length, got {} and {}",
                indptr[rows],
                data.len()
            )));
        }

        // Check if indices are within bounds
        if let Some(&max_index) = indices.iter().max() {
            if max_index >= cols {
                return Err(LinalgError::ValueError(format!(
                    "Column index out of bounds: {} for a matrix with {} columns",
                    max_index, cols
                )));
            }
        }

        Ok(SparseMatrixView {
            rows,
            cols,
            indptr,
            indices,
            data,
        })
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Check if the matrix is empty
    pub fn is_empty(&self) -> bool {
        self.nnz() == 0
    }

    /// Convert the sparse matrix to a dense ndarray
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = Array2::zeros((self.rows, self.cols));

        for row in 0..self.rows {
            for j in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[j];
                dense[[row, col]] = self.data[j];
            }
        }

        dense
    }
}

/// Convert a dense ndarray to a sparse matrix in CSR format
///
/// # Arguments
///
/// * `array` - Dense ndarray to convert
/// * `threshold` - Threshold below which values are considered zero
///
/// # Returns
///
/// A sparse matrix view in CSR format
pub fn sparse_from_ndarray<T, F>(
    array: &ArrayView2<T>,
    threshold: F,
) -> LinalgResult<SparseMatrixView<T>>
where
    T: Clone + Copy + Debug + PartialOrd + Zero + NumCast,
    F: Float + NumCast,
{
    let shape = array.dim();
    let (rows, cols) = (shape.0, shape.1);

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0; rows + 1];

    let threshold_abs: F = threshold.abs();

    // Count non-zeros and fill data
    for (i, row) in array.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let val_abs: F = if let Some(v) = cast::<T, F>(val) {
                v.abs()
            } else {
                F::zero()
            };
            if val_abs > threshold_abs {
                data.push(val);
                indices.push(j);
                indptr[i + 1] += 1;
            }
        }
    }

    // Convert counts to offsets
    for i in 1..=rows {
        indptr[i] += indptr[i - 1];
    }

    SparseMatrixView::new(data, indptr, indices, (rows, cols))
}

/// Matrix-matrix multiplication between a sparse matrix and a dense matrix
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
/// * `dense` - Dense matrix
///
/// # Returns
///
/// A dense matrix containing the result
pub fn sparse_dense_matmul<T>(
    sparse: &SparseMatrixView<T>,
    dense: &ArrayView2<T>,
) -> LinalgResult<Array2<T>>
where
    T: Clone + Copy + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    // Check dimensions
    if sparse.cols != dense.dim().0 {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions incompatible for multiplication: {}x{} and {}x{}",
            sparse.rows,
            sparse.cols,
            dense.dim().0,
            dense.dim().1
        )));
    }

    // Initialize result matrix
    let result_rows = sparse.rows;
    let result_cols = dense.dim().1;
    let mut result = Array2::zeros((result_rows, result_cols));

    // Perform matrix multiplication
    for i in 0..sparse.rows {
        for j in 0..result_cols {
            let mut sum = T::zero();

            // Multiply sparse row by dense column
            for k in sparse.indptr[i]..sparse.indptr[i + 1] {
                let col = sparse.indices[k];
                sum = sum + sparse.data[k] * dense[[col, j]];
            }

            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Matrix-matrix multiplication between a dense matrix and a sparse matrix
///
/// # Arguments
///
/// * `dense` - Dense matrix
/// * `sparse` - Sparse matrix in CSR format
///
/// # Returns
///
/// A dense matrix containing the result
pub fn dense_sparse_matmul<T>(
    dense: &ArrayView2<T>,
    sparse: &SparseMatrixView<T>,
) -> LinalgResult<Array2<T>>
where
    T: Clone + Copy + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    // Check dimensions
    if dense.dim().1 != sparse.rows {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions incompatible for multiplication: {}x{} and {}x{}",
            dense.dim().0,
            dense.dim().1,
            sparse.rows,
            sparse.cols
        )));
    }

    // Initialize result matrix
    let result_rows = dense.dim().0;
    let result_cols = sparse.cols;
    let mut result = Array2::zeros((result_rows, result_cols));

    // Initialize column sums for each sparse column
    let mut col_sums: HashMap<usize, Vec<T>> = HashMap::new();

    // Populate column sums
    for i in 0..sparse.rows {
        for j in sparse.indptr[i]..sparse.indptr[i + 1] {
            let col = sparse.indices[j];
            let val = sparse.data[j];

            col_sums
                .entry(col)
                .or_insert_with(|| vec![T::zero(); result_rows]);

            // Compute dense_row * sparse_val for each dense row
            for (k, dense_row) in dense.axis_iter(Axis(0)).enumerate() {
                col_sums.get_mut(&col).unwrap()[k] = col_sums[&col][k] + dense_row[i] * val;
            }
        }
    }

    // Fill the result matrix
    for (col, sums) in col_sums.iter() {
        for (row, &sum) in sums.iter().enumerate() {
            result[[row, *col]] = sum;
        }
    }

    Ok(result)
}

/// Matrix-vector multiplication between a sparse matrix and a dense vector
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
/// * `vector` - Dense vector
///
/// # Returns
///
/// A dense vector containing the result
pub fn sparse_dense_matvec<T>(
    sparse: &SparseMatrixView<T>,
    vector: &ArrayView1<T>,
) -> LinalgResult<Array1<T>>
where
    T: Clone + Copy + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    // Check dimensions
    if sparse.cols != vector.len() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix-vector dimensions incompatible: {}x{} and {}",
            sparse.rows,
            sparse.cols,
            vector.len()
        )));
    }

    // Initialize result vector
    let mut result = Array1::zeros(sparse.rows);

    // Perform sparse matrix-vector multiplication
    for i in 0..sparse.rows {
        let mut sum = T::zero();

        for j in sparse.indptr[i]..sparse.indptr[i + 1] {
            let col = sparse.indices[j];
            sum = sum + sparse.data[j] * vector[col];
        }

        result[i] = sum;
    }

    Ok(result)
}

/// Matrix-vector multiplication between a dense matrix and a sparse vector
///
/// # Arguments
///
/// * `dense` - Dense matrix
/// * `sparse` - Sparse vector in CSR format (as a single-row sparse matrix)
///
/// # Returns
///
/// A dense vector containing the result
pub fn dense_sparse_matvec<T>(
    dense: &ArrayView2<T>,
    sparse: &SparseMatrixView<T>,
) -> LinalgResult<Array1<T>>
where
    T: Clone + Copy + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    // Check if the sparse matrix is actually a vector (single row)
    if sparse.rows != 1 {
        return Err(LinalgError::DimensionError(format!(
            "Expected a sparse vector (single row), got {} rows",
            sparse.rows
        )));
    }

    // Check dimensions
    if dense.dim().1 != sparse.cols {
        return Err(LinalgError::DimensionError(format!(
            "Matrix-vector dimensions incompatible: {}x{} and {}",
            dense.dim().0,
            dense.dim().1,
            sparse.cols
        )));
    }

    // Initialize result vector
    let mut result = Array1::zeros(dense.dim().0);

    // Perform dense matrix-sparse vector multiplication
    for j in sparse.indptr[0]..sparse.indptr[1] {
        let col = sparse.indices[j];
        let val = sparse.data[j];

        for (i, row_val) in result.iter_mut().enumerate() {
            *row_val = *row_val + dense[[i, col]] * val;
        }
    }

    Ok(result)
}

/// Element-wise addition between a sparse matrix and a dense matrix
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
/// * `dense` - Dense matrix
///
/// # Returns
///
/// A dense matrix containing the result
pub fn sparse_dense_add<T>(
    sparse: &SparseMatrixView<T>,
    dense: &ArrayView2<T>,
) -> LinalgResult<Array2<T>>
where
    T: Clone + Copy + Debug + Zero + Add<Output = T>,
{
    // Check dimensions
    if sparse.shape() != dense.dim() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions incompatible for addition: {}x{} and {}x{}",
            sparse.rows,
            sparse.cols,
            dense.dim().0,
            dense.dim().1
        )));
    }

    // Start with a copy of the dense matrix
    let mut result = dense.to_owned();

    // Add the sparse elements
    for i in 0..sparse.rows {
        for j in sparse.indptr[i]..sparse.indptr[i + 1] {
            let col = sparse.indices[j];
            result[[i, col]] = result[[i, col]] + sparse.data[j];
        }
    }

    Ok(result)
}

/// Element-wise subtraction between a sparse matrix and a dense matrix
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
/// * `dense` - Dense matrix
///
/// # Returns
///
/// A dense matrix containing the result (sparse - dense)
pub fn sparse_dense_sub<T>(
    sparse: &SparseMatrixView<T>,
    dense: &ArrayView2<T>,
) -> LinalgResult<Array2<T>>
where
    T: Clone + Copy + Debug + Zero + Sub<Output = T> + Neg<Output = T>,
{
    // Check dimensions
    if sparse.shape() != dense.dim() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions incompatible for subtraction: {}x{} and {}x{}",
            sparse.rows,
            sparse.cols,
            dense.dim().0,
            dense.dim().1
        )));
    }

    // Start with negative of the dense matrix
    let mut result = dense.mapv(|x| -x);

    // Add the sparse elements
    for i in 0..sparse.rows {
        for j in sparse.indptr[i]..sparse.indptr[i + 1] {
            let col = sparse.indices[j];
            result[[i, col]] = result[[i, col]] + sparse.data[j];
        }
    }

    Ok(result)
}

/// Element-wise multiplication between a sparse matrix and a dense matrix
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
/// * `dense` - Dense matrix
///
/// # Returns
///
/// A sparse matrix containing the result
pub fn sparse_dense_elementwise_mul<T>(
    sparse: &SparseMatrixView<T>,
    dense: &ArrayView2<T>,
) -> LinalgResult<SparseMatrixView<T>>
where
    T: Clone + Copy + Debug + Zero + Mul<Output = T> + PartialEq,
{
    // Check dimensions
    if sparse.shape() != dense.dim() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions incompatible for element-wise multiplication: {}x{} and {}x{}",
            sparse.rows,
            sparse.cols,
            dense.dim().0,
            dense.dim().1
        )));
    }

    // Prepare data structures for result
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0; sparse.rows + 1];

    // Multiply sparse and dense elements
    for i in 0..sparse.rows {
        for j in sparse.indptr[i]..sparse.indptr[i + 1] {
            let col = sparse.indices[j];
            let val = sparse.data[j] * dense[[i, col]];

            // Only include non-zero values
            if val != T::zero() {
                data.push(val);
                indices.push(col);
                indptr[i + 1] += 1;
            }
        }
    }

    // Convert counts to offsets
    for i in 1..=sparse.rows {
        indptr[i] += indptr[i - 1];
    }

    SparseMatrixView::new(data, indptr, indices, sparse.shape())
}

/// Transpose a sparse matrix
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
///
/// # Returns
///
/// A transposed sparse matrix
pub fn sparse_transpose<T>(sparse: &SparseMatrixView<T>) -> LinalgResult<SparseMatrixView<T>>
where
    T: Clone + Copy + Debug + Zero,
{
    // Compute the number of non-zeros per column
    let mut col_counts = vec![0; sparse.cols];
    for &col in &sparse.indices {
        col_counts[col] += 1;
    }

    // Compute column pointers (cumulative sum)
    let mut col_ptrs = vec![0; sparse.cols + 1];
    for i in 0..sparse.cols {
        col_ptrs[i + 1] = col_ptrs[i] + col_counts[i];
    }

    // Fill the transposed matrix
    let nnz = sparse.nnz();
    let mut indices_t = vec![0; nnz];
    let mut data_t = vec![T::zero(); nnz];
    let mut col_offsets = vec![0; sparse.cols];

    for row in 0..sparse.rows {
        for j in sparse.indptr[row]..sparse.indptr[row + 1] {
            let col = sparse.indices[j];
            let dest = col_ptrs[col] + col_offsets[col];

            indices_t[dest] = row;
            data_t[dest] = sparse.data[j];
            col_offsets[col] += 1;
        }
    }

    SparseMatrixView::new(data_t, col_ptrs, indices_t, (sparse.cols, sparse.rows))
}

// Implementation of Neg trait for T
use std::ops::Neg;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sparse_from_ndarray() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let sparse = sparse_from_ndarray(&dense.view(), 1e-10).unwrap();

        // Check dimensions
        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 5);

        // Convert back to dense and compare
        let dense_again = sparse.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dense[[i, j]], dense_again[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_dense_matmul() {
        let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let dense_b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();

        // Sparse-dense multiplication
        let result = sparse_dense_matmul(&sparse_a, &dense_b.view()).unwrap();

        // Expected result of A * B
        let expected = array![[11.0, 14.0], [15.0, 18.0], [19.0, 28.0]];

        // Compare
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dense_sparse_matmul() {
        let dense_a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let dense_b = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let sparse_b = sparse_from_ndarray(&dense_b.view(), 1e-10).unwrap();

        // Dense-sparse multiplication
        let result = dense_sparse_matmul(&dense_a.view(), &sparse_b).unwrap();

        // Manually compute the expected result by direct multiplication
        let expected = dense_a.dot(&dense_b);

        // Compare
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_dense_matvec() {
        let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let vec_b = array![1.0, 2.0, 3.0];

        let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();

        // Sparse-dense matvec
        let result = sparse_dense_matvec(&sparse_a, &vec_b.view()).unwrap();

        // Expected result of A * v
        let expected = array![7.0, 9.0, 14.0];

        // Compare
        for i in 0..3 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sparse_dense_add() {
        let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let dense_b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();

        // Sparse + dense
        let result = sparse_dense_add(&sparse_a, &dense_b.view()).unwrap();

        // Expected result of A + B
        let expected = array![[2.0, 2.0, 5.0], [4.0, 5.0, 9.0], [11.0, 13.0, 9.0]];

        // Compare
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_dense_elementwise_mul() {
        let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let dense_b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();

        // Element-wise multiplication
        let result = sparse_dense_elementwise_mul(&sparse_a, &dense_b.view()).unwrap();

        // Convert to dense for comparison
        let result_dense = result.to_dense();

        // Expected result of A .* B
        let expected = array![[1.0, 0.0, 6.0], [0.0, 0.0, 18.0], [28.0, 40.0, 0.0]];

        // Compare
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(result_dense[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_transpose() {
        let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

        let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();

        // Transpose
        let transposed = sparse_transpose(&sparse_a).unwrap();

        // Convert to dense for comparison
        let transposed_dense = transposed.to_dense();

        // Expected result of A^T
        let expected = array![[1.0, 0.0, 4.0], [0.0, 0.0, 5.0], [2.0, 3.0, 0.0]];

        // Compare
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(transposed_dense[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
