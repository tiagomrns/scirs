//! Compressed Sparse Column (CSC) matrix format
//!
//! This module provides the CSC matrix format implementation, which is
//! efficient for column operations, sparse matrix multiplication, and more.

use crate::error::{SparseError, SparseResult};
use num_traits::Zero;
use std::cmp::PartialEq;

/// Compressed Sparse Column (CSC) matrix
///
/// A sparse matrix format that compresses columns, making it efficient for
/// column operations and matrix operations.
pub struct CscMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Column pointers (size cols+1)
    indptr: Vec<usize>,
    /// Row indices
    indices: Vec<usize>,
    /// Data values
    data: Vec<T>,
}

impl<T> CscMatrix<T>
where
    T: Clone + Copy + Zero + PartialEq,
{
    /// Create a new CSC matrix from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of non-zero values
    /// * `row_indices` - Vector of row indices for each non-zero value
    /// * `col_indices` - Vector of column indices for each non-zero value
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new CSC matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::csc::CscMatrix;
    ///
    /// // Create a 3x3 sparse matrix with 5 non-zero elements
    /// let rows = vec![0, 0, 1, 2, 2];
    /// let cols = vec![0, 2, 2, 0, 1];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let shape = (3, 3);
    ///
    /// let matrix = CscMatrix::new(data.clone(), rows, cols, shape).unwrap();
    /// ```
    pub fn new(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        // Validate input data
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: std::cmp::min(row_indices.len(), col_indices.len()),
            });
        }

        let (rows, cols) = shape;

        // Check indices are within bounds
        if row_indices.iter().any(|&i| i >= rows) {
            return Err(SparseError::ValueError(
                "Row index out of bounds".to_string(),
            ));
        }

        if col_indices.iter().any(|&i| i >= cols) {
            return Err(SparseError::ValueError(
                "Column index out of bounds".to_string(),
            ));
        }

        // Convert triplet format to CSC
        // First, sort by column, then by row
        let mut triplets: Vec<(usize, usize, T)> = col_indices
            .into_iter()
            .zip(row_indices)
            .zip(data)
            .map(|((c, r), v)| (c, r, v))
            .collect();
        triplets.sort_by_key(|&(c, r, _)| (c, r));

        // Create indptr, indices, and data arrays
        let nnz = triplets.len();
        let mut indptr = vec![0; cols + 1];
        let mut indices = Vec::with_capacity(nnz);
        let mut data_out = Vec::with_capacity(nnz);

        // Count elements per column to build indptr
        for &(c, _, _) in &triplets {
            indptr[c + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=cols {
            indptr[i] += indptr[i - 1];
        }

        // Fill indices and data
        for (_, r, v) in triplets {
            indices.push(r);
            data_out.push(v);
        }

        Ok(CscMatrix {
            rows,
            cols,
            indptr,
            indices,
            data: data_out,
        })
    }

    /// Create a new CSC matrix from raw CSC format
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of non-zero values
    /// * `indptr` - Vector of column pointers (size cols+1)
    /// * `indices` - Vector of row indices
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new CSC matrix
    pub fn from_raw_csc(
        data: Vec<T>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;

        // Validate input data
        if indptr.len() != cols + 1 {
            return Err(SparseError::DimensionMismatch {
                expected: cols + 1,
                found: indptr.len(),
            });
        }

        if data.len() != indices.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: indices.len(),
            });
        }

        // Check if indptr is monotonically increasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(SparseError::ValueError(
                    "Column pointer array must be monotonically increasing".to_string(),
                ));
            }
        }

        // Check if the last indptr entry matches the data length
        if indptr[cols] != data.len() {
            return Err(SparseError::ValueError(
                "Last column pointer entry must match data length".to_string(),
            ));
        }

        // Check if indices are within bounds
        if indices.iter().any(|&i| i >= rows) {
            return Err(SparseError::ValueError(
                "Row index out of bounds".to_string(),
            ));
        }

        Ok(CscMatrix {
            rows,
            cols,
            indptr,
            indices,
            data,
        })
    }

    /// Create a new empty CSC matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty CSC matrix
    ///   Create a CSC matrix from CSC data (values, row indices, column pointers)
    ///
    /// # Arguments
    ///
    /// * `values` - Values
    /// * `row_indices` - Row indices
    /// * `col_ptrs` - Column pointers
    /// * `shape` - Shape of the matrix (rows, cols)
    ///
    /// # Returns
    ///
    /// * Result containing the CSC matrix
    pub fn from_csc_data(
        values: Vec<T>,
        row_indices: Vec<usize>,
        col_ptrs: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        Self::from_raw_csc(values, col_ptrs, row_indices, shape)
    }

    pub fn empty(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;
        let indptr = vec![0; cols + 1];

        CscMatrix {
            rows,
            cols,
            indptr,
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Get the number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the shape (dimensions) of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get column range for iterating over elements in a column
    ///
    /// # Arguments
    ///
    /// * `col` - Column index
    ///
    /// # Returns
    ///
    /// * Range of indices in the data and indices arrays for this column
    pub fn col_range(&self, col: usize) -> std::ops::Range<usize> {
        assert!(col < self.cols, "Column index out of bounds");
        self.indptr[col]..self.indptr[col + 1]
    }

    /// Get row indices array
    pub fn row_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get data array
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Convert to dense matrix (as Vec<Vec<T>>)
    pub fn to_dense(&self) -> Vec<Vec<T>>
    where
        T: Zero + Copy,
    {
        let mut result = vec![vec![T::zero(); self.cols]; self.rows];

        for col_idx in 0..self.cols {
            for j in self.indptr[col_idx]..self.indptr[col_idx + 1] {
                let row_idx = self.indices[j];
                result[row_idx][col_idx] = self.data[j];
            }
        }

        result
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        // Compute the number of non-zeros per row
        let mut row_counts = vec![0; self.rows];
        for &row in &self.indices {
            row_counts[row] += 1;
        }

        // Compute row pointers (cumulative sum)
        let mut row_ptrs = vec![0; self.rows + 1];
        for i in 0..self.rows {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }

        // Fill the transposed matrix
        let nnz = self.nnz();
        let mut indices_t = vec![0; nnz];
        let mut data_t = vec![T::zero(); nnz];
        let mut row_counts = vec![0; self.rows];

        for col in 0..self.cols {
            for j in self.indptr[col]..self.indptr[col + 1] {
                let row = self.indices[j];
                let dest = row_ptrs[row] + row_counts[row];

                indices_t[dest] = col;
                data_t[dest] = self.data[j];
                row_counts[row] += 1;
            }
        }

        CscMatrix {
            rows: self.cols,
            cols: self.rows,
            indptr: row_ptrs,
            indices: indices_t,
            data: data_t,
        }
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> crate::csr::CsrMatrix<T> {
        // The transpose of a CSC is essentially a CSR with transposed dimensions
        let transposed = self.transpose();

        crate::csr::CsrMatrix::from_raw_csr(
            transposed.data,
            transposed.indptr,
            transposed.indices,
            (self.rows, self.cols),
        )
        .unwrap()
    }
}

impl CscMatrix<f64> {
    /// Matrix-vector multiplication
    ///
    /// # Arguments
    ///
    /// * `vec` - Vector to multiply with
    ///
    /// # Returns
    ///
    /// * Result of matrix-vector multiplication
    pub fn dot(&self, vec: &[f64]) -> SparseResult<Vec<f64>> {
        if vec.len() != self.cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.cols,
                found: vec.len(),
            });
        }

        let mut result = vec![0.0; self.rows];

        for (col_idx, &col_val) in vec.iter().enumerate().take(self.cols) {
            for j in self.indptr[col_idx]..self.indptr[col_idx + 1] {
                let row_idx = self.indices[j];
                result[row_idx] += self.data[j] * col_val;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csc_create() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CscMatrix::new(data, rows, cols, shape).unwrap();

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 5);
    }

    #[test]
    fn test_csc_to_dense() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CscMatrix::new(data, rows, cols, shape).unwrap();
        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_csc_dot() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CscMatrix::new(data, rows, cols, shape).unwrap();

        // Matrix:
        // [1 0 2]
        // [0 0 3]
        // [4 5 0]

        let vec = vec![1.0, 2.0, 3.0];
        let result = matrix.dot(&vec).unwrap();

        // Expected:
        // 1*1 + 0*2 + 2*3 = 7
        // 0*1 + 0*2 + 3*3 = 9
        // 4*1 + 5*2 + 0*3 = 14
        let expected = vec![7.0, 9.0, 14.0];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_csc_transpose() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CscMatrix::new(data, rows, cols, shape).unwrap();
        let transposed = matrix.transpose();

        assert_eq!(transposed.shape(), (3, 3));
        assert_eq!(transposed.nnz(), 5);

        let dense = transposed.to_dense();
        let expected = vec![
            vec![1.0, 0.0, 4.0],
            vec![0.0, 0.0, 5.0],
            vec![2.0, 3.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_csc_to_csr() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csc_matrix = CscMatrix::new(data, rows, cols, shape).unwrap();
        let csr_matrix = csc_matrix.to_csr();

        assert_eq!(csr_matrix.shape(), (3, 3));
        assert_eq!(csr_matrix.nnz(), 5);

        let dense_from_csc = csc_matrix.to_dense();
        let dense_from_csr = csr_matrix.to_dense();

        assert_eq!(dense_from_csc, dense_from_csr);
    }
}
