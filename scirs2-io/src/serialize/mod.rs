//! Data serialization utilities
//!
//! This module provides functionality for serializing and deserializing
//! scientific data structures including arrays, matrices, and structured data.
//!
//! Features:
//! - Binary serialization of ndarray arrays
//! - JSON serialization for structured data
//! - Sparse matrix serialization

use ndarray::{Array, ArrayBase, IxDyn};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// Format for data serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Binary format (compact, not human-readable)
    Binary,
    /// JSON format (human-readable)
    JSON,
    /// MessagePack format (compact binary, cross-language)
    MessagePack,
}

/// Serialize an ndarray to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `array` - The ndarray to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array2, IxDyn};
/// use scirs2_io::serialize::{serialize_array, SerializationFormat};
///
/// let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let array_dyn = array.into_dyn();
///
/// // Binary serialization (compact)
/// serialize_array("data.bin", &array_dyn, SerializationFormat::Binary).unwrap();
///
/// // JSON serialization (human-readable)
/// serialize_array("data.json", &array_dyn, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_array<P, A, S>(
    path: P,
    array: &ArrayBase<S, IxDyn>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: ndarray::Data<Elem = A>,
{
    // Create a serializable representation
    let shape = array.shape().to_vec();
    let data: Vec<A> = array.iter().cloned().collect();

    // Create a serializable struct
    let serializable = SerializedArray {
        metadata: ArrayMetadata {
            shape,
            dtype: std::any::type_name::<A>().to_string(),
            order: 'C',
            metadata: std::collections::HashMap::new(),
        },
        data,
    };

    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize an ndarray from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<Array<A, D>>` - Deserialized array or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array, IxDyn};
/// use scirs2_io::serialize::{deserialize_array, SerializationFormat};
///
/// // Binary deserialization
/// let array = deserialize_array::<_, f64>("data.bin", SerializationFormat::Binary).unwrap();
/// println!("Deserialized array shape: {:?}", array.shape());
///
/// // JSON deserialization
/// let array = deserialize_array::<_, f64>("data.json", SerializationFormat::JSON).unwrap();
/// ```
pub fn deserialize_array<P, A>(path: P, format: SerializationFormat) -> Result<Array<A, IxDyn>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let serialized: SerializedArray<A> = match format {
        SerializationFormat::Binary => bincode::deserialize_from(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::JSON => serde_json::from_reader(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::MessagePack => rmp_serde::from_read(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
    };

    // Convert back to ndarray
    let array = Array::from_shape_vec(IxDyn(&serialized.metadata.shape), serialized.data)
        .map_err(|e| IoError::FormatError(format!("Failed to reconstruct array: {}", e)))?;

    Ok(array)
}

/// Array metadata for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadata {
    /// Array shape
    pub shape: Vec<usize>,
    /// Array element type
    pub dtype: String,
    /// Array order ('C' for row-major, 'F' for column-major)
    pub order: char,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Serialized array data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedArray<A> {
    /// Array metadata
    pub metadata: ArrayMetadata,
    /// Flattened array data
    pub data: Vec<A>,
}

/// Serialize an ndarray with metadata
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `array` - The ndarray to serialize
/// * `metadata` - Additional metadata
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_io::serialize::{serialize_array_with_metadata, SerializationFormat};
/// use std::collections::HashMap;
///
/// let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let array_dyn = array.into_dyn();
///
/// // Create custom metadata
/// let mut metadata = HashMap::new();
/// metadata.insert("description".to_string(), "Test array".to_string());
/// metadata.insert("units".to_string(), "meters".to_string());
///
/// serialize_array_with_metadata(
///     "data_with_metadata.json",
///     &array_dyn,
///     metadata,
///     SerializationFormat::JSON
/// ).unwrap();
/// ```
pub fn serialize_array_with_metadata<P, A, S>(
    path: P,
    array: &ArrayBase<S, IxDyn>,
    metadata: std::collections::HashMap<String, String>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: ndarray::Data<Elem = A>,
{
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Create array metadata
    let shape = array.shape().to_vec();
    let dtype = std::any::type_name::<A>().to_string();

    let array_metadata = ArrayMetadata {
        shape,
        dtype,
        order: 'C', // Rust ndarray uses row-major (C) order
        metadata,
    };

    // Create serialized array
    let serialized = SerializedArray {
        metadata: array_metadata,
        data: array.iter().cloned().collect(),
    };

    // Serialize
    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize an ndarray with metadata from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<(Array<A, IxDyn>, std::collections::HashMap<String, String>)>` - Deserialized array and metadata
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_io::serialize::{deserialize_array_with_metadata, SerializationFormat};
///
/// // Deserialize array with metadata
/// let (array, metadata) = deserialize_array_with_metadata::<_, f64>(
///     "data_with_metadata.json",
///     SerializationFormat::JSON
/// ).unwrap();
///
/// println!("Deserialized array shape: {:?}", array.shape());
/// println!("Metadata: {:?}", metadata);
/// ```
pub fn deserialize_array_with_metadata<P, A>(
    path: P,
    format: SerializationFormat,
) -> Result<(Array<A, IxDyn>, std::collections::HashMap<String, String>)>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let serialized: SerializedArray<A> = match format {
        SerializationFormat::Binary => bincode::deserialize_from(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::JSON => serde_json::from_reader(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::MessagePack => rmp_serde::from_read(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
    };

    // Extract shape and data
    let shape = serialized.metadata.shape;
    let data = serialized.data;

    // Create ndarray
    let array = Array::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| IoError::FormatError(format!("Invalid shape: {:?}", e)))?;

    Ok((array, serialized.metadata.metadata))
}

/// Serialize a struct to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `data` - The struct to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use serde::{Serialize, Deserialize};
/// use scirs2_io::serialize::{serialize_struct, SerializationFormat};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
///     height: f64,
/// }
///
/// let person = Person {
///     name: "Alice".to_string(),
///     age: 30,
///     height: 175.5,
/// };
///
/// // JSON serialization
/// serialize_struct("person.json", &person, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_struct<P, T>(path: P, data: &T, format: SerializationFormat) -> Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize a struct from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<T>` - Deserialized struct or error
///
/// # Examples
///
/// ```no_run
/// use serde::{Serialize, Deserialize};
/// use scirs2_io::serialize::{deserialize_struct, SerializationFormat};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
///     height: f64,
/// }
///
/// // JSON deserialization
/// let person: Person = deserialize_struct("person.json", SerializationFormat::JSON).unwrap();
/// println!("Name: {}, Age: {}, Height: {}", person.name, person.age, person.height);
/// ```
pub fn deserialize_struct<P, T>(path: P, format: SerializationFormat) -> Result<T>
where
    P: AsRef<Path>,
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    match format {
        SerializationFormat::Binary => {
            let data = bincode::deserialize_from(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
        SerializationFormat::JSON => {
            let data = serde_json::from_reader(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
        SerializationFormat::MessagePack => {
            let data = rmp_serde::from_read(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
    }
}

/// Type representing a sparse matrix in COO (Coordinate) format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixCOO<A> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<A>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl<A> SparseMatrixCOO<A> {
    /// Create a new sparse matrix in COO format
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a value to the sparse matrix
    pub fn push(&mut self, row: usize, col: usize, value: A) {
        if row < self.rows && col < self.cols {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Serialize a sparse matrix to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `matrix` - The sparse matrix to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::serialize::{serialize_sparse_matrix, SparseMatrixCOO, SerializationFormat};
///
/// // Create a sparse matrix
/// let mut sparse = SparseMatrixCOO::<f64>::new(100, 100);
/// sparse.push(0, 0, 1.0);
/// sparse.push(10, 10, 2.0);
/// sparse.push(50, 50, 3.0);
///
/// // Add metadata
/// sparse.metadata.insert("description".to_string(), "Sparse test matrix".to_string());
///
/// // Serialize to file
/// serialize_sparse_matrix("sparse.json", &sparse, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_sparse_matrix<P, A>(
    path: P,
    matrix: &SparseMatrixCOO<A>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize,
{
    serialize_struct(path, matrix, format)
}

/// Deserialize a sparse matrix from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<SparseMatrixCOO<A>>` - Deserialized sparse matrix or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::serialize::{deserialize_sparse_matrix, SerializationFormat};
///
/// // Deserialize sparse matrix
/// let sparse = deserialize_sparse_matrix::<_, f64>("sparse.json", SerializationFormat::JSON).unwrap();
/// println!("Sparse matrix: {}x{} with {} non-zero elements", sparse.rows, sparse.cols, sparse.nnz());
/// ```
pub fn deserialize_sparse_matrix<P, A>(
    path: P,
    format: SerializationFormat,
) -> Result<SparseMatrixCOO<A>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de>,
{
    deserialize_struct(path, format)
}
