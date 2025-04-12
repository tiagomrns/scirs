//! Compression utilities for ndarray types
//!
//! This module provides specialized compression functionality for ndarray
//! types, optimizing for common patterns in scientific data arrays.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use ndarray::{ArrayBase, Data, Dimension, IxDyn, OwnedRepr};
use serde::{Deserialize, Serialize};

use super::{compress_data, decompress_data, CompressionAlgorithm};
use crate::error::{IoError, Result};

/// Metadata for compressed array data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedArrayMetadata {
    /// Shape of the array
    pub shape: Vec<usize>,
    /// Data type of the array elements
    pub dtype: String,
    /// Element size in bytes
    pub element_size: usize,
    /// Compression algorithm used
    pub algorithm: String,
    /// Original data size in bytes
    pub original_size: usize,
    /// Compressed data size in bytes
    pub compressed_size: usize,
    /// Compression ratio (original_size / compressed_size)
    pub compression_ratio: f64,
    /// Compression level used
    pub compression_level: u32,
    /// Additional metadata as key-value pairs
    pub additional_metadata: std::collections::HashMap<String, String>,
}

/// Container for compressed array data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedArray {
    /// Metadata about the compressed array
    pub metadata: CompressedArrayMetadata,
    /// The compressed binary data
    pub data: Vec<u8>,
}

/// Compress an ndarray and serialize both data and metadata to a file
///
/// # Arguments
///
/// * `path` - Path to save the compressed array
/// * `array` - The array to compress
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, where 0 is no compression, 9 is maximum compression)
/// * `additional_metadata` - Optional additional metadata to include
///
/// # Returns
///
/// Result indicating success or failure
pub fn compress_array<P, A, S, D>(
    path: P,
    array: &ArrayBase<S, D>,
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
    additional_metadata: Option<std::collections::HashMap<String, String>>,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: Data<Elem = A>,
    D: Dimension + Serialize,
{
    // Convert array data to a flat vector for compression
    let flat_data: Vec<u8> =
        bincode::serialize(array).map_err(|e| IoError::SerializationError(e.to_string()))?;

    // Compress the flattened data
    let level = level.unwrap_or(6);
    let compressed_data = compress_data(&flat_data, algorithm, Some(level))?;

    // Create metadata
    let metadata = CompressedArrayMetadata {
        shape: array.shape().to_vec(),
        dtype: std::any::type_name::<A>().to_string(),
        element_size: std::mem::size_of::<A>(),
        algorithm: format!("{:?}", algorithm),
        original_size: flat_data.len(),
        compressed_size: compressed_data.len(),
        compression_ratio: flat_data.len() as f64 / compressed_data.len() as f64,
        compression_level: level,
        additional_metadata: additional_metadata.unwrap_or_default(),
    };

    // Create the complete compressed array structure
    let compressed_array = CompressedArray {
        metadata,
        data: compressed_data,
    };

    // Serialize and save to file
    let serialized = bincode::serialize(&compressed_array)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    File::create(path)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {}", e)))?
        .write_all(&serialized)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {}", e)))?;

    Ok(())
}

/// Decompress an array from a file
///
/// # Arguments
///
/// * `path` - Path to the compressed array file
///
/// # Returns
///
/// The decompressed array
pub fn decompress_array<P, A, D>(path: P) -> Result<ArrayBase<OwnedRepr<A>, D>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
    D: Dimension + for<'de> Deserialize<'de>,
{
    // Read the compressed file
    let mut file = File::open(path)
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {}", e)))?;

    let mut serialized = Vec::new();
    file.read_to_end(&mut serialized)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {}", e)))?;

    // Deserialize the compressed array structure
    let compressed_array: CompressedArray = bincode::deserialize(&serialized)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;

    // Determine algorithm from metadata
    let algorithm = match compressed_array.metadata.algorithm.as_str() {
        "Gzip" => CompressionAlgorithm::Gzip,
        "Zstd" => CompressionAlgorithm::Zstd,
        "Lz4" => CompressionAlgorithm::Lz4,
        "Bzip2" => CompressionAlgorithm::Bzip2,
        _ => {
            return Err(IoError::DecompressionError(format!(
                "Unknown compression algorithm: {}",
                compressed_array.metadata.algorithm
            )))
        }
    };

    // Decompress the data
    let decompressed_data = decompress_data(&compressed_array.data, algorithm)?;

    // Deserialize the array
    let array: ArrayBase<OwnedRepr<A>, D> = bincode::deserialize(&decompressed_data)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;

    Ok(array)
}

/// Compress an array in chunks for memory-efficient processing of large arrays
///
/// This function processes the array in chunks to avoid loading the entire
/// array into memory at once, which is useful for very large arrays.
///
/// # Arguments
///
/// * `path` - Path to save the compressed array
/// * `array` - The array to compress
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9)
/// * `chunk_size` - Size of chunks to process at once (number of elements)
///
/// # Returns
///
/// Result indicating success or failure
pub fn compress_array_chunked<P, A, S>(
    path: P,
    array: &ArrayBase<S, IxDyn>,
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
    chunk_size: usize,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: Data<Elem = A>,
{
    // Create a temporary buffer for chunked processing
    let mut compressed_chunks = Vec::new();
    let mut total_original_size = 0;
    let mut total_compressed_size = 0;

    // Process the array in chunks
    // Calculate ceiling division (equivalent to div_ceil in newer Rust versions)
    for chunk_idx in 0..((array.len() + chunk_size - 1) / chunk_size) {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(array.len());

        // Extract chunk data (this creates a copy, but only of the current chunk)
        let chunk_data: Vec<A> = array
            .iter()
            .skip(start)
            .take(end - start)
            .cloned()
            .collect();

        // Serialize the chunk
        let serialized_chunk = bincode::serialize(&chunk_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        // Compress the chunk
        let compressed_chunk = compress_data(&serialized_chunk, algorithm, level)?;

        // Track sizes
        total_original_size += serialized_chunk.len();
        total_compressed_size += compressed_chunk.len();

        // Add to compressed chunks collection
        compressed_chunks.push(compressed_chunk);
    }

    // Create metadata
    let metadata = CompressedArrayMetadata {
        shape: array.shape().to_vec(),
        dtype: std::any::type_name::<A>().to_string(),
        element_size: std::mem::size_of::<A>(),
        algorithm: format!("{:?}", algorithm),
        original_size: total_original_size,
        compressed_size: total_compressed_size,
        compression_ratio: total_original_size as f64 / total_compressed_size as f64,
        compression_level: level.unwrap_or(6),
        additional_metadata: {
            let mut map = std::collections::HashMap::new();
            map.insert("chunked".to_string(), "true".to_string());
            map.insert(
                "num_chunks".to_string(),
                compressed_chunks.len().to_string(),
            );
            map.insert("chunk_size".to_string(), chunk_size.to_string());
            map
        },
    };

    // Combine all chunks and metadata
    let mut file = File::create(path)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {}", e)))?;

    // Write metadata size and metadata first
    let serialized_metadata =
        bincode::serialize(&metadata).map_err(|e| IoError::SerializationError(e.to_string()))?;

    let metadata_size = serialized_metadata.len() as u64;
    file.write_all(&metadata_size.to_le_bytes())
        .map_err(|e| IoError::FileError(format!("Failed to write metadata size: {}", e)))?;

    file.write_all(&serialized_metadata)
        .map_err(|e| IoError::FileError(format!("Failed to write metadata: {}", e)))?;

    // Write number of chunks
    let num_chunks = compressed_chunks.len() as u64;
    file.write_all(&num_chunks.to_le_bytes())
        .map_err(|e| IoError::FileError(format!("Failed to write chunk count: {}", e)))?;

    // Write each chunk with its size prefix
    for chunk in compressed_chunks {
        let chunk_size = chunk.len() as u64;
        file.write_all(&chunk_size.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write chunk size: {}", e)))?;

        file.write_all(&chunk)
            .map_err(|e| IoError::FileError(format!("Failed to write chunk data: {}", e)))?;
    }

    Ok(())
}

/// Decompress an array that was compressed in chunks
///
/// # Arguments
///
/// * `path` - Path to the compressed array file
///
/// # Returns
///
/// The decompressed array and its metadata
pub fn decompress_array_chunked<P, A>(
    path: P,
) -> Result<(ArrayBase<OwnedRepr<A>, IxDyn>, CompressedArrayMetadata)>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
{
    let mut file = File::open(path)
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {}", e)))?;

    // Read metadata size
    let mut metadata_size_bytes = [0u8; 8];
    file.read_exact(&mut metadata_size_bytes)
        .map_err(|e| IoError::FileError(format!("Failed to read metadata size: {}", e)))?;

    let metadata_size = u64::from_le_bytes(metadata_size_bytes) as usize;

    // Read metadata
    let mut metadata_bytes = vec![0u8; metadata_size];
    file.read_exact(&mut metadata_bytes)
        .map_err(|e| IoError::FileError(format!("Failed to read metadata: {}", e)))?;

    let metadata: CompressedArrayMetadata = bincode::deserialize(&metadata_bytes)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;

    // Determine algorithm from metadata
    let algorithm = match metadata.algorithm.as_str() {
        "Gzip" => CompressionAlgorithm::Gzip,
        "Zstd" => CompressionAlgorithm::Zstd,
        "Lz4" => CompressionAlgorithm::Lz4,
        "Bzip2" => CompressionAlgorithm::Bzip2,
        _ => {
            return Err(IoError::DecompressionError(format!(
                "Unknown compression algorithm: {}",
                metadata.algorithm
            )))
        }
    };

    // Read number of chunks
    let mut num_chunks_bytes = [0u8; 8];
    file.read_exact(&mut num_chunks_bytes)
        .map_err(|e| IoError::FileError(format!("Failed to read chunk count: {}", e)))?;

    let num_chunks = u64::from_le_bytes(num_chunks_bytes) as usize;

    // Prepare to store all decompressed elements
    let total_elements: usize = metadata.shape.iter().product();
    let mut all_elements = Vec::with_capacity(total_elements);

    // Read and process each chunk
    for _ in 0..num_chunks {
        // Read chunk size
        let mut chunk_size_bytes = [0u8; 8];
        file.read_exact(&mut chunk_size_bytes)
            .map_err(|e| IoError::FileError(format!("Failed to read chunk size: {}", e)))?;

        let chunk_size = u64::from_le_bytes(chunk_size_bytes) as usize;

        // Read chunk data
        let mut chunk_bytes = vec![0u8; chunk_size];
        file.read_exact(&mut chunk_bytes)
            .map_err(|e| IoError::FileError(format!("Failed to read chunk data: {}", e)))?;

        // Decompress chunk
        let decompressed_chunk = decompress_data(&chunk_bytes, algorithm)?;

        // Deserialize chunk elements and add to our collection
        let chunk_elements: Vec<A> = bincode::deserialize(&decompressed_chunk)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?;

        all_elements.extend(chunk_elements);
    }

    // Construct the full array from all elements
    let array = ArrayBase::from_shape_vec(IxDyn(&metadata.shape), all_elements)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;

    Ok((array, metadata))
}

/// Returns compression statistics for a given array and set of algorithms
///
/// This is useful for determining which compression algorithm is most
/// effective for a particular dataset.
///
/// # Arguments
///
/// * `array` - The array to test
/// * `algorithms` - List of compression algorithms to test
/// * `level` - Compression level to use for all algorithms
///
/// # Returns
///
/// A vector of (algorithm, ratio, compressed_size) tuples
pub fn compare_compression_algorithms<A, S, D>(
    array: &ArrayBase<S, D>,
    algorithms: &[CompressionAlgorithm],
    level: Option<u32>,
) -> Result<Vec<(CompressionAlgorithm, f64, usize)>>
where
    A: Serialize + Clone,
    S: Data<Elem = A>,
    D: Dimension + Serialize,
{
    // Serialize the array once
    let serialized =
        bincode::serialize(array).map_err(|e| IoError::SerializationError(e.to_string()))?;

    let original_size = serialized.len();

    // Test each algorithm
    let mut results = Vec::new();

    for &algorithm in algorithms {
        // Compress with this algorithm
        let compressed = compress_data(&serialized, algorithm, level)?;
        let compressed_size = compressed.len();
        let ratio = original_size as f64 / compressed_size as f64;

        results.push((algorithm, ratio, compressed_size));
    }

    // Sort by compression ratio (best first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}
