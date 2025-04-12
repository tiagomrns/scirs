//! Compression utilities for scientific data
//!
//! This module provides functionality for compressing and decompressing data using
//! various algorithms suitable for scientific computing. It focuses on lossless
//! compression to ensure data integrity while reducing storage requirements.
//!
//! ## Features
//!
//! - Multiple compression algorithms (GZIP, ZSTD, LZ4, BZIP2)
//! - Configurable compression levels
//! - Memory-efficient compression of large datasets
//! - Metadata preservation during compression
//! - Array-specific compression optimizations
//!
//! ## Sub-modules
//!
//! - `ndarray`: Specialized compression utilities for ndarray types
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::compression::{compress_data, decompress_data, CompressionAlgorithm};
//! use std::fs::File;
//! use std::io::prelude::*;
//!
//! // Compress some data using ZSTD with default compression level
//! let data = b"Large scientific dataset with repetitive patterns";
//! let compressed = compress_data(data, CompressionAlgorithm::Zstd, None).unwrap();
//!
//! // Save the compressed data to a file
//! let mut file = File::create("data.zst").unwrap();
//! file.write_all(&compressed).unwrap();
//!
//! // Later, read and decompress the data
//! let mut compressed_data = Vec::new();
//! File::open("data.zst").unwrap().read_to_end(&mut compressed_data).unwrap();
//! let original = decompress_data(&compressed_data, CompressionAlgorithm::Zstd).unwrap();
//! assert_eq!(original, data);
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use bzip2::read::{BzDecoder, BzEncoder};
use bzip2::Compression as Bzip2Compression;
use flate2::read::{GzDecoder, GzEncoder};
use flate2::Compression as GzipCompression;
use lz4::{Decoder, EncoderBuilder};
use zstd::{decode_all, encode_all};

// Re-export ndarray submodule
pub mod ndarray;

use crate::error::{IoError, Result};

/// Compression algorithms supported by the library
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// GZIP compression (good balance of speed and compression ratio)
    Gzip,
    /// Zstandard compression (excellent compression ratio, fast decompression)
    Zstd,
    /// LZ4 compression (extremely fast, moderate compression ratio)
    Lz4,
    /// BZIP2 compression (high compression ratio, slower speed)
    Bzip2,
}

impl CompressionAlgorithm {
    /// Get the file extension associated with this compression algorithm
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionAlgorithm::Gzip => "gz",
            CompressionAlgorithm::Zstd => "zst",
            CompressionAlgorithm::Lz4 => "lz4",
            CompressionAlgorithm::Bzip2 => "bz2",
        }
    }

    /// Try to determine the compression algorithm from a file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "gz" | "gzip" => Some(CompressionAlgorithm::Gzip),
            "zst" | "zstd" => Some(CompressionAlgorithm::Zstd),
            "lz4" => Some(CompressionAlgorithm::Lz4),
            "bz2" | "bzip2" => Some(CompressionAlgorithm::Bzip2),
            _ => None,
        }
    }
}

/// Convert a compression level (0-9) to the appropriate internal level for each algorithm
fn normalize_compression_level(level: Option<u32>, algorithm: CompressionAlgorithm) -> Result<u32> {
    let level = level.unwrap_or(6); // Default compression level

    if level > 9 {
        return Err(IoError::CompressionError(format!(
            "Compression level must be between 0 and 9, got {}",
            level
        )));
    }

    // Each compression library has different ranges for compression levels
    match algorithm {
        CompressionAlgorithm::Gzip => Ok(level),
        CompressionAlgorithm::Zstd => {
            // ZSTD supports levels 1-22, map our 0-9 to 1-22
            Ok(1 + (level * 21) / 9)
        }
        CompressionAlgorithm::Lz4 => Ok(level),
        CompressionAlgorithm::Bzip2 => Ok(level),
    }
}

/// Compress data using the specified algorithm and compression level
///
/// # Arguments
///
/// * `data` - The data to compress
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, where 0 is no compression, 9 is maximum compression)
///
/// # Returns
///
/// The compressed data as a `Vec<u8>`
pub fn compress_data(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<Vec<u8>> {
    let normalized_level = normalize_compression_level(level, algorithm)?;

    match algorithm {
        CompressionAlgorithm::Gzip => {
            let mut encoder = GzEncoder::new(data, GzipCompression::new(normalized_level as u32));
            let mut compressed = Vec::new();
            encoder
                .read_to_end(&mut compressed)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;
            Ok(compressed)
        }
        CompressionAlgorithm::Zstd => encode_all(data, normalized_level as i32)
            .map_err(|e| IoError::CompressionError(e.to_string())),
        CompressionAlgorithm::Lz4 => {
            let mut encoder = EncoderBuilder::new()
                .level(normalized_level as u32)
                .build(Vec::new())
                .map_err(|e| IoError::CompressionError(e.to_string()))?;

            encoder
                .write_all(data)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;

            let (compressed, result) = encoder.finish();
            result.map_err(|e| IoError::CompressionError(e.to_string()))?;

            Ok(compressed)
        }
        CompressionAlgorithm::Bzip2 => {
            let mut encoder = BzEncoder::new(data, Bzip2Compression::new(normalized_level as u32));
            let mut compressed = Vec::new();
            encoder
                .read_to_end(&mut compressed)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;
            Ok(compressed)
        }
    }
}

/// Decompress data using the specified algorithm
///
/// # Arguments
///
/// * `data` - The compressed data
/// * `algorithm` - The compression algorithm used
///
/// # Returns
///
/// The decompressed data as a `Vec<u8>`
pub fn decompress_data(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Gzip => {
            let mut decoder = GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Zstd => {
            decode_all(data).map_err(|e| IoError::DecompressionError(e.to_string()))
        }
        CompressionAlgorithm::Lz4 => {
            let mut decoder =
                Decoder::new(data).map_err(|e| IoError::DecompressionError(e.to_string()))?;
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Bzip2 => {
            let mut decoder = BzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
    }
}

/// Compress a file using the specified algorithm and save it to a new file
///
/// # Arguments
///
/// * `input_path` - Path to the file to compress
/// * `output_path` - Path to save the compressed file (if None, appends algorithm extension to input_path)
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, where 0 is no compression, 9 is maximum compression)
///
/// # Returns
///
/// The path to the compressed file
pub fn compress_file<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<String> {
    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {}", e)))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {}", e)))?;

    // Compress the data
    let compressed_data = compress_data(&input_data, algorithm, level)?;

    // Determine output path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            // Generate output path by appending algorithm extension
            let mut path_buf = input_path.as_ref().to_path_buf();
            let ext = algorithm.extension();

            // Get the file name as a string
            let file_name = path_buf
                .file_name()
                .ok_or_else(|| IoError::FileError("Invalid input file path".to_string()))?
                .to_string_lossy()
                .to_string();

            // Append the extension and update the file name
            let new_file_name = format!("{}.{}", file_name, ext);
            path_buf.set_file_name(new_file_name);

            path_buf.to_string_lossy().to_string()
        }
    };

    // Write the compressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {}", e)))?
        .write_all(&compressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {}", e)))?;

    Ok(output_path_string)
}

/// Decompress a file using the specified algorithm and save it to a new file
///
/// # Arguments
///
/// * `input_path` - Path to the compressed file
/// * `output_path` - Path to save the decompressed file (if None, removes algorithm extension from input_path)
/// * `algorithm` - The compression algorithm to use (if None, tries to determine from file extension)
///
/// # Returns
///
/// The path to the decompressed file
pub fn decompress_file<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: Option<CompressionAlgorithm>,
) -> Result<String> {
    // Determine the compression algorithm
    let algorithm = match algorithm {
        Some(algo) => algo,
        None => {
            // Try to determine from the file extension
            let ext = input_path
                .as_ref()
                .extension()
                .ok_or_else(|| {
                    IoError::DecompressionError("Unable to determine file extension".to_string())
                })?
                .to_string_lossy()
                .to_string();

            CompressionAlgorithm::from_extension(&ext)
                .ok_or(IoError::UnsupportedCompressionAlgorithm(ext))?
        }
    };

    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {}", e)))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {}", e)))?;

    // Decompress the data
    let decompressed_data = decompress_data(&input_data, algorithm)?;

    // Determine output path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            // Generate output path by removing algorithm extension
            let path_str = input_path.as_ref().to_string_lossy().to_string();
            let ext = algorithm.extension();

            if path_str.ends_with(&format!(".{}", ext)) {
                // Remove the extension
                path_str[0..path_str.len() - ext.len() - 1].to_string()
            } else {
                // If the extension doesn't match, add a ".decompressed" suffix
                format!("{}.decompressed", path_str)
            }
        }
    };

    // Write the decompressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {}", e)))?
        .write_all(&decompressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {}", e)))?;

    Ok(output_path_string)
}

/// Calculate the compression ratio for the given data and algorithm
///
/// # Arguments
///
/// * `data` - The original data
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, optional)
///
/// # Returns
///
/// The compression ratio (original size / compressed size)
pub fn compression_ratio(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<f64> {
    let compressed = compress_data(data, algorithm, level)?;
    let original_size = data.len() as f64;
    let compressed_size = compressed.len() as f64;

    // Avoid division by zero
    if compressed_size == 0.0 {
        return Err(IoError::CompressionError(
            "Compressed data has zero size".to_string(),
        ));
    }

    Ok(original_size / compressed_size)
}

/// Information about a compression algorithm
pub struct CompressionInfo {
    /// Name of the compression algorithm
    pub name: String,
    /// Brief description of the algorithm
    pub description: String,
    /// Typical compression ratio for scientific data (higher is better)
    pub typical_compression_ratio: f64,
    /// Relative compression speed (1-10, higher is faster)
    pub compression_speed: u8,
    /// Relative decompression speed (1-10, higher is faster)
    pub decompression_speed: u8,
    /// File extension associated with this compression
    pub file_extension: String,
}

/// Get information about a specific compression algorithm
pub fn algorithm_info(algorithm: CompressionAlgorithm) -> CompressionInfo {
    match algorithm {
        CompressionAlgorithm::Gzip => CompressionInfo {
            name: "GZIP".to_string(),
            description: "General-purpose compression algorithm with good balance of speed and compression ratio".to_string(),
            typical_compression_ratio: 2.5,
            compression_speed: 6,
            decompression_speed: 7,
            file_extension: "gz".to_string(),
        },
        CompressionAlgorithm::Zstd => CompressionInfo {
            name: "Zstandard".to_string(),
            description: "Modern compression algorithm with excellent compression ratio and fast decompression".to_string(),
            typical_compression_ratio: 3.2,
            compression_speed: 7,
            decompression_speed: 9,
            file_extension: "zst".to_string(),
        },
        CompressionAlgorithm::Lz4 => CompressionInfo {
            name: "LZ4".to_string(),
            description: "Extremely fast compression algorithm with moderate compression ratio".to_string(),
            typical_compression_ratio: 1.8,
            compression_speed: 10,
            decompression_speed: 10,
            file_extension: "lz4".to_string(),
        },
        CompressionAlgorithm::Bzip2 => CompressionInfo {
            name: "BZIP2".to_string(),
            description: "High compression ratio but slower speed, good for archival storage".to_string(),
            typical_compression_ratio: 3.5,
            compression_speed: 3,
            decompression_speed: 4,
            file_extension: "bz2".to_string(),
        },
    }
}
