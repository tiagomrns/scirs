//! Data validation and integrity checking utilities
//!
//! This module provides functionality for validating data integrity through
//! checksums, hash verification, and other validation methods.
//!
//! ## Features
//!
//! - Checksums (CRC32, MD5, SHA-256, BLAKE3)
//! - File integrity validation
//! - Data format validation
//! - Integrity metadata for scientific data

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use blake3::Hasher as Blake3Hasher;
use chrono::{DateTime, Utc};
use crc32fast::Hasher as CrcHasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{IoError, Result};

// Export submodules
pub mod formats;

/// Checksum algorithm types available for data validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// CRC32 - Fast but less secure, good for error detection
    CRC32,
    /// SHA-256 - Secure cryptographic hash
    SHA256,
    /// BLAKE3 - Modern, faster cryptographic hash
    BLAKE3,
}

impl ChecksumAlgorithm {
    /// Get a string representation of the algorithm
    pub fn as_str(&self) -> &'static str {
        match self {
            ChecksumAlgorithm::CRC32 => "CRC32",
            ChecksumAlgorithm::SHA256 => "SHA256",
            ChecksumAlgorithm::BLAKE3 => "BLAKE3",
        }
    }

    /// Parse algorithm name from a string
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "CRC32" => Some(ChecksumAlgorithm::CRC32),
            "SHA256" => Some(ChecksumAlgorithm::SHA256),
            "BLAKE3" => Some(ChecksumAlgorithm::BLAKE3),
            _ => None,
        }
    }
}

/// File integrity metadata used for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityMetadata {
    /// Size of the file in bytes
    pub size: u64,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Checksum value (hex encoded)
    pub checksum: String,
    /// Creation or last modification timestamp
    pub timestamp: u64,
    /// Additional integrity information
    pub additional_info: std::collections::HashMap<String, String>,
}

/// Calculate a checksum for the provided data
///
/// # Arguments
///
/// * `data` - The data to calculate the checksum for
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// The checksum as a hex encoded string
pub fn calculate_checksum(data: &[u8], algorithm: ChecksumAlgorithm) -> String {
    match algorithm {
        ChecksumAlgorithm::CRC32 => {
            let mut hasher = CrcHasher::new();
            hasher.update(data);
            format!("{:08x}", hasher.finalize())
        }
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hex::encode(hasher.finalize())
        }
        ChecksumAlgorithm::BLAKE3 => {
            let mut hasher = Blake3Hasher::new();
            hasher.update(data);
            hex::encode(hasher.finalize().as_bytes())
        }
    }
}

/// Calculate a checksum for a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// The checksum as a hex encoded string
pub fn calculate_file_checksum<P: AsRef<Path>>(
    path: P,
    algorithm: ChecksumAlgorithm,
) -> Result<String> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    match algorithm {
        ChecksumAlgorithm::CRC32 => {
            let mut hasher = CrcHasher::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(format!("{:08x}", hasher.finalize()))
        }
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(hex::encode(hasher.finalize()))
        }
        ChecksumAlgorithm::BLAKE3 => {
            let mut hasher = Blake3Hasher::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(hex::encode(hasher.finalize().as_bytes()))
        }
    }
}

/// Verify a checksum against the provided data
///
/// # Arguments
///
/// * `data` - The data to verify
/// * `expected_checksum` - The expected checksum value (hex encoded)
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// `true` if the checksum matches, `false` otherwise
pub fn verify_checksum(data: &[u8], expected_checksum: &str, algorithm: ChecksumAlgorithm) -> bool {
    let calculated = calculate_checksum(data, algorithm);
    calculated.eq_ignore_ascii_case(expected_checksum)
}

/// Verify a file's checksum
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `expected_checksum` - The expected checksum value (hex encoded)
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// `Ok(true)` if the checksum matches, `Ok(false)` otherwise, or an error
pub fn verify_file_checksum<P: AsRef<Path>>(
    path: P,
    expected_checksum: &str,
    algorithm: ChecksumAlgorithm,
) -> Result<bool> {
    let calculated = calculate_file_checksum(path, algorithm)?;
    Ok(calculated.eq_ignore_ascii_case(expected_checksum))
}

/// Generate integrity metadata for a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// A struct containing integrity metadata
pub fn generate_file_integrity_metadata<P: AsRef<Path>>(
    path: P,
    algorithm: ChecksumAlgorithm,
) -> Result<IntegrityMetadata> {
    let path = path.as_ref();

    // Get file metadata
    let file_metadata = std::fs::metadata(path)
        .map_err(|e| IoError::FileError(format!("Failed to read file metadata: {}", e)))?;

    let size = file_metadata.len();
    let modified = file_metadata
        .modified()
        .map_err(|e| IoError::FileError(format!("Failed to get modification time: {}", e)))?;

    // Convert to timestamp
    let timestamp = modified
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_err(|e| IoError::FileError(format!("Failed to convert time: {}", e)))?
        .as_secs();

    // Calculate checksum
    let checksum = calculate_file_checksum(path, algorithm)?;

    Ok(IntegrityMetadata {
        size,
        algorithm: algorithm.as_str().to_string(),
        checksum,
        timestamp,
        additional_info: std::collections::HashMap::new(),
    })
}

/// Save integrity metadata to a file
///
/// # Arguments
///
/// * `metadata` - The integrity metadata to save
/// * `path` - Path to save the metadata to
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_integrity_metadata<P: AsRef<Path>>(
    metadata: &IntegrityMetadata,
    path: P,
) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    serde_json::to_writer_pretty(file, metadata)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;
    Ok(())
}

/// Load integrity metadata from a file
///
/// # Arguments
///
/// * `path` - Path to the metadata file
///
/// # Returns
///
/// The loaded integrity metadata
pub fn load_integrity_metadata<P: AsRef<Path>>(path: P) -> Result<IntegrityMetadata> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);
    let metadata: IntegrityMetadata = serde_json::from_reader(reader)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;
    Ok(metadata)
}

/// Validate a file's integrity using metadata
///
/// # Arguments
///
/// * `file_path` - Path to the file to validate
/// * `metadata` - The integrity metadata to validate against
///
/// # Returns
///
/// Result with validation details:
/// - Ok(true) - Validation passed
/// - Ok(false) - Validation failed
/// - Err(e) - Error during validation
pub fn validate_file_integrity<P: AsRef<Path>>(
    file_path: P,
    metadata: &IntegrityMetadata,
) -> Result<bool> {
    let file_path = file_path.as_ref();

    // Check if file exists
    if !file_path.exists() {
        return Err(IoError::ValidationError(format!(
            "File does not exist: {}",
            file_path.display()
        )));
    }

    // Get file size
    let file_size = std::fs::metadata(file_path)
        .map_err(|e| IoError::FileError(e.to_string()))?
        .len();

    // Check file size
    if file_size != metadata.size {
        return Ok(false);
    }

    // Parse algorithm
    let algorithm = ChecksumAlgorithm::from_str(&metadata.algorithm).ok_or_else(|| {
        IoError::ValidationError(format!(
            "Unknown checksum algorithm: {}",
            metadata.algorithm
        ))
    })?;

    // Verify checksum
    verify_file_checksum(file_path, &metadata.checksum, algorithm)
}

/// Generate a validation report for a file
///
/// # Arguments
///
/// * `file_path` - Path to the file to validate
/// * `metadata` - The integrity metadata to validate against
///
/// # Returns
///
/// A struct containing validation results
pub fn generate_validation_report<P: AsRef<Path>>(
    file_path: P,
    metadata: &IntegrityMetadata,
) -> Result<ValidationReport> {
    let file_path = file_path.as_ref();

    // Check if file exists
    if !file_path.exists() {
        return Err(IoError::ValidationError(format!(
            "File does not exist: {}",
            file_path.display()
        )));
    }

    // Get file metadata
    let file_metadata = std::fs::metadata(file_path)
        .map_err(|e| IoError::FileError(format!("Failed to read file metadata: {}", e)))?;

    let actual_size = file_metadata.len();
    let size_valid = actual_size == metadata.size;

    // Parse algorithm
    let algorithm = match ChecksumAlgorithm::from_str(&metadata.algorithm) {
        Some(algo) => algo,
        None => {
            return Err(IoError::ValidationError(format!(
                "Unknown checksum algorithm: {}",
                metadata.algorithm
            )))
        }
    };

    // Calculate and check checksum
    let actual_checksum = calculate_file_checksum(file_path, algorithm)?;
    let checksum_valid = actual_checksum.eq_ignore_ascii_case(&metadata.checksum);

    // Overall validity
    let valid = size_valid && checksum_valid;

    Ok(ValidationReport {
        file_path: file_path.to_string_lossy().to_string(),
        expected_size: metadata.size,
        actual_size,
        size_valid,
        expected_checksum: metadata.checksum.clone(),
        actual_checksum,
        checksum_valid,
        algorithm: metadata.algorithm.clone(),
        valid,
        validation_time: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

/// Report containing the results of a file validation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Path to the validated file
    pub file_path: String,
    /// Expected file size in bytes
    pub expected_size: u64,
    /// Actual file size in bytes
    pub actual_size: u64,
    /// Whether the size check passed
    pub size_valid: bool,
    /// Expected checksum value
    pub expected_checksum: String,
    /// Actual calculated checksum value
    pub actual_checksum: String,
    /// Whether the checksum check passed
    pub checksum_valid: bool,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Overall validation result
    pub valid: bool,
    /// Time of validation (Unix timestamp)
    pub validation_time: u64,
}

impl ValidationReport {
    /// Get a formatted validation report as a string
    pub fn format(&self) -> String {
        let status = if self.valid { "PASSED" } else { "FAILED" };

        format!(
            "Validation Report ({})\n\
             -------------------------------\n\
             File: {}\n\
             Algorithm: {}\n\
             Size Check: {} (Expected: {} bytes, Found: {} bytes)\n\
             Checksum Check: {} (Expected: {}, Found: {})\n\
             Validation Time: {}\n",
            status,
            self.file_path,
            self.algorithm,
            if self.size_valid { "PASSED" } else { "FAILED" },
            self.expected_size,
            self.actual_size,
            if self.checksum_valid {
                "PASSED"
            } else {
                "FAILED"
            },
            self.expected_checksum,
            self.actual_checksum,
            DateTime::<Utc>::from_timestamp(self.validation_time as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "Unknown".to_string()),
        )
    }

    /// Save the validation report to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        Ok(())
    }
}

/// Data source used for format validation
pub enum ValidationSource<'a> {
    /// Raw data bytes
    Data(&'a [u8]),
    /// File path
    FilePath(&'a Path),
}

/// Format validation rule with type and validation function
pub struct FormatValidator {
    /// Name of the format
    pub format_name: String,
    /// Function to validate the format
    validator: Box<dyn Fn(&[u8]) -> bool + Send + Sync>, // Type complexity is necessary here
}

impl FormatValidator {
    /// Create a new format validator
    pub fn new<F>(format_name: &str, validator: F) -> Self
    where
        F: Fn(&[u8]) -> bool + Send + Sync + 'static,
    {
        Self {
            format_name: format_name.to_string(),
            validator: Box::new(validator),
        }
    }

    /// Validate data against this format
    pub fn validate(&self, data: &[u8]) -> bool {
        (self.validator)(data)
    }
}

/// Registry of available format validators
pub struct FormatValidatorRegistry {
    validators: Vec<FormatValidator>,
}

impl Default for FormatValidatorRegistry {
    fn default() -> Self {
        let mut registry = Self {
            validators: Vec::new(),
        };

        // Add default validators
        registry.add_default_validators();

        registry
    }
}

impl FormatValidatorRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the registry
    pub fn add_validator<F>(&mut self, format_name: &str, validator: F)
    where
        F: Fn(&[u8]) -> bool + Send + Sync + 'static,
    {
        self.validators
            .push(FormatValidator::new(format_name, validator));
    }

    /// Check if data matches any registered format
    pub fn validate_format(&self, source: ValidationSource) -> Result<Option<String>> {
        // Get data as bytes
        let data = match source {
            ValidationSource::Data(bytes) => bytes.to_vec(),
            ValidationSource::FilePath(path) => {
                let mut file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;

                // Read first 8192 bytes for format detection
                let mut buffer = Vec::with_capacity(8192);
                file.read_to_end(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                buffer
            }
        };

        // Check all validators
        for validator in &self.validators {
            if validator.validate(&data) {
                return Ok(Some(validator.format_name.clone()));
            }
        }

        Ok(None)
    }

    /// Add default format validators
    fn add_default_validators(&mut self) {
        // PNG validator
        self.add_validator("PNG", |data| {
            data.len() >= 8 && data[0..8] == [137, 80, 78, 71, 13, 10, 26, 10]
        });

        // JPEG validator
        self.add_validator("JPEG", |data| {
            data.len() >= 3 && data[0..3] == [0xFF, 0xD8, 0xFF]
        });

        // GZIP validator
        self.add_validator("GZIP", |data| data.len() >= 2 && data[0..2] == [0x1F, 0x8B]);

        // ZIP validator
        self.add_validator("ZIP", |data| {
            data.len() >= 4 && data[0..4] == [0x50, 0x4B, 0x03, 0x04]
        });

        // JSON validator (very basic check)
        self.add_validator("JSON", |data| {
            if data.is_empty() {
                return false;
            }

            // Look for { or [ as first non-whitespace
            for &byte in data {
                if byte == b'{' || byte == b'[' {
                    return true;
                }
                if !byte.is_ascii_whitespace() {
                    return false;
                }
            }

            false
        });

        // CSV validator (very basic check)
        self.add_validator("CSV", |data| {
            if data.is_empty() {
                return false;
            }

            // Check for commas and newlines
            let has_comma = data.contains(&b',');
            let has_newline = data.contains(&b'\n');

            has_comma && has_newline
        });
    }
}

/// Check if a file exists and has the expected size
pub fn validate_file_exists_with_size<P: AsRef<Path>>(
    path: P,
    expected_size: Option<u64>,
) -> Result<bool> {
    let path = path.as_ref();

    if !path.exists() {
        return Ok(false);
    }

    if let Some(size) = expected_size {
        let file_size = std::fs::metadata(path)
            .map_err(|e| IoError::FileError(e.to_string()))?
            .len();

        Ok(file_size == size)
    } else {
        Ok(true)
    }
}

/// Utility to create a checksum file for a data file
///
/// # Arguments
///
/// * `data_path` - Path to the data file
/// * `algorithm` - The checksum algorithm to use
/// * `output_path` - Optional path to save the checksum file (if None, uses data_path + ".checksum")
///
/// # Returns
///
/// Result with the path to the checksum file
pub fn create_checksum_file<P, Q>(
    data_path: P,
    algorithm: ChecksumAlgorithm,
    output_path: Option<Q>,
) -> Result<String>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let data_path = data_path.as_ref();

    // Calculate checksum
    let checksum = calculate_file_checksum(data_path, algorithm)?;

    // Determine output path
    let output_path = match output_path {
        Some(path) => path.as_ref().to_path_buf(),
        None => {
            let mut path = data_path.to_path_buf();
            path.set_extension(format!(
                "{}.checksum",
                path.extension().unwrap_or_default().to_string_lossy()
            ));
            path
        }
    };

    // Generate content
    let content = format!(
        "{} *{}\n",
        checksum,
        data_path.file_name().unwrap_or_default().to_string_lossy()
    );

    // Write checksum file
    let mut file = File::create(&output_path)
        .map_err(|e| IoError::FileError(format!("Failed to create checksum file: {}", e)))?;

    file.write_all(content.as_bytes())
        .map_err(|e| IoError::FileError(format!("Failed to write checksum file: {}", e)))?;

    Ok(output_path.to_string_lossy().to_string())
}

/// Verify a file against a checksum file
///
/// # Arguments
///
/// * `data_path` - Path to the data file
/// * `checksum_path` - Path to the checksum file
///
/// # Returns
///
/// Result indicating if the verification passed
pub fn verify_checksum_file<P, Q>(data_path: P, checksum_path: Q) -> Result<bool>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let data_path = data_path.as_ref();

    // Read checksum file
    let mut checksum_file = File::open(checksum_path)
        .map_err(|e| IoError::FileError(format!("Failed to open checksum file: {}", e)))?;

    let mut content = String::new();
    checksum_file
        .read_to_string(&mut content)
        .map_err(|e| IoError::FileError(format!("Failed to read checksum file: {}", e)))?;

    // Parse checksum file (format: "<checksum> *<filename>")
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(IoError::ValidationError(
            "Invalid checksum file format".to_string(),
        ));
    }

    let expected_checksum = parts[0];

    // Determine algorithm from checksum length
    let algorithm = match expected_checksum.len() {
        8 => ChecksumAlgorithm::CRC32,
        64 => ChecksumAlgorithm::SHA256,
        // BLAKE3 produces 64 hex chars by default
        _ => {
            return Err(IoError::ValidationError(format!(
                "Unable to determine checksum algorithm from length: {}",
                expected_checksum.len()
            )))
        }
    };

    // Calculate actual checksum
    let actual_checksum = calculate_file_checksum(data_path, algorithm)?;

    // Compare checksums
    Ok(actual_checksum.eq_ignore_ascii_case(expected_checksum))
}

/// Add integrity metadata to an array of objects
pub fn add_integrity_metadata<T: Serialize>(
    data: &[T],
    algorithm: ChecksumAlgorithm,
) -> Result<std::collections::HashMap<String, String>> {
    // Serialize the data to calculate checksum
    let serialized =
        serde_json::to_vec(data).map_err(|e| IoError::SerializationError(e.to_string()))?;

    // Calculate checksum
    let checksum = calculate_checksum(&serialized, algorithm);

    // Create metadata
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("algorithm".to_string(), algorithm.as_str().to_string());
    metadata.insert("checksum".to_string(), checksum);
    metadata.insert("length".to_string(), data.len().to_string());
    metadata.insert(
        "timestamp".to_string(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string(),
    );

    Ok(metadata)
}

/// Verify array integrity using metadata
pub fn verify_array_integrity<T: Serialize>(
    data: &[T],
    metadata: &std::collections::HashMap<String, String>,
) -> Result<bool> {
    // Check array length
    if let Some(length) = metadata.get("length") {
        if let Ok(expected_length) = length.parse::<usize>() {
            if data.len() != expected_length {
                return Ok(false);
            }
        } else {
            return Err(IoError::ValidationError(
                "Invalid length in metadata".to_string(),
            ));
        }
    } else {
        return Err(IoError::ValidationError(
            "Missing length in metadata".to_string(),
        ));
    }

    // Get algorithm and checksum
    let algorithm_str = metadata
        .get("algorithm")
        .ok_or_else(|| IoError::ValidationError("Missing algorithm in metadata".to_string()))?;

    let algorithm = ChecksumAlgorithm::from_str(algorithm_str).ok_or_else(|| {
        IoError::ValidationError(format!("Unknown algorithm in metadata: {}", algorithm_str))
    })?;

    let expected_checksum = metadata
        .get("checksum")
        .ok_or_else(|| IoError::ValidationError("Missing checksum in metadata".to_string()))?;

    // Serialize the data to calculate checksum
    let serialized =
        serde_json::to_vec(data).map_err(|e| IoError::SerializationError(e.to_string()))?;

    // Calculate and compare checksums
    let actual_checksum = calculate_checksum(&serialized, algorithm);

    Ok(actual_checksum.eq_ignore_ascii_case(expected_checksum))
}

/// Create a manifest file for a directory with checksums
pub fn create_directory_manifest<P, Q>(
    dir_path: P,
    output_path: Q,
    algorithm: ChecksumAlgorithm,
    recursive: bool,
) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let dir_path = dir_path.as_ref();

    // Check if directory exists
    if !dir_path.is_dir() {
        return Err(IoError::ValidationError(format!(
            "Not a directory: {}",
            dir_path.display()
        )));
    }

    // Collect all files
    let mut files = Vec::new();
    collect_files(dir_path, &mut files, recursive)?;

    // Calculate checksums and create manifest entries
    let mut manifest = Vec::new();

    for file_path in files {
        let relative_path = file_path
            .strip_prefix(dir_path)
            .map_err(|e| IoError::ValidationError(e.to_string()))?;

        let checksum = calculate_file_checksum(&file_path, algorithm)?;
        let size = std::fs::metadata(&file_path)
            .map_err(|e| IoError::FileError(e.to_string()))?
            .len();

        manifest.push(ManifestEntry {
            path: relative_path.to_string_lossy().to_string(),
            size,
            checksum,
        });
    }

    // Create manifest file
    let manifest_file = DirectoryManifest {
        directory: dir_path.to_string_lossy().to_string(),
        algorithm: algorithm.as_str().to_string(),
        creation_time: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        files: manifest,
    };

    // Save to output file
    let file = File::create(output_path).map_err(|e| IoError::FileError(e.to_string()))?;

    serde_json::to_writer_pretty(file, &manifest_file)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    Ok(())
}

/// Helper function to collect files in a directory
fn collect_files(dir: &Path, files: &mut Vec<std::path::PathBuf>, recursive: bool) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .map_err(|e| IoError::FileError(format!("Failed to read directory: {}", e)))?
    {
        let entry = entry.map_err(|e| IoError::FileError(e.to_string()))?;
        let path = entry.path();

        if path.is_file() {
            files.push(path);
        } else if path.is_dir() && recursive {
            collect_files(&path, files, recursive)?;
        }
    }

    Ok(())
}

/// Entry in a directory manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Relative path to the file
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Checksum value
    pub checksum: String,
}

/// Manifest file for a directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryManifest {
    /// Directory path
    pub directory: String,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Creation time (Unix timestamp)
    pub creation_time: u64,
    /// Files in the directory
    pub files: Vec<ManifestEntry>,
}

impl DirectoryManifest {
    /// Verify a directory against the manifest
    pub fn verify_directory<P: AsRef<Path>>(
        &self,
        dir_path: P,
    ) -> Result<ManifestVerificationReport> {
        let dir_path = dir_path.as_ref();

        // Check if directory exists
        if !dir_path.is_dir() {
            return Err(IoError::ValidationError(format!(
                "Not a directory: {}",
                dir_path.display()
            )));
        }

        // Parse algorithm
        let algorithm = ChecksumAlgorithm::from_str(&self.algorithm).ok_or_else(|| {
            IoError::ValidationError(format!("Unknown algorithm in manifest: {}", self.algorithm))
        })?;

        // Check each file
        let mut missing_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut verified_files = Vec::new();

        for entry in &self.files {
            let file_path = dir_path.join(&entry.path);

            // Check if file exists
            if !file_path.exists() {
                missing_files.push(entry.path.clone());
                continue;
            }

            // Check file size
            let file_size = std::fs::metadata(&file_path)
                .map_err(|e| IoError::FileError(e.to_string()))?
                .len();

            if file_size != entry.size {
                modified_files.push(entry.path.clone());
                continue;
            }

            // Check checksum
            let checksum = calculate_file_checksum(&file_path, algorithm)?;
            if !checksum.eq_ignore_ascii_case(&entry.checksum) {
                modified_files.push(entry.path.clone());
                continue;
            }

            // File verified
            verified_files.push(entry.path.clone());
        }

        Ok(ManifestVerificationReport {
            directory: dir_path.to_string_lossy().to_string(),
            total_files: self.files.len(),
            verified_files,
            missing_files,
            modified_files,
            verification_time: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
}

/// Report from a manifest verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestVerificationReport {
    /// Directory that was verified
    pub directory: String,
    /// Total number of files in the manifest
    pub total_files: usize,
    /// Files that were successfully verified
    pub verified_files: Vec<String>,
    /// Files that were missing
    pub missing_files: Vec<String>,
    /// Files that were modified
    pub modified_files: Vec<String>,
    /// Time of verification (Unix timestamp)
    pub verification_time: u64,
}

impl ManifestVerificationReport {
    /// Check if the verification passed (all files verified)
    pub fn passed(&self) -> bool {
        self.missing_files.is_empty() && self.modified_files.is_empty()
    }

    /// Get a formatted report as a string
    pub fn format(&self) -> String {
        let status = if self.passed() { "PASSED" } else { "FAILED" };

        let mut report = format!(
            "Manifest Verification Report ({})\n\
             -------------------------------------\n\
             Directory: {}\n\
             Total Files: {}\n\
             Verified: {} files\n",
            status,
            self.directory,
            self.total_files,
            self.verified_files.len(),
        );

        if !self.missing_files.is_empty() {
            report.push_str(&format!(
                "\nMissing Files ({}):\n",
                self.missing_files.len()
            ));
            for file in &self.missing_files {
                report.push_str(&format!("  - {}\n", file));
            }
        }

        if !self.modified_files.is_empty() {
            report.push_str(&format!(
                "\nModified Files ({}):\n",
                self.modified_files.len()
            ));
            for file in &self.modified_files {
                report.push_str(&format!("  - {}\n", file));
            }
        }

        report.push_str(&format!(
            "\nVerification Time: {}\n",
            DateTime::<Utc>::from_timestamp(self.verification_time as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "Unknown".to_string()),
        ));

        report
    }

    /// Save the report to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        Ok(())
    }
}
