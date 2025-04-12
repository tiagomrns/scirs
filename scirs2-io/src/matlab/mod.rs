//! MATLAB file format (.mat) handling module
//!
//! This module provides functionality for reading and writing MATLAB .mat files.
//! Currently supporting MATLAB v5 format (Level 5 MAT-File).

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// MATLAB data types
#[derive(Debug, Clone)]
pub enum MatType {
    /// Double-precision floating point
    Double(ArrayD<f64>),
    /// Single-precision floating point
    Single(ArrayD<f32>),
    /// 8-bit signed integer
    Int8(ArrayD<i8>),
    /// 16-bit signed integer
    Int16(ArrayD<i16>),
    /// 32-bit signed integer
    Int32(ArrayD<i32>),
    /// 64-bit signed integer
    Int64(ArrayD<i64>),
    /// 8-bit unsigned integer
    UInt8(ArrayD<u8>),
    /// 16-bit unsigned integer
    UInt16(ArrayD<u16>),
    /// 32-bit unsigned integer
    UInt32(ArrayD<u32>),
    /// 64-bit unsigned integer
    UInt64(ArrayD<u64>),
    /// Logical/boolean
    Logical(ArrayD<bool>),
    /// Character array
    Char(String),
    /// Cell array
    Cell(Vec<MatType>),
    /// Structure
    Struct(HashMap<String, MatType>),
}

/// MATLAB header information
#[derive(Debug, Clone)]
struct MatHeader {
    /// Version of the MAT file
    _version: u16,
    /// Endianness indicator
    endian_indicator: u16,
}

// MATLAB data type identifiers (miTYPE values)
const MI_INT8: i32 = 1;
const _MI_UINT8: i32 = 2;
const _MI_INT16: i32 = 3;
const _MI_UINT16: i32 = 4;
const MI_INT32: i32 = 5;
const MI_UINT32: i32 = 6;
const _MI_SINGLE: i32 = 7;
const _MI_DOUBLE: i32 = 9;
const _MI_INT64: i32 = 12;
const _MI_UINT64: i32 = 13;
const MI_MATRIX: i32 = 14;
const _MI_COMPRESSED: i32 = 15;
const MI_UTF8: i32 = 16;
const _MI_UTF16: i32 = 17;
const _MI_UTF32: i32 = 18;

// MATLAB array type values (mxCLASS values)
const _MX_CELL_CLASS: i32 = 1;
const _MX_STRUCT_CLASS: i32 = 2;
const _MX_OBJECT_CLASS: i32 = 3;
const MX_CHAR_CLASS: i32 = 4;
const _MX_SPARSE_CLASS: i32 = 5;
const MX_DOUBLE_CLASS: i32 = 6;
const MX_SINGLE_CLASS: i32 = 7;
const MX_INT8_CLASS: i32 = 8;
const MX_UINT8_CLASS: i32 = 9;
const MX_INT16_CLASS: i32 = 10;
const MX_UINT16_CLASS: i32 = 11;
const MX_INT32_CLASS: i32 = 12;
const MX_UINT32_CLASS: i32 = 13;
const MX_INT64_CLASS: i32 = 14;
const MX_UINT64_CLASS: i32 = 15;

/// Matrix flags for MATLAB data
#[derive(Debug, Clone)]
struct MatrixFlags {
    /// Class type (double, single, etc.)
    class_type: i32,
    /// Whether the matrix is complex
    is_complex: bool,
    /// Whether the matrix is a global variable
    _is_global: bool,
    /// Whether the matrix is logical
    _is_logical: bool,
}

impl MatrixFlags {
    /// Parse matrix flags from a u32
    fn from_u32(flags: u32) -> Self {
        let class_type = (flags & 0xFF) as i32;
        let is_complex = (flags & 0x800) != 0;
        let _is_global = (flags & 0x400) != 0;
        let _is_logical = (flags & 0x200) != 0;

        MatrixFlags {
            class_type,
            is_complex,
            _is_global,
            _is_logical,
        }
    }

    /// Convert to a u32 for writing
    fn _to_u32(&self) -> u32 {
        let mut flags = self.class_type as u32;
        if self.is_complex {
            flags |= 0x800;
        }
        if self._is_global {
            flags |= 0x400;
        }
        if self._is_logical {
            flags |= 0x200;
        }
        flags
    }
}

/// Data element for MATLAB file
#[derive(Debug, Clone)]
struct _DataElement {
    /// Data type
    data_type: i32,
    /// Data (as bytes)
    data: Vec<u8>,
}

/// Matrix array for MATLAB file
#[derive(Debug, Clone)]
struct _MatrixArray {
    /// Matrix flags
    flags: MatrixFlags,
    /// Array dimensions
    dims: Vec<i32>,
    /// Array name
    name: String,
    /// Real data
    real_data: Vec<u8>,
    /// Imaginary data (if complex)
    imag_data: Option<Vec<u8>>,
}

/// Reads a MATLAB .mat file
///
/// # Arguments
///
/// * `path` - Path to the .mat file
///
/// # Returns
///
/// * A HashMap mapping variable names to their values
///
/// # Example
///
/// ```no_run
/// use scirs2_io::matlab::read_mat;
/// use std::path::Path;
///
/// let vars = read_mat(Path::new("data.mat")).unwrap();
/// for (name, _) in vars.iter() {
///     println!("Variable: {}", name);
/// }
/// ```
pub fn read_mat<P: AsRef<Path>>(path: P) -> Result<HashMap<String, MatType>> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Read the MAT file header (116 bytes)
    let mut header_bytes = [0u8; 116];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|e| IoError::FileError(format!("Failed to read MAT header: {}", e)))?;

    // Check magic string "MATLAB"
    let magic = std::str::from_utf8(&header_bytes[0..6])
        .map_err(|_| IoError::FormatError("Invalid MAT file header".to_string()))?;

    if magic != "MATLAB" {
        return Err(IoError::FormatError("Not a valid MATLAB file".to_string()));
    }

    // Parse version and endianness
    let subsystem_data_offset = &header_bytes[108..116];
    let version = LittleEndian::read_u16(&subsystem_data_offset[0..2]);
    let endian_indicator = LittleEndian::read_u16(&subsystem_data_offset[2..4]);

    let header = MatHeader {
        _version: version,
        endian_indicator,
    };

    // Check endianness indicator
    if header.endian_indicator != 0x4D49 && header.endian_indicator != 0x494D {
        return Err(IoError::FormatError(
            "Invalid endianness indicator".to_string(),
        ));
    }

    // Read data elements
    let mut variables = HashMap::<String, MatType>::new();

    // Read data elements until EOF
    while let Ok(element_type) = read_i32(&mut reader) {
        // Check if we've reached EOF
        if element_type == 0 {
            break;
        }

        // Read element size
        let element_size = read_i32(&mut reader)?;

        // Handle different data types
        match element_type {
            MI_MATRIX => {
                // Read matrix data
                let mut matrix_data = vec![0u8; element_size as usize];
                reader.read_exact(&mut matrix_data).map_err(|e| {
                    IoError::FileError(format!("Failed to read matrix data: {}", e))
                })?;

                // Parse matrix data
                if let Ok((name, mat_type)) = parse_matrix_data(&matrix_data) {
                    variables.insert(name, mat_type);
                }
            }
            _ => {
                // Skip unknown element types
                reader
                    .by_ref()
                    .take(element_size as u64)
                    .read_to_end(&mut vec![])
                    .map_err(|e| IoError::FileError(format!("Failed to skip element: {}", e)))?;
            }
        }
    }

    Ok(variables)
}

/// Parse matrix data from byte array
fn parse_matrix_data(data: &[u8]) -> Result<(String, MatType)> {
    let mut cursor = 0;

    // Read array flags
    let array_flags_type = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    let array_flags_size = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    if array_flags_type != MI_UINT32 || array_flags_size != 8 {
        return Err(IoError::FormatError("Invalid array flags".to_string()));
    }

    let flags = MatrixFlags::from_u32(LittleEndian::read_u32(&data[cursor..cursor + 4]));
    cursor += 8; // Skip flags (4 bytes) and reserved (4 bytes)

    // Read dimensions
    let dimensions_type = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    let dimensions_size = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    if dimensions_type != MI_INT32 {
        return Err(IoError::FormatError("Invalid dimensions type".to_string()));
    }

    let num_dims = dimensions_size / 4;
    let mut dims = Vec::with_capacity(num_dims as usize);
    for i in 0..num_dims {
        dims.push(LittleEndian::read_i32(
            &data[cursor + (i * 4) as usize..cursor + ((i + 1) * 4) as usize],
        ));
    }
    cursor += dimensions_size as usize;

    // Pad to 8-byte boundary
    if cursor % 8 != 0 {
        cursor += 8 - (cursor % 8);
    }

    // Read array name
    let name_type = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    let name_size = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    if name_type != MI_INT8 && name_type != MI_UTF8 {
        return Err(IoError::FormatError("Invalid name type".to_string()));
    }

    let name = std::str::from_utf8(&data[cursor..cursor + name_size as usize])
        .map_err(|_| IoError::FormatError("Invalid name encoding".to_string()))?
        .to_string();

    cursor += name_size as usize;

    // Pad to 8-byte boundary
    if cursor % 8 != 0 {
        cursor += 8 - (cursor % 8);
    }

    // Read data
    let data_type = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    let data_size = LittleEndian::read_i32(&data[cursor..cursor + 4]);
    cursor += 4;

    let real_data = &data[cursor..cursor + data_size as usize];
    cursor += data_size as usize;

    // Pad to 8-byte boundary
    if cursor % 8 != 0 {
        cursor += 8 - (cursor % 8);
    }

    // Read imaginary part if complex
    let _imag_data = if flags.is_complex {
        let imag_type = LittleEndian::read_i32(&data[cursor..cursor + 4]);
        cursor += 4;

        let imag_size = LittleEndian::read_i32(&data[cursor..cursor + 4]);
        cursor += 4;

        if imag_type != data_type {
            return Err(IoError::FormatError(
                "Mismatched imaginary type".to_string(),
            ));
        }

        Some(&data[cursor..cursor + imag_size as usize])
    } else {
        None
    };

    // Convert to appropriate MatType based on class type
    let mat_type = match flags.class_type {
        MX_DOUBLE_CLASS => {
            let data_vec = bytes_to_f64_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Double(ndarray)
        }
        MX_SINGLE_CLASS => {
            let data_vec = bytes_to_f32_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Single(ndarray)
        }
        MX_INT8_CLASS => {
            let data_vec = real_data.to_vec();
            let ndarray = Array::from_shape_vec(
                IxDyn(&convert_dims(&dims)),
                data_vec.into_iter().map(|b| b as i8).collect(),
            )
            .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Int8(ndarray)
        }
        MX_UINT8_CLASS => {
            let data_vec = real_data.to_vec();
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::UInt8(ndarray)
        }
        MX_INT16_CLASS => {
            let data_vec = bytes_to_i16_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Int16(ndarray)
        }
        MX_UINT16_CLASS => {
            let data_vec = bytes_to_u16_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::UInt16(ndarray)
        }
        MX_INT32_CLASS => {
            let data_vec = bytes_to_i32_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Int32(ndarray)
        }
        MX_UINT32_CLASS => {
            let data_vec = bytes_to_u32_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::UInt32(ndarray)
        }
        MX_INT64_CLASS => {
            let data_vec = bytes_to_i64_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::Int64(ndarray)
        }
        MX_UINT64_CLASS => {
            let data_vec = bytes_to_u64_vec(real_data);
            let ndarray = Array::from_shape_vec(IxDyn(&convert_dims(&dims)), data_vec)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            MatType::UInt64(ndarray)
        }
        MX_CHAR_CLASS => {
            // Convert to string
            let chars: Vec<u16> = bytes_to_u16_vec(real_data);
            let utf16_chars: Vec<u16> = chars.into_iter().collect();
            let string = String::from_utf16_lossy(&utf16_chars);
            MatType::Char(string)
        }
        _ => {
            // Unsupported class type
            return Err(IoError::FormatError(format!(
                "Unsupported class type: {}",
                flags.class_type
            )));
        }
    };

    Ok((name, mat_type))
}

/// Convert MATLAB dimensions to ndarray dimensions
fn convert_dims(dims: &[i32]) -> Vec<usize> {
    // MATLAB stores dimensions in column-major order
    // For compatibility with ndarray (row-major), we reverse the dimensions
    dims.iter().rev().map(|&d| d as usize).collect()
}

/// Read an i32 from the reader
fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buffer = [0u8; 4];
    match reader.read_exact(&mut buffer) {
        Ok(_) => Ok(LittleEndian::read_i32(&buffer)),
        Err(_) => Ok(0), // EOF
    }
}

/// Convert bytes to f64 vector
fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    let mut result = Vec::with_capacity(bytes.len() / 8);
    for i in (0..bytes.len()).step_by(8) {
        if i + 8 <= bytes.len() {
            result.push(LittleEndian::read_f64(&bytes[i..i + 8]));
        }
    }
    result
}

/// Convert bytes to f32 vector
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for i in (0..bytes.len()).step_by(4) {
        if i + 4 <= bytes.len() {
            result.push(LittleEndian::read_f32(&bytes[i..i + 4]));
        }
    }
    result
}

/// Convert bytes to i16 vector
fn bytes_to_i16_vec(bytes: &[u8]) -> Vec<i16> {
    let mut result = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        if i + 2 <= bytes.len() {
            result.push(LittleEndian::read_i16(&bytes[i..i + 2]));
        }
    }
    result
}

/// Convert bytes to u16 vector
fn bytes_to_u16_vec(bytes: &[u8]) -> Vec<u16> {
    let mut result = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        if i + 2 <= bytes.len() {
            result.push(LittleEndian::read_u16(&bytes[i..i + 2]));
        }
    }
    result
}

/// Convert bytes to i32 vector
fn bytes_to_i32_vec(bytes: &[u8]) -> Vec<i32> {
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for i in (0..bytes.len()).step_by(4) {
        if i + 4 <= bytes.len() {
            result.push(LittleEndian::read_i32(&bytes[i..i + 4]));
        }
    }
    result
}

/// Convert bytes to u32 vector
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for i in (0..bytes.len()).step_by(4) {
        if i + 4 <= bytes.len() {
            result.push(LittleEndian::read_u32(&bytes[i..i + 4]));
        }
    }
    result
}

/// Convert bytes to i64 vector
fn bytes_to_i64_vec(bytes: &[u8]) -> Vec<i64> {
    let mut result = Vec::with_capacity(bytes.len() / 8);
    for i in (0..bytes.len()).step_by(8) {
        if i + 8 <= bytes.len() {
            result.push(LittleEndian::read_i64(&bytes[i..i + 8]));
        }
    }
    result
}

/// Convert bytes to u64 vector
fn bytes_to_u64_vec(bytes: &[u8]) -> Vec<u64> {
    let mut result = Vec::with_capacity(bytes.len() / 8);
    for i in (0..bytes.len()).step_by(8) {
        if i + 8 <= bytes.len() {
            result.push(LittleEndian::read_u64(&bytes[i..i + 8]));
        }
    }
    result
}

/// Writes data to a MATLAB .mat file
///
/// # Arguments
///
/// * `path` - Path where the .mat file should be written
/// * `vars` - A HashMap mapping variable names to their values
///
/// # Example
///
/// ```no_run
/// use scirs2_io::matlab::{write_mat, MatType};
/// use ndarray::Array;
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// let mut vars = HashMap::new();
/// let data = Array::linspace(0.0, 10.0, 100).into_dyn();
/// vars.insert("x".to_string(), MatType::Double(data));
///
/// write_mat(Path::new("output.mat"), &vars).unwrap();
/// ```
pub fn write_mat<P: AsRef<Path>>(path: P, _vars: &HashMap<String, MatType>) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write MAT file header

    // Write "MATLAB" magic string
    writer
        .write_all(b"MATLAB ")
        .map_err(|e| IoError::FileError(format!("Failed to write MAT header: {}", e)))?;

    // Write version platform info (padding to 116 bytes total)
    let version = format!(
        "7.0 MAT-file, Platform: SCIRS2, Created by: scirs2-io {}",
        env!("CARGO_PKG_VERSION")
    );
    let version_bytes = version.as_bytes();
    let version_len = version_bytes.len().min(110); // Ensure it fits in the header

    writer
        .write_all(&version_bytes[0..version_len])
        .map_err(|e| IoError::FileError(format!("Failed to write version info: {}", e)))?;

    // Pad to 116 bytes
    let padding = vec![0u8; 116 - 6 - version_len];
    writer
        .write_all(&padding)
        .map_err(|e| IoError::FileError(format!("Failed to write header padding: {}", e)))?;

    // Write version (0x0100 for MAT 5) and endianness indicator (0x4D49 for little endian)
    writer
        .write_u16::<LittleEndian>(0x0100)
        .map_err(|e| IoError::FileError(format!("Failed to write version: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(0x4D49)
        .map_err(|e| IoError::FileError(format!("Failed to write endianness: {}", e)))?;

    // TODO: Implement full writing of variables
    // For now, return a placeholder message
    Err(IoError::Other(
        "MATLAB file writing not fully implemented yet".to_string(),
    ))
}
