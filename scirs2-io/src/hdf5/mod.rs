//! HDF5 file format support
//!
//! This module provides functionality for reading and writing HDF5 (Hierarchical Data Format version 5) files.
//! HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited
//! variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data.
//!
//! Features:
//! - Reading and writing HDF5 files
//! - Support for groups and datasets
//! - Attributes on groups and datasets
//! - Multiple datatypes (integers, floats, strings, compound types)
//! - Chunking and compression support
//! - Integration with ndarray for efficient array operations
//! - Enhanced functionality with compression and parallel I/O (see `enhanced` module)
//! - Extended data type support including complex numbers and boolean types
//! - Performance optimizations for large datasets

use crate::error::{IoError, Result};
use ndarray::{ArrayBase, ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::File;

/// HDF5 data type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HDF5DataType {
    /// Integer types
    Integer {
        /// Size in bytes (1, 2, 4, or 8)
        size: usize,
        /// Whether the integer is signed
        signed: bool,
    },
    /// Floating point types
    Float {
        /// Size in bytes (4 or 8)
        size: usize,
    },
    /// String type
    String {
        /// String encoding (UTF-8 or ASCII)
        encoding: StringEncoding,
    },
    /// Array type
    Array {
        /// Base data type of array elements
        base_type: Box<HDF5DataType>,
        /// Shape of the array
        shape: Vec<usize>,
    },
    /// Compound type
    Compound {
        /// Fields in the compound type (name, type) pairs
        fields: Vec<(String, HDF5DataType)>,
    },
    /// Enum type
    Enum {
        /// Enumeration values (name, value) pairs
        values: Vec<(String, i64)>,
    },
}

/// String encoding types
#[derive(Debug, Clone, PartialEq)]
pub enum StringEncoding {
    /// UTF-8 encoding
    UTF8,
    /// ASCII encoding
    ASCII,
}

/// HDF5 compression options
#[derive(Debug, Clone, Default)]
pub struct CompressionOptions {
    /// Enable gzip compression
    pub gzip: Option<u8>,
    /// Enable szip compression  
    pub szip: Option<(u32, u32)>,
    /// Enable LZF compression
    pub lzf: bool,
    /// Enable shuffle filter
    pub shuffle: bool,
}

/// HDF5 dataset creation options
#[derive(Debug, Clone, Default)]
pub struct DatasetOptions {
    /// Chunk size for chunked storage
    pub chunk_size: Option<Vec<usize>>,
    /// Compression options
    pub compression: CompressionOptions,
    /// Fill value for uninitialized elements
    pub fill_value: Option<f64>,
    /// Enable fletcher32 checksum
    pub fletcher32: bool,
}

/// HDF5 file handle
pub struct HDF5File {
    /// File path
    #[allow(dead_code)]
    path: String,
    /// Root group
    root: Group,
    /// File access mode
    #[allow(dead_code)]
    mode: FileMode,
    /// Native HDF5 file handle (when feature is enabled)
    #[cfg(feature = "hdf5")]
    native_file: Option<File>,
}

/// File access mode
#[derive(Debug, Clone, PartialEq)]
pub enum FileMode {
    /// Read-only mode
    ReadOnly,
    /// Read-write mode
    ReadWrite,
    /// Create new file (fail if exists)
    Create,
    /// Create or truncate existing file
    Truncate,
}

/// HDF5 group
#[derive(Debug, Clone)]
pub struct Group {
    /// Group name
    pub name: String,
    /// Child groups
    pub groups: HashMap<String, Group>,
    /// Datasets in this group
    pub datasets: HashMap<String, Dataset>,
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl Group {
    /// Create a new empty group
    pub fn new(name: String) -> Self {
        Self {
            name,
            groups: HashMap::new(),
            datasets: HashMap::new(),
            attributes: HashMap::new(),
        }
    }

    /// Create a subgroup
    pub fn create_group(&mut self, name: &str) -> &mut Group {
        self.groups
            .entry(name.to_string())
            .or_insert_with(|| Group::new(name.to_string()))
    }

    /// Get a subgroup
    pub fn get_group(&self, name: &str) -> Option<&Group> {
        self.groups.get(name)
    }

    /// Get a mutable subgroup
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut Group> {
        self.groups.get_mut(name)
    }

    /// Add an attribute
    pub fn set_attribute(&mut self, name: &str, value: AttributeValue) {
        self.attributes.insert(name.to_string(), value);
    }

    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    /// Remove an attribute
    pub fn remove_attribute(&mut self, name: &str) -> Option<AttributeValue> {
        self.attributes.remove(name)
    }

    /// List all attribute names
    pub fn attribute_names(&self) -> Vec<&str> {
        self.attributes.keys().map(|s| s.as_str()).collect()
    }

    /// Check if group has a specific attribute
    pub fn has_attribute(&self, name: &str) -> bool {
        self.attributes.contains_key(name)
    }

    /// Get dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    /// Get mutable dataset by name
    pub fn get_dataset_mut(&mut self, name: &str) -> Option<&mut Dataset> {
        self.datasets.get_mut(name)
    }

    /// List all dataset names
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }

    /// List all group names
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }

    /// Check if group has a specific dataset
    pub fn has_dataset(&self, name: &str) -> bool {
        self.datasets.contains_key(name)
    }

    /// Check if group has a specific subgroup
    pub fn has_group(&self, name: &str) -> bool {
        self.groups.contains_key(name)
    }

    /// Remove a dataset
    pub fn remove_dataset(&mut self, name: &str) -> Option<Dataset> {
        self.datasets.remove(name)
    }

    /// Remove a subgroup
    pub fn remove_group(&mut self, name: &str) -> Option<Group> {
        self.groups.remove(name)
    }
}

/// HDF5 dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Dataset name
    pub name: String,
    /// Data type
    pub dtype: HDF5DataType,
    /// Shape
    pub shape: Vec<usize>,
    /// Data (stored as flattened array)
    pub data: DataArray,
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Dataset options
    pub options: DatasetOptions,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(
        name: String,
        dtype: HDF5DataType,
        shape: Vec<usize>,
        data: DataArray,
        options: DatasetOptions,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            data,
            attributes: HashMap::new(),
            options,
        }
    }

    /// Set an attribute on the dataset
    pub fn set_attribute(&mut self, name: &str, value: AttributeValue) {
        self.attributes.insert(name.to_string(), value);
    }

    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    /// Remove an attribute
    pub fn remove_attribute(&mut self, name: &str) -> Option<AttributeValue> {
        self.attributes.remove(name)
    }

    /// Get the number of elements in the dataset
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total size in bytes (estimate)
    pub fn size_bytes(&self) -> usize {
        let element_size = match &self.dtype {
            HDF5DataType::Integer { size, .. } => *size,
            HDF5DataType::Float { size } => *size,
            HDF5DataType::String { .. } => 8,   // Estimate
            HDF5DataType::Array { .. } => 8,    // Estimate
            HDF5DataType::Compound { .. } => 8, // Estimate
            HDF5DataType::Enum { .. } => 8,     // Estimate
        };
        self.len() * element_size
    }

    /// Get data as float vector (if possible)
    pub fn as_float_vec(&self) -> Option<Vec<f64>> {
        match &self.data {
            DataArray::Float(data) => Some(data.clone()),
            DataArray::Integer(data) => Some(data.iter().map(|&x| x as f64).collect()),
            _ => None,
        }
    }

    /// Get data as integer vector (if possible)
    pub fn as_integer_vec(&self) -> Option<Vec<i64>> {
        match &self.data {
            DataArray::Integer(data) => Some(data.clone()),
            DataArray::Float(data) => Some(data.iter().map(|&x| x as i64).collect()),
            _ => None,
        }
    }

    /// Get data as string vector (if possible)
    pub fn as_string_vec(&self) -> Option<Vec<String>> {
        match &self.data {
            DataArray::String(data) => Some(data.clone()),
            _ => None,
        }
    }
}

/// Data array storage
#[derive(Debug, Clone)]
pub enum DataArray {
    /// Integer data
    Integer(Vec<i64>),
    /// Float data
    Float(Vec<f64>),
    /// String data
    String(Vec<String>),
    /// Binary data
    Binary(Vec<u8>),
}

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Integer array
    IntegerArray(Vec<i64>),
    /// Float array
    FloatArray(Vec<f64>),
    /// String array
    StringArray(Vec<String>),
}

/// File statistics
#[derive(Debug, Clone, Default)]
pub struct FileStats {
    /// Number of groups in the file
    pub num_groups: usize,
    /// Number of datasets in the file
    pub num_datasets: usize,
    /// Number of attributes in the file
    pub num_attributes: usize,
    /// Total data size in bytes
    pub total_data_size: usize,
}

impl HDF5File {
    /// Create a new HDF5 file
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        #[cfg(feature = "hdf5")]
        {
            let native_file = File::create(&path_str)
                .map_err(|e| IoError::FormatError(format!("Failed to create HDF5 file: {e}")))?;

            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode: FileMode::Create,
                native_file: Some(native_file),
            })
        }

        #[cfg(not(feature = "hdf5"))]
        {
            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode: FileMode::Create,
            })
        }
    }

    /// Open an existing HDF5 file
    pub fn open<P: AsRef<Path>>(path: P, mode: FileMode) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        #[cfg(feature = "hdf5")]
        {
            let native_file = match mode {
                FileMode::ReadOnly => File::open(&path_str)
                    .map_err(|e| IoError::FormatError(format!("Failed to open HDF5 file: {e}")))?,
                FileMode::ReadWrite => File::open_rw(&path_str)
                    .map_err(|e| IoError::FormatError(format!("Failed to open HDF5 file: {e}")))?,
                FileMode::Create => File::create(&path_str).map_err(|e| {
                    IoError::FormatError(format!("Failed to create HDF5 file: {e}"))
                })?,
                FileMode::Truncate => File::create(&path_str).map_err(|e| {
                    IoError::FormatError(format!("Failed to create HDF5 file: {e}"))
                })?,
            };

            // Load existing structure from the file
            let mut root = Group::new("/".to_string());
            Self::load_group_structure(&native_file, &mut root)?;

            Ok(Self {
                path: path_str,
                root,
                mode,
                native_file: Some(native_file),
            })
        }

        #[cfg(not(feature = "hdf5"))]
        {
            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode,
            })
        }
    }

    /// Get the root group
    pub fn root(&self) -> &Group {
        &self.root
    }

    /// Get the root group mutably
    pub fn root_mut(&mut self) -> &mut Group {
        &mut self.root
    }

    /// Get access to the native HDF5 file handle (when feature is enabled)
    #[cfg(feature = "hdf5")]
    pub fn native_file(&self) -> Option<&File> {
        self.native_file.as_ref()
    }

    /// Load group structure from native HDF5 file
    #[cfg(feature = "hdf5")]
    fn load_group_structure(file: &File, group: &mut Group) -> Result<()> {
        use hdf5::types::TypeDescriptor;

        // Load all datasets in the current group
        let dataset_names = _file
            .dataset_names()
            .map_err(|e| IoError::FormatError(format!("Failed to get dataset names: {e}")))?;

        for dataset_name in dataset_names {
            if let Ok(h5_dataset) = file.dataset(&dataset_name) {
                // Get dataset metadata
                let shape: Vec<usize> = h5_dataset.shape().to_vec();
                let dtype = h5_dataset.dtype().map_err(|e| {
                    IoError::FormatError(format!("Failed to get dataset dtype: {e}"))
                })?;

                // Convert HDF5 datatype to our internal representation
                let internal_dtype = Self::convert_hdf5_datatype(&dtype)?;

                // Read dataset data based on type
                let data = Self::read_dataset_data(&h5_dataset, &dtype)?;

                // Read attributes
                let mut attributes = HashMap::new();
                if let Ok(attr_names) = h5_dataset.attr_names() {
                    for attr_name in attr_names {
                        if let Ok(attr) = h5_dataset.attr(&attr_name) {
                            if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                                attributes.insert(attr_name, attr_value);
                            }
                        }
                    }
                }

                // Create dataset
                let dataset = Dataset {
                    name: dataset_name.clone(),
                    dtype: internal_dtype,
                    shape,
                    data,
                    attributes,
                    options: DatasetOptions::default(),
                };

                group.datasets.insert(dataset_name, dataset);
            }
        }

        // Load all subgroups recursively
        let group_names = _file
            .group_names()
            .map_err(|e| IoError::FormatError(format!("Failed to get group names: {e}")))?;

        for group_name in group_names {
            if let Ok(h5_subgroup) = file.group(&group_name) {
                let mut subgroup = Group::new(group_name.clone());

                // Read group attributes
                if let Ok(attr_names) = h5_subgroup.attr_names() {
                    for attr_name in attr_names {
                        if let Ok(attr) = h5_subgroup.attr(&attr_name) {
                            if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                                subgroup.attributes.insert(attr_name, attr_value);
                            }
                        }
                    }
                }

                // Recursively load subgroup structure
                Self::load_group_structure(&h5_subgroup, &mut subgroup)?;

                group.groups.insert(group_name, subgroup);
            }
        }

        Ok(())
    }

    /// Convert HDF5 datatype to internal representation
    #[cfg(feature = "hdf5")]
    fn convert_hdf5_datatype(dtype: &hdf5::Datatype) -> Result<HDF5DataType> {
        use hdf5::types::TypeDescriptor;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(int_type)) => Ok(HDF5DataType::Integer {
                size: int_type.size(),
                signed: int_type.is_signed(),
            }),
            Ok(TypeDescriptor::Float(float_type)) => Ok(HDF5DataType::Float {
                size: float_type.size(),
            }),
            Ok(TypeDescriptor::FixedUnicode(size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::FixedAscii(size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            Ok(TypeDescriptor::VarLenUnicode) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::VarLenAscii) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            Ok(TypeDescriptor::Array(array_type)) => {
                let base_type = Self::convert_hdf5_datatype(&array_type.base_type())?;
                Ok(HDF5DataType::Array {
                    base_type: Box::new(base_type),
                    shape: array_type.shape().to_vec(),
                })
            }
            Ok(TypeDescriptor::Compound(comp_type)) => {
                let mut fields = Vec::new();
                for field in comp_type.fields() {
                    let field_type = Self::convert_hdf5_datatype(&field.ty())?;
                    fields.push((field.name().to_string(), field_type));
                }
                Ok(HDF5DataType::Compound { fields })
            }
            Ok(TypeDescriptor::Enum(enum_type)) => {
                let mut values = Vec::new();
                for member in enum_type.members() {
                    values.push((member.name().to_string(), member.value()));
                }
                Ok(HDF5DataType::Enum { values })
            }
            _ => {
                // Fallback for unsupported types
                Ok(HDF5DataType::String {
                    encoding: StringEncoding::UTF8,
                })
            }
        }
    }

    /// Read dataset data based on HDF5 datatype
    #[cfg(feature = "hdf5")]
    fn read_dataset_data(dataset: &hdf5::Dataset, dtype: &hdf5::Datatype) -> Result<DataArray> {
        use hdf5::types::TypeDescriptor;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(_)) => {
                let data: Vec<i64> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read integer dataset: {e}"))
                })?;
                Ok(DataArray::Integer(data))
            }
            Ok(TypeDescriptor::Float(_)) => {
                let data: Vec<f64> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read float dataset: {e}"))
                })?;
                Ok(DataArray::Float(data))
            }
            Ok(TypeDescriptor::FixedUnicode(_))
            | Ok(TypeDescriptor::FixedAscii(_))
            | Ok(TypeDescriptor::VarLenUnicode)
            | Ok(TypeDescriptor::VarLenAscii) => {
                let data: Vec<String> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read string dataset: {e}"))
                })?;
                Ok(DataArray::String(data))
            }
            _ => {
                // For unsupported types, read as binary data
                let data: Vec<u8> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read binary dataset: {e}"))
                })?;
                Ok(DataArray::Binary(data))
            }
        }
    }

    /// Read attribute value
    #[cfg(feature = "hdf5")]
    fn read_attribute_value(attr: &hdf5::Attribute) -> Result<AttributeValue> {
        use hdf5::types::TypeDescriptor;

        let dtype = _attr
            .dtype()
            .map_err(|e| IoError::FormatError(format!("Failed to get attribute dtype: {e}")))?;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(_)) => {
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: i64 = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read integer attribute: {e}"))
                    })?;
                    Ok(AttributeValue::Integer(value))
                } else {
                    let value: Vec<i64> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read integer array attribute: {}",
                            e
                        ))
                    })?;
                    Ok(AttributeValue::IntegerArray(value))
                }
            }
            Ok(TypeDescriptor::Float(_)) => {
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: f64 = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read float attribute: {e}"))
                    })?;
                    Ok(AttributeValue::Float(value))
                } else {
                    let value: Vec<f64> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!("Failed to read float array attribute: {e}"))
                    })?;
                    Ok(AttributeValue::FloatArray(value))
                }
            }
            Ok(TypeDescriptor::FixedUnicode(_))
            | Ok(TypeDescriptor::FixedAscii(_))
            | Ok(TypeDescriptor::VarLenUnicode)
            | Ok(TypeDescriptor::VarLenAscii) => {
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: String = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read string attribute: {e}"))
                    })?;
                    Ok(AttributeValue::String(value))
                } else {
                    let value: Vec<String> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read string array attribute: {}",
                            e
                        ))
                    })?;
                    Ok(AttributeValue::StringArray(value))
                }
            }
            _ => {
                // Fallback: treat as string
                let value: String = _attr
                    .read_scalar()
                    .unwrap_or_else(|_| "unknown".to_string());
                Ok(AttributeValue::String(value))
            }
        }
    }

    /// Create a dataset from an ndarray
    pub fn create_dataset_from_array<A, D>(
        &mut self,
        path: &str,
        array: &ArrayBase<A, D>,
        options: Option<DatasetOptions>,
    ) -> Result<()>
    where
        A: ndarray::Data,
        A::Elem: Clone + Into<f64>,
        D: ndarray::Dimension,
    {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &mut self.root;

        // Navigate to the parent group, creating groups as needed
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group.create_group(group_name);
        }

        // Convert array to dataset
        let shape: Vec<usize> = array.shape().to_vec();
        let flat_data: Vec<f64> = array.iter().map(|x| x.clone().into()).collect();

        let dataset = Dataset {
            name: dataset_name.to_string(),
            dtype: HDF5DataType::Float { size: 8 },
            shape: shape.clone(),
            data: DataArray::Float(flat_data.clone()),
            attributes: HashMap::new(),
            options: options.unwrap_or_default(),
        };

        current_group
            .datasets
            .insert(dataset_name.to_string(), dataset);

        // Also write to the native HDF5 file if available
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                // Handle nested groups properly by creating the full path
                let mut current_group = file.clone();

                // Create nested groups as needed
                for &group_name in &parts[..parts.len() - 1] {
                    current_group = if let Ok(existing_group) = current_group.group(group_name) {
                        existing_group
                    } else {
                        current_group.create_group(group_name).map_err(|e| {
                            IoError::FormatError(format!(
                                "Failed to create HDF5 group '{}': {}",
                                group_name, e
                            ))
                        })?
                    };
                }

                // Create dataset with proper multidimensional shape
                let h5_dataset = if shape.len() == 1 {
                    current_group
                        .new_dataset::<f64>()
                        .shape(shape[0])
                        .create(*dataset_name)
                } else {
                    current_group
                        .new_dataset::<f64>()
                        .shape(&shape[..])
                        .create(*dataset_name)
                }
                .map_err(|e| {
                    IoError::FormatError(format!(
                        "Failed to create HDF5 dataset '{}': {}",
                        dataset_name, e
                    ))
                })?;

                // Write data with proper reshaping
                if shape.len() == 1 {
                    h5_dataset.write(&flat_data).map_err(|e| {
                        IoError::FormatError(format!("Failed to write HDF5 dataset: {e}"))
                    })?;
                } else {
                    // For multidimensional arrays, we need to reshape the data
                    let array_data =
                        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), flat_data).map_err(
                            |e| IoError::FormatError(format!("Failed to reshape data: {e}")),
                        )?;

                    h5_dataset.write(&array_data).map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to write HDF5 multidimensional dataset: {}",
                            e
                        ))
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Read a dataset as an ndarray
    pub fn read_dataset(&self, path: &str) -> Result<ArrayD<f64>> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &self.root;

        // Navigate to the parent group
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        // Get the dataset
        let dataset = current_group
            .datasets
            .get(*dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))?;

        // Try to read from native HDF5 file first if available
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                // Handle nested groups properly by navigating the full path
                let mut current_group = file.clone();

                // Navigate to the parent group
                for &group_name in &parts[..parts.len() - 1] {
                    current_group = current_group.group(group_name).map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to open HDF5 group '{}': {}",
                            group_name, e
                        ))
                    })?;
                }

                // Read the dataset from the correct group
                if let Ok(h5_dataset) = current_group.dataset(dataset_name) {
                    let data: Vec<f64> = h5_dataset.read_raw().map_err(|e| {
                        IoError::FormatError(format!("Failed to read HDF5 dataset: {e}"))
                    })?;

                    let shape = IxDyn(&dataset.shape);
                    return ArrayD::from_shape_vec(shape, data)
                        .map_err(|e| IoError::FormatError(e.to_string()));
                }
            }
        }

        // Fall back to in-memory data
        match &dataset.data {
            DataArray::Float(data) => {
                let shape = IxDyn(&dataset.shape);
                ArrayD::from_shape_vec(shape, data.clone())
                    .map_err(|e| IoError::FormatError(e.to_string()))
            }
            DataArray::Integer(data) => {
                let float_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let shape = IxDyn(&dataset.shape);
                ArrayD::from_shape_vec(shape, float_data)
                    .map_err(|e| IoError::FormatError(e.to_string()))
            }
            _ => Err(IoError::FormatError(
                "Unsupported data type for ndarray conversion".to_string(),
            )),
        }
    }

    /// Write the file to disk
    pub fn write(&self) -> Result<()> {
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                // For HDF5, writing happens automatically when datasets are created
                // So we just need to flush any pending operations
                _file
                    .flush()
                    .map_err(|e| IoError::FormatError(format!("Failed to flush HDF5 file: {e}")))?;
            }
        }

        #[cfg(not(feature = "hdf5"))]
        {
            // Placeholder implementation
            println!("Writing HDF5 file to: {} (placeholder)", self.path);
        }

        Ok(())
    }

    /// Get a dataset by path (e.g., "/group1/group2/dataset")
    pub fn get_dataset(&self, path: &str) -> Result<&Dataset> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &self.root;

        // Navigate to the parent group
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        // Get the dataset
        current_group
            .get_dataset(dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))
    }

    /// Get a group by path (e.g., "/group1/group2")
    pub fn get_group(&self, path: &str) -> Result<&Group> {
        if path == "/" || path.is_empty() {
            return Ok(&self.root);
        }

        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut current_group = &self.root;

        for &group_name in &parts {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        Ok(current_group)
    }

    /// List all datasets in the file recursively
    pub fn list_datasets(&self) -> Vec<String> {
        let mut datasets = Vec::new();
        self.collect_datasets(&self.root, String::new(), &mut datasets);
        datasets
    }

    /// List all groups in the file recursively
    pub fn list_groups(&self) -> Vec<String> {
        let mut groups = Vec::new();
        self.collect_groups(&self.root, String::new(), &mut groups);
        groups
    }

    /// Helper method to recursively collect dataset paths
    #[allow(clippy::only_used_in_recursion)]
    fn collect_datasets(&self, group: &Group, prefix: String, datasets: &mut Vec<String>) {
        for dataset_name in group.dataset_names() {
            let fullpath = if prefix.is_empty() {
                dataset_name.to_string()
            } else {
                format!("{prefix}/{dataset_name}")
            };
            datasets.push(fullpath);
        }

        for (group_name, subgroup) in &group.groups {
            let new_prefix = if prefix.is_empty() {
                group_name.clone()
            } else {
                format!("{prefix}/{group_name}")
            };
            self.collect_datasets(subgroup, new_prefix, datasets);
        }
    }

    /// Helper method to recursively collect group paths
    #[allow(clippy::only_used_in_recursion)]
    fn collect_groups(&self, group: &Group, prefix: String, groups: &mut Vec<String>) {
        for (group_name, subgroup) in &group.groups {
            let fullpath = if prefix.is_empty() {
                group_name.clone()
            } else {
                format!("{prefix}/{group_name}")
            };
            groups.push(fullpath.clone());
            self.collect_groups(subgroup, fullpath, groups);
        }
    }

    /// Get file statistics
    pub fn stats(&self) -> FileStats {
        let mut stats = FileStats::default();
        self.collect_stats(&self.root, &mut stats);
        stats
    }

    /// Helper method to collect file statistics
    #[allow(clippy::only_used_in_recursion)]
    fn collect_stats(&self, group: &Group, stats: &mut FileStats) {
        stats.num_groups += group.groups.len();
        stats.num_datasets += group.datasets.len();
        stats.num_attributes += group.attributes.len();

        for dataset in group.datasets.values() {
            stats.num_attributes += dataset.attributes.len();
            stats.total_data_size += dataset.size_bytes();
        }

        for subgroup in group.groups.values() {
            self.collect_stats(subgroup, stats);
        }
    }

    /// Close the file
    pub fn close(self) -> Result<()> {
        #[cfg(feature = "hdf5")]
        {
            if let Some(file) = self.native_file {
                // File is automatically closed when dropped
                drop(file);
            }
        }

        Ok(())
    }
}

/// Read an HDF5 file and return the root group
///
/// # Arguments
/// * `path` - Path to the HDF5 file
///
/// # Returns
/// The root group of the HDF5 file
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::read_hdf5;
///
/// let root_group = read_hdf5("data.h5")?;
/// println!("Groups: {:?}", root_group.groups.keys().collect::<Vec<_>>());
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn read_hdf5<P: AsRef<Path>>(path: P) -> Result<Group> {
    let file = HDF5File::open(path, FileMode::ReadOnly)?;
    Ok(file.root)
}

/// Write data to an HDF5 file
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `datasets` - Map of dataset paths to arrays
///
/// # Example
/// ```no_run
/// use ndarray::array;
/// use std::collections::HashMap;
/// use scirs2_io::hdf5::write_hdf5;
///
/// let mut datasets = HashMap::new();
/// datasets.insert("data/temperature".to_string(), array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
/// datasets.insert("data/pressure".to_string(), array![100.0, 200.0, 300.0].into_dyn());
///
/// write_hdf5("output.h5", datasets)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn write_hdf5<P: AsRef<Path>>(path: P, datasets: HashMap<String, ArrayD<f64>>) -> Result<()> {
    let mut file = HDF5File::create(path)?;

    for (datasetpath, array) in datasets {
        file.create_dataset_from_array(&datasetpath, &array, None)?;
    }

    file.write()?;
    file.close()?;
    Ok(())
}

/// Create an HDF5 file with groups and attributes
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `builder` - Function to build the file structure
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::{create_hdf5_with_structure, AttributeValue};
/// use ndarray::array;
///
/// create_hdf5_with_structure("structured.h5", |file| {
///     let root = file.root_mut();
///     
///     // Create groups
///     let experiment = root.create_group("experiment");
///     experiment.set_attribute("date", AttributeValue::String("2024-01-01".to_string()));
///     experiment.set_attribute("temperature", AttributeValue::Float(25.0));
///     
///     // Add datasets
///     let data = array![[1.0, 2.0], [3.0, 4.0]];
///     file.create_dataset_from_array("experiment/measurements", &data, None)?;
///     
///     Ok(())
/// })?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn create_hdf5_with_structure<P, F>(path: P, builder: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut HDF5File) -> Result<()>,
{
    let mut file = HDF5File::create(path)?;
    builder(&mut file)?;
    file.write()?;
    file.close()?;
    Ok(())
}

/// Enhanced HDF5 functionality with compression and parallel I/O
pub mod enhanced;

// Re-export enhanced functionality for convenience
pub use enhanced::{
    create_optimal_compression_options, read_hdf5_enhanced, write_hdf5_enhanced, CompressionStats,
    EnhancedHDF5File, ExtendedDataType, ParallelConfig,
};

// Tests module moved to /tmp/

// Legacy inline tests for backward compatibility
#[cfg(test)]
mod legacy_tests {
    use super::*;

    #[test]
    fn test_group_creation() {
        let mut root = Group::new("/".to_string());
        let subgroup = root.create_group("data");
        assert_eq!(subgroup.name, "data");
        assert!(root.get_group("data").is_some());
    }

    #[test]
    fn test_attribute_setting() {
        let mut group = Group::new("test".to_string());
        group.set_attribute("version", AttributeValue::Integer(1));
        group.set_attribute(
            "description",
            AttributeValue::String("Test group".to_string()),
        );

        assert_eq!(group.attributes.len(), 2);
    }

    #[test]
    fn test_dataset_creation() {
        let dataset = Dataset {
            name: "test_data".to_string(),
            dtype: HDF5DataType::Float { size: 8 },
            shape: vec![2, 3],
            data: DataArray::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            attributes: HashMap::new(),
            options: DatasetOptions::default(),
        };

        assert_eq!(dataset.shape, vec![2, 3]);
        if let DataArray::Float(data) = &dataset.data {
            assert_eq!(data.len(), 6);
        }
    }

    #[test]
    fn test_compression_options() {
        let mut options = CompressionOptions::default();
        options.gzip = Some(6);
        options.shuffle = true;

        assert_eq!(options.gzip, Some(6));
        assert!(options.shuffle);
    }

    #[test]
    fn test_hdf5_file_creation() {
        let file = HDF5File::create("test.h5").unwrap();
        assert_eq!(file.mode, FileMode::Create);
        assert_eq!(file.root.name, "/");
    }
}
