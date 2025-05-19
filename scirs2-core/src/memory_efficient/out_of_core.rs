use super::chunked::{ChunkedArray, ChunkingStrategy, OPTIMAL_CHUNK_SIZE};
use super::validation;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use bincode::{deserialize, serialize};
use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, Ix2, IxDyn, ShapeBuilder};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// An array that stores data on disk to reduce memory usage
#[derive(Debug)]
pub struct OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// The shape of the array
    pub shape: Vec<usize>,
    /// The path to the file containing the data
    pub file_path: PathBuf,
    /// The chunking strategy used for reading/writing
    pub strategy: ChunkingStrategy,
    /// The total number of elements
    pub size: usize,
    /// Whether the file is temporary and should be deleted on drop
    is_temp: bool,
    /// Phantom data for type parameters
    _phantom: PhantomData<(A, D)>,
}

impl<A, D> OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new out-of-core array with the given data and file path
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        validation::check_not_empty(data)?;

        let shape = data.shape().to_vec();
        let size = data.len();

        // Create file and write data
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| CoreError::IoError(e))?;

        // Serialize data to file (in chunks if data is large)
        let _chunked = ChunkedArray::new(data.to_owned(), strategy);

        // Note: This is a simplified implementation that writes the entire array at once.
        // A real implementation would write chunks to save memory.
        let serialized = serialize(&data.to_owned()).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("Failed to serialize data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        file.write_all(&serialized)
            .map_err(|e| CoreError::IoError(e))?;

        Ok(Self {
            shape,
            file_path: file_path.to_path_buf(),
            strategy,
            size,
            is_temp: false,
            _phantom: PhantomData,
        })
    }

    /// Create a new out-of-core array with a temporary file
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let temp_file = NamedTempFile::new().map_err(|e| CoreError::IoError(e))?;
        let file_path = temp_file.path().to_path_buf();

        // Manually persist the temp file so it stays around after we return
        let _file = temp_file.persist(&file_path).map_err(|e| {
            CoreError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        let mut result = Self::new(data, &file_path, strategy)?;
        result.is_temp = true;

        Ok(result)
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        let mut file = File::open(&self.file_path).map_err(|e| CoreError::IoError(e))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| CoreError::IoError(e))?;

        let array: Array<A, D> = deserialize(&buffer).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("Failed to deserialize data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(array)
    }

    /// Load a chunk of the array into memory
    pub fn load_chunk(&self, chunk_index: usize) -> Result<Array<A, D>, CoreError> {
        if chunk_index >= self.num_chunks() {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk index out of bounds: {} >= {}",
                    chunk_index,
                    self.num_chunks()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // This is a placeholder - a real implementation would calculate the offset
        // and size of the chunk, then read only that portion of the file
        self.load()
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        let chunk_size = match self.strategy {
            ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),
            ChunkingStrategy::Fixed(size) => size,
            ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),
            ChunkingStrategy::NumChunks(n) => (self.size + n - 1) / n,
        };

        (self.size + chunk_size - 1) / chunk_size
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Apply a function to each chunk of the array
    pub fn map<F, B, R>(&self, _f: F) -> Result<R, CoreError>
    where
        F: FnMut(Array<A, D>) -> B,
        R: FromIterator<B>,
    {
        // This is a placeholder - a real implementation would iterate through chunks
        // and apply the function to each
        Err(CoreError::ImplementationError(
            ErrorContext::new("OutOfCoreArray::map is not yet implemented".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

impl<A, D> Drop for OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    fn drop(&mut self) {
        if self.is_temp {
            // Attempt to remove the temporary file
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

/// A specialized out-of-core array that uses memory mapping for efficient access
#[derive(Debug)]
pub struct DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// The underlying out-of-core array
    pub array: OutOfCoreArray<A, D>,
    /// Whether the array is read-only
    pub read_only: bool,
}

impl<A, D> DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new disk-backed array with the given data and file path
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new(data, file_path, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Create a new disk-backed array with a temporary file
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new_temp(data, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        self.array.load()
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        self.array.is_temp()
    }
}

/// Create a disk-backed array from the given data and file path
pub fn create_disk_array<A, S, D>(
    data: &ArrayBase<S, D>,
    file_path: &Path,
    strategy: ChunkingStrategy,
    read_only: bool,
) -> Result<DiskBackedArray<A, D>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    S: Data<Elem = A>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    DiskBackedArray::new(data, file_path, strategy, read_only)
}

/// Load chunks of an out-of-core array into memory
pub fn load_chunks<A, D>(
    array: &OutOfCoreArray<A, D>,
    chunk_indices: &[usize],
) -> Result<Vec<Array<A, D>>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    let num_chunks = array.num_chunks();

    // Validate chunk indices
    for &idx in chunk_indices {
        if idx >= num_chunks {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk index out of bounds: {} >= {}",
                    idx, num_chunks
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }

    // Load each requested chunk
    let mut result = Vec::with_capacity(chunk_indices.len());
    for &idx in chunk_indices {
        result.push(array.load_chunk(idx)?);
    }

    Ok(result)
}
