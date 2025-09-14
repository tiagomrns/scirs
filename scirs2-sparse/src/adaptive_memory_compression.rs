//! Adaptive Memory Compression for Advanced-Large Sparse Matrices
//!
//! This module provides advanced memory management and compression techniques
//! specifically designed for handling advanced-large sparse matrices that exceed
//! available system memory.

use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

// Memory mapping support
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Configuration for adaptive memory compression
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionConfig {
    /// Maximum memory budget in bytes
    pub memory_budget: usize,
    /// Compression algorithm to use
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable hierarchical compression
    pub hierarchical_compression: bool,
    /// Block size for compression
    pub block_size: usize,
    /// Compression threshold (compress when usage exceeds this ratio)
    pub compression_threshold: f64,
    /// Enable adaptive compression based on access patterns
    pub adaptive_compression: bool,
    /// Cache size for frequently accessed blocks
    pub cache_size: usize,
    /// Enable out-of-core processing
    pub out_of_core: bool,
    /// Temporary directory for out-of-core storage
    pub temp_directory: String,
    /// Enable memory mapping
    pub memory_mapping: bool,
}

/// Compression algorithms for sparse matrix data
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Run-Length Encoding
    RLE,
    /// Delta encoding for indices
    Delta,
    /// Huffman coding
    Huffman,
    /// LZ77 compression
    LZ77,
    /// Sparse-specific compression
    SparseOptimized,
    /// Hybrid adaptive compression
    Adaptive,
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            memory_budget: 8 * 1024 * 1024 * 1024, // 8GB default
            compression_algorithm: CompressionAlgorithm::Adaptive,
            hierarchical_compression: true,
            block_size: 1024 * 1024, // 1MB blocks
            compression_threshold: 0.8,
            adaptive_compression: true,
            cache_size: 256 * 1024 * 1024, // 256MB cache
            out_of_core: true,
            temp_directory: "/tmp/scirs2_sparse".to_string(),
            memory_mapping: true,
        }
    }
}

/// Adaptive memory compression manager
#[allow(dead_code)]
pub struct AdaptiveMemoryCompressor {
    config: AdaptiveCompressionConfig,
    memory_usage: AtomicUsize,
    compression_stats: Arc<Mutex<CompressionStats>>,
    block_cache: Arc<Mutex<BlockCache>>,
    access_tracker: Arc<Mutex<AccessTracker>>,
    hierarchical_levels: Vec<CompressionLevel>,
    out_of_core_manager: Option<OutOfCoreManager>,
}

/// Statistics for compression operations
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    pub total_blocks: usize,
    pub compressed_blocks: usize,
    pub total_uncompressed_size: usize,
    pub total_compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
    pub decompression_time: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub out_of_core_reads: usize,
    pub out_of_core_writes: usize,
}

/// Block cache for frequently accessed data
#[derive(Debug)]
#[allow(dead_code)]
struct BlockCache {
    cache: HashMap<BlockId, CachedBlock>,
    access_order: VecDeque<BlockId>,
    _maxsize: usize,
    current_size: usize,
}

/// Cached block information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CachedBlock {
    data: Vec<u8>,
    compressed: bool,
    access_count: usize,
    last_access: u64,
    compression_level: u8,
}

/// Block identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BlockId {
    matrixid: u64,
    block_row: usize,
    block_col: usize,
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}-{}", self.matrixid, self.block_row, self.block_col)
    }
}

impl BlockId {
    /// Convert BlockId to u64 for serialization (using a hash-like approach)
    pub fn to_u64(&self) -> u64 {
        // Simple hash combining the fields
        self.matrixid
            .wrapping_mul(1000000)
            .wrapping_add((self.block_row as u64) * 1000)
            .wrapping_add(self.block_col as u64)
    }

    /// Create BlockId from u64 (for deserialization)
    pub fn from_u64(value: u64) -> Self {
        // This is a simplified reverse operation - in practice you'd want a proper bijection
        let matrixid = value / 1000000;
        let remainder = value % 1000000;
        let block_row = (remainder / 1000) as usize;
        let block_col = (remainder % 1000) as usize;
        Self {
            matrixid,
            block_row,
            block_col,
        }
    }
}

/// Access pattern tracking
#[derive(Debug, Default)]
#[allow(dead_code)]
struct AccessTracker {
    access_patterns: HashMap<BlockId, AccessPattern>,
    temporal_patterns: VecDeque<AccessEvent>,
    spatial_patterns: HashMap<usize, Vec<usize>>, // row -> accessed columns
    sequential_threshold: usize,
}

/// Access pattern for a block
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AccessPattern {
    access_count: usize,
    last_access: u64,
    access_frequency: f64,
    sequential_accesses: usize,
    random_accesses: usize,
    temporal_locality: f64,
    spatial_locality: f64,
}

/// Access event for pattern analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AccessEvent {
    blockid: BlockId,
    timestamp: u64,
    access_type: AccessType,
}

/// Type of access
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Hierarchical compression level
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CompressionLevel {
    level: u8,
    compression_ratio: f64,
    algorithm: CompressionAlgorithm,
    block_size: usize,
    access_threshold: usize,
}

/// Out-of-core memory manager
#[derive(Debug)]
#[allow(dead_code)]
struct OutOfCoreManager {
    temp_dir: String,
    file_counter: AtomicUsize,
    active_files: HashMap<BlockId, String>,
    memory_mapped_files: HashMap<String, MemoryMappedFile>,
}

/// Memory-mapped file wrapper
#[derive(Debug)]
#[allow(dead_code)]
struct MemoryMappedFile {
    filepath: PathBuf,
    file: File,
    size: usize,
    mapped: bool,
    // Memory mapping would require unsafe code and platform-specific implementation
    // For now, we'll use buffered I/O as a safe alternative
    _phantom: PhantomData<()>,
}

impl MemoryMappedFile {
    /// Create a new memory-mapped file
    #[allow(dead_code)]
    fn new(filepath: PathBuf, size: usize) -> SparseResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to create file {filepath:?}: {e}")))?;

        // Set file size if creating new file
        file.set_len(size as u64)
            .map_err(|e| SparseError::Io(format!("Failed to set file size: {e}")))?;

        Ok(Self {
            filepath,
            file,
            size,
            mapped: true, // We'll treat buffered I/O as "mapped" for this implementation
            _phantom: PhantomData,
        })
    }

    /// Read data from the mapped file at offset
    #[allow(dead_code)]
    fn read_at(&self, offset: usize, buffer: &mut [u8]) -> SparseResult<usize> {
        #[cfg(unix)]
        {
            self.file
                .read_at(buffer, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to read at offset {offset}: {e}")))
        }

        #[cfg(windows)]
        {
            let mut file_ref = &self.file;
            file_ref
                .seek_read(buffer, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to read at offset {offset}: {e}")))
        }

        #[cfg(not(any(unix, windows)))]
        {
            // Fallback for other platforms - use regular seeking
            let mut file_clone = self
                .file
                .try_clone()
                .map_err(|e| SparseError::Io(format!("Failed to clone file handle: {e}")))?;
            file_clone
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| SparseError::Io(format!("Failed to seek to offset {offset}: {e}")))?;
            file_clone
                .read(buffer)
                .map_err(|e| SparseError::Io(format!("Failed to read data: {e}")))
        }
    }

    /// Write data to the mapped file at offset
    #[allow(dead_code)]
    fn write_at(&self, offset: usize, data: &[u8]) -> SparseResult<usize> {
        #[cfg(unix)]
        {
            self.file
                .write_at(data, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to write at offset {offset}: {e}")))
        }

        #[cfg(windows)]
        {
            let mut file_ref = &self.file;
            file_ref
                .seek_write(data, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to write at offset {offset}: {e}")))
        }

        #[cfg(not(any(unix, windows)))]
        {
            // Fallback for other platforms - use regular seeking
            let mut file_clone = self
                .file
                .try_clone()
                .map_err(|e| SparseError::Io(format!("Failed to clone file handle: {e}")))?;
            file_clone
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| SparseError::Io(format!("Failed to seek to offset {offset}: {e}")))?;
            file_clone
                .write(data)
                .map_err(|e| SparseError::Io(format!("Failed to write data: {e}")))
        }
    }

    /// Flush data to disk
    #[allow(dead_code)]
    fn flush(&self) -> SparseResult<()> {
        self.file
            .sync_all()
            .map_err(|e| SparseError::Io(format!("Failed to flush file: {e}")))
    }
}

impl AdaptiveMemoryCompressor {
    /// Create a new adaptive memory compressor
    pub fn new(config: AdaptiveCompressionConfig) -> SparseResult<Self> {
        let block_cache = BlockCache::new(config.cache_size);
        let access_tracker = AccessTracker::default();

        // Initialize hierarchical compression levels
        let hierarchical_levels = vec![
            CompressionLevel {
                level: 1,
                compression_ratio: 2.0,
                algorithm: CompressionAlgorithm::RLE,
                block_size: config.block_size,
                access_threshold: 100,
            },
            CompressionLevel {
                level: 2,
                compression_ratio: 4.0,
                algorithm: CompressionAlgorithm::Delta,
                block_size: config.block_size / 2,
                access_threshold: 50,
            },
            CompressionLevel {
                level: 3,
                compression_ratio: 8.0,
                algorithm: CompressionAlgorithm::LZ77,
                block_size: config.block_size / 4,
                access_threshold: 10,
            },
        ];

        // Initialize out-of-core manager if enabled
        let out_of_core_manager = if config.out_of_core {
            Some(OutOfCoreManager::new(&config.temp_directory)?)
        } else {
            None
        };

        Ok(Self {
            config,
            memory_usage: AtomicUsize::new(0),
            compression_stats: Arc::new(Mutex::new(CompressionStats::default())),
            block_cache: Arc::new(Mutex::new(block_cache)),
            access_tracker: Arc::new(Mutex::new(access_tracker)),
            hierarchical_levels,
            out_of_core_manager,
        })
    }

    /// Compress sparse matrix data adaptively
    #[allow(clippy::too_many_arguments)]
    pub fn compress_matrix<T>(
        &mut self,
        matrixid: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<CompressedMatrix<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let total_size = std::mem::size_of_val(indptr)
            + std::mem::size_of_val(indices)
            + std::mem::size_of_val(data);

        // Check if compression is needed
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let usage_ratio = (current_usage + total_size) as f64 / self.config.memory_budget as f64;

        if usage_ratio < self.config.compression_threshold && !self.config.adaptive_compression {
            // No compression needed
            return self.create_uncompressed_matrix(matrixid, rows, indptr, indices, data);
        }

        let start_time = std::time::Instant::now();

        // Determine optimal compression strategy
        let compression_strategy =
            self.determine_compression_strategy(matrixid, rows, indptr, indices)?;

        // Apply compression based on strategy
        let compressed_blocks = match compression_strategy.algorithm {
            CompressionAlgorithm::None => {
                self.compress_with_none(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::RLE => {
                self.compress_with_rle(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::Delta => {
                self.compress_with_delta(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::Huffman => {
                self.compress_with_huffman(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::LZ77 => {
                self.compress_with_lz77(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::SparseOptimized => {
                self.compress_with_sparse_optimized(matrixid, rows, indptr, indices, data)?
            }
            CompressionAlgorithm::Adaptive => {
                self.compress_with_adaptive(matrixid, rows, indptr, indices, data)?
            }
        };

        let compression_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        self.update_compression_stats(total_size, &compressed_blocks, compression_time);

        // Update memory usage
        let compressed_size = compressed_blocks
            .iter()
            .map(|b| b.compressed_data.len())
            .sum::<usize>();
        self.memory_usage
            .fetch_add(compressed_size, Ordering::Relaxed);

        Ok(CompressedMatrix {
            matrixid,
            original_rows: rows,
            original_cols: if !indptr.is_empty() {
                *indices.iter().max().unwrap_or(&0) + 1
            } else {
                0
            },
            compressed_blocks,
            compression_algorithm: compression_strategy.algorithm,
            block_size: compression_strategy.block_size,
            metadata: CompressionMetadata {
                original_size: total_size,
                compressed_size,
                compression_ratio: total_size as f64 / compressed_size.max(1) as f64,
                compression_time,
            },
            _phantom: PhantomData,
        })
    }

    /// Decompress matrix data
    pub fn decompress_matrix<T>(
        &mut self,
        compressed_matrix: &CompressedMatrix<T>,
    ) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<T>)>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let start_time = std::time::Instant::now();

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut data = Vec::new();

        // Decompress each block
        for block in &compressed_matrix.compressed_blocks {
            let decompressed_data =
                self.decompress_block(block, compressed_matrix.compression_algorithm)?;

            // Parse decompressed data based on block type
            match block.block_type {
                BlockType::IndPtr => {
                    indptr.extend(self.parse_indptr_data(&decompressed_data)?);
                }
                BlockType::Indices => {
                    indices.extend(self.parse_indices_data(&decompressed_data)?);
                }
                BlockType::Data => {
                    data.extend(self.parse_data_values::<T>(&decompressed_data)?);
                }
                BlockType::Combined => {
                    let (block_indptr, block_indices, block_data) =
                        self.parse_combined_data::<T>(&decompressed_data)?;
                    indptr.extend(block_indptr);
                    indices.extend(block_indices);
                    data.extend(block_data);
                }
            }
        }

        let decompression_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        if let Ok(mut stats) = self.compression_stats.lock() {
            stats.decompression_time += decompression_time;
        }

        Ok((indptr, indices, data))
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let usage_ratio = current_usage as f64 / self.config.memory_budget as f64;

        let compression_stats = self.compression_stats.lock().unwrap().clone();
        let cache_stats = self.get_cache_stats();

        MemoryStats {
            total_memory_budget: self.config.memory_budget,
            current_memory_usage: current_usage,
            memory_usage_ratio: usage_ratio,
            compression_stats,
            cache_hits: cache_stats.hits,
            cache_misses: cache_stats.misses,
            cache_hit_ratio: cache_stats.hit_ratio,
            out_of_core_enabled: self.config.out_of_core,
        }
    }

    // Private helper methods

    fn determine_compression_strategy(
        &self,
        matrixid: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
    ) -> SparseResult<CompressionStrategy> {
        // Analyze matrix characteristics
        let nnz = indices.len();
        let density = if rows > 0 && !indices.is_empty() {
            let max_col = *indices.iter().max().unwrap_or(&0);
            nnz as f64 / (rows as f64 * (max_col + 1) as f64)
        } else {
            0.0
        };

        // Analyze sparsity patterns
        let pattern_analysis = self.analyze_sparsity_patterns(indptr, indices);

        // Check access patterns if available
        let access_info = self.get_access_pattern_info(matrixid);

        // Select compression algorithm based on analysis
        let algorithm = if self.config.adaptive_compression {
            self.select_adaptive_algorithm(density, &pattern_analysis, &access_info)
        } else {
            self.config.compression_algorithm
        };

        // Determine block size
        let block_size = self.determine_optimal_block_size(rows, nnz, density);

        Ok(CompressionStrategy {
            algorithm,
            block_size,
            hierarchical: self.config.hierarchical_compression,
            predicted_ratio: self.predict_compression_ratio(algorithm, density, &pattern_analysis),
        })
    }

    fn analyze_sparsity_patterns(
        &self,
        indptr: &[usize],
        indices: &[usize],
    ) -> SparsityPatternAnalysis {
        let mut analysis = SparsityPatternAnalysis::default();

        if indptr.len() <= 1 {
            return analysis;
        }

        let rows = indptr.len() - 1;

        // Analyze row distribution
        let mut row_nnz = Vec::new();
        for row in 0..rows {
            row_nnz.push(indptr[row + 1] - indptr[row]);
        }

        analysis.avg_nnz_per_row = row_nnz.iter().sum::<usize>() as f64 / rows as f64;
        analysis.max_nnz_per_row = *row_nnz.iter().max().unwrap_or(&0);
        analysis.min_nnz_per_row = *row_nnz.iter().min().unwrap_or(&0);

        // Analyze column patterns
        analysis.sequential_patterns = self.count_sequential_patterns(indices);
        analysis.clustering_factor = self.calculate_clustering_factor(indptr, indices);
        analysis.bandwidth = self.calculate_bandwidth(indptr, indices);

        analysis
    }

    fn count_sequential_patterns(&self, indices: &[usize]) -> usize {
        let mut sequential_count = 0;
        let mut current_sequence = 0;

        for window in indices.windows(2) {
            if window[1] == window[0] + 1 {
                current_sequence += 1;
            } else {
                if current_sequence >= 3 {
                    sequential_count += current_sequence;
                }
                current_sequence = 0;
            }
        }

        if current_sequence >= 3 {
            sequential_count += current_sequence;
        }

        sequential_count
    }

    fn calculate_clustering_factor(&self, indptr: &[usize], indices: &[usize]) -> f64 {
        if indptr.len() <= 1 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut total_pairs = 0;

        let rows = indptr.len() - 1;
        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            if end > start + 1 {
                for i in start..(end - 1) {
                    total_distance += (indices[i + 1] - indices[i]) as f64;
                    total_pairs += 1;
                }
            }
        }

        if total_pairs > 0 {
            total_distance / total_pairs as f64
        } else {
            0.0
        }
    }

    fn calculate_bandwidth(&self, indptr: &[usize], indices: &[usize]) -> usize {
        if indptr.len() <= 1 {
            return 0;
        }

        let mut max_bandwidth = 0;
        let rows = indptr.len() - 1;

        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            if end > start {
                let min_col = indices[start];
                let max_col = indices[end - 1];
                let bandwidth = max_col.saturating_sub(min_col);
                max_bandwidth = max_bandwidth.max(bandwidth);
            }
        }

        max_bandwidth
    }

    fn get_access_pattern_info(&self, matrixid: u64) -> AccessPatternInfo {
        let access_tracker = self.access_tracker.lock().unwrap();
        let mut info = AccessPatternInfo::default();

        // Aggregate access patterns for this matrix
        for (blockid, pattern) in &access_tracker.access_patterns {
            if blockid.matrixid == matrixid {
                info.total_accesses += pattern.access_count;
                info.avg_temporal_locality += pattern.temporal_locality;
                info.avg_spatial_locality += pattern.spatial_locality;
                info.pattern_count += 1;
            }
        }

        if info.pattern_count > 0 {
            info.avg_temporal_locality /= info.pattern_count as f64;
            info.avg_spatial_locality /= info.pattern_count as f64;
        }

        info
    }

    fn select_adaptive_algorithm(
        &self,
        density: f64,
        pattern_analysis: &SparsityPatternAnalysis,
        access_info: &AccessPatternInfo,
    ) -> CompressionAlgorithm {
        // Decision tree for algorithm selection
        if density > 0.1 {
            // Dense matrices benefit from general compression
            CompressionAlgorithm::LZ77
        } else if pattern_analysis.sequential_patterns
            > pattern_analysis.avg_nnz_per_row as usize * 10
        {
            // High sequential patterns favor RLE
            CompressionAlgorithm::RLE
        } else if pattern_analysis.clustering_factor < 2.0 {
            // Low clustering suggests delta encoding
            CompressionAlgorithm::Delta
        } else if access_info.avg_temporal_locality > 0.8 {
            // High temporal locality suggests sparse-optimized
            CompressionAlgorithm::SparseOptimized
        } else {
            // Default to Huffman for general case
            CompressionAlgorithm::Huffman
        }
    }

    fn determine_optimal_block_size(&self, rows: usize, nnz: usize, density: f64) -> usize {
        let base_block_size = self.config.block_size;

        // Adjust block size based on matrix characteristics
        let size_factor = if rows > 1_000_000 {
            2.0 // Larger blocks for large matrices
        } else if rows < 10_000 {
            0.5 // Smaller blocks for small matrices
        } else {
            1.0
        };

        let density_factor = if density > 0.1 {
            1.5 // Larger blocks for denser matrices
        } else {
            1.0
        };

        let nnz_factor = if nnz > 10_000_000 {
            1.5 // Larger blocks for many non-zeros
        } else {
            1.0
        };

        let optimal_size =
            (base_block_size as f64 * size_factor * density_factor * nnz_factor) as usize;
        optimal_size.clamp(4096, 16 * 1024 * 1024) // 4KB to 16MB range
    }

    fn predict_compression_ratio(
        &self,
        algorithm: CompressionAlgorithm,
        density: f64,
        pattern_analysis: &SparsityPatternAnalysis,
    ) -> f64 {
        let base_ratio = match algorithm {
            CompressionAlgorithm::None => 1.0,
            CompressionAlgorithm::RLE => 2.0 + pattern_analysis.sequential_patterns as f64 / 1000.0,
            CompressionAlgorithm::Delta => {
                1.5 + (10.0 - pattern_analysis.clustering_factor).max(0.0) * 0.1
            }
            CompressionAlgorithm::Huffman => 2.5 + (1.0 - density) * 2.0,
            CompressionAlgorithm::LZ77 => 3.0 + (1.0 - density) * 1.5,
            CompressionAlgorithm::SparseOptimized => 4.0 + (1.0 - density) * 3.0,
            CompressionAlgorithm::Adaptive => 3.5 + (1.0 - density) * 2.5,
        };

        // Adjust based on matrix characteristics
        let adjustment = if pattern_analysis.bandwidth > 100000 {
            0.8 // Lower compression for high bandwidth
        } else if pattern_analysis.bandwidth < 100 {
            1.2 // Higher compression for low bandwidth
        } else {
            1.0
        };

        base_ratio * adjustment
    }

    // Compression algorithm implementations (simplified)

    fn compress_with_none<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        // Create uncompressed blocks
        let blocks = vec![
            CompressedBlock {
                blockid: BlockId {
                    matrixid,
                    block_row: 0,
                    block_col: 0,
                },
                block_type: BlockType::IndPtr,
                compressed_data: self.serialize_indptr(indptr)?,
                original_size: std::mem::size_of_val(indptr),
                compression_level: 0,
            },
            CompressedBlock {
                blockid: BlockId {
                    matrixid,
                    block_row: 0,
                    block_col: 1,
                },
                block_type: BlockType::Indices,
                compressed_data: self.serialize_indices(indices)?,
                original_size: std::mem::size_of_val(indices),
                compression_level: 0,
            },
            CompressedBlock {
                blockid: BlockId {
                    matrixid,
                    block_row: 0,
                    block_col: 2,
                },
                block_type: BlockType::Data,
                compressed_data: self.serialize_data(data)?,
                original_size: std::mem::size_of_val(data),
                compression_level: 0,
            },
        ];

        Ok(blocks)
    }

    fn compress_with_rle<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let mut blocks = Vec::new();

        // Apply RLE compression to indices (most beneficial)
        let compressed_indices = self.apply_rle_compression(indices)?;

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 0,
            },
            block_type: BlockType::IndPtr,
            compressed_data: self.serialize_indptr(indptr)?,
            original_size: std::mem::size_of_val(indptr),
            compression_level: 0,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 1,
            },
            block_type: BlockType::Indices,
            compressed_data: compressed_indices,
            original_size: std::mem::size_of_val(indices),
            compression_level: 1,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 2,
            },
            block_type: BlockType::Data,
            compressed_data: self.serialize_data(data)?,
            original_size: std::mem::size_of_val(data),
            compression_level: 0,
        });

        Ok(blocks)
    }

    fn compress_with_delta<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let mut blocks = Vec::new();

        // Apply delta compression to both indptr and indices
        let compressed_indptr = self.apply_delta_compression(indptr)?;
        let compressed_indices = self.apply_delta_compression(indices)?;

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 0,
            },
            block_type: BlockType::IndPtr,
            compressed_data: compressed_indptr,
            original_size: std::mem::size_of_val(indptr),
            compression_level: 1,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 1,
            },
            block_type: BlockType::Indices,
            compressed_data: compressed_indices,
            original_size: std::mem::size_of_val(indices),
            compression_level: 1,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 2,
            },
            block_type: BlockType::Data,
            compressed_data: self.serialize_data(data)?,
            original_size: std::mem::size_of_val(data),
            compression_level: 0,
        });

        Ok(blocks)
    }

    fn compress_with_huffman<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let mut blocks = Vec::new();

        // Apply Huffman compression to indices (most beneficial for sparse data)
        let compressed_indices = self.apply_huffman_compression(indices)?;

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 0,
            },
            block_type: BlockType::IndPtr,
            compressed_data: self.serialize_indptr(indptr)?,
            original_size: std::mem::size_of_val(indptr),
            compression_level: 0,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 1,
            },
            block_type: BlockType::Indices,
            compressed_data: compressed_indices,
            original_size: std::mem::size_of_val(indices),
            compression_level: 2,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 2,
            },
            block_type: BlockType::Data,
            compressed_data: self.serialize_data(data)?,
            original_size: std::mem::size_of_val(data),
            compression_level: 0,
        });

        Ok(blocks)
    }

    fn compress_with_lz77<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let mut blocks = Vec::new();

        // Apply LZ77 compression to all components
        let compressed_indptr = self.apply_lz77_compression(indptr)?;
        let compressed_indices = self.apply_lz77_compression(indices)?;

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 0,
            },
            block_type: BlockType::IndPtr,
            compressed_data: compressed_indptr,
            original_size: std::mem::size_of_val(indptr),
            compression_level: 3,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 1,
            },
            block_type: BlockType::Indices,
            compressed_data: compressed_indices,
            original_size: std::mem::size_of_val(indices),
            compression_level: 3,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 2,
            },
            block_type: BlockType::Data,
            compressed_data: self.serialize_data(data)?,
            original_size: std::mem::size_of_val(data),
            compression_level: 0,
        });

        Ok(blocks)
    }

    fn compress_with_sparse_optimized<T>(
        &self,
        matrixid: u64,
        _rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        // Combine delta compression for indices with RLE for patterns
        let mut blocks = Vec::new();

        let compressed_indptr = self.apply_delta_compression(indptr)?;
        let compressed_indices = self.apply_sparse_optimized_compression(indices)?;

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 0,
            },
            block_type: BlockType::IndPtr,
            compressed_data: compressed_indptr,
            original_size: std::mem::size_of_val(indptr),
            compression_level: 2,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 1,
            },
            block_type: BlockType::Indices,
            compressed_data: compressed_indices,
            original_size: std::mem::size_of_val(indices),
            compression_level: 2,
        });

        blocks.push(CompressedBlock {
            blockid: BlockId {
                matrixid,
                block_row: 0,
                block_col: 2,
            },
            block_type: BlockType::Data,
            compressed_data: self.serialize_data(data)?,
            original_size: std::mem::size_of_val(data),
            compression_level: 0,
        });

        Ok(blocks)
    }

    fn compress_with_adaptive<T>(
        &self,
        matrixid: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        // Try multiple algorithms and pick the best
        let strategies = vec![
            CompressionAlgorithm::RLE,
            CompressionAlgorithm::Delta,
            CompressionAlgorithm::SparseOptimized,
        ];

        let mut best_compression = None;
        let mut best_ratio = 0.0;

        for &strategy in &strategies {
            let test_blocks = match strategy {
                CompressionAlgorithm::RLE => {
                    self.compress_with_rle(matrixid, rows, indptr, indices, data)?
                }
                CompressionAlgorithm::Delta => {
                    self.compress_with_delta(matrixid, rows, indptr, indices, data)?
                }
                CompressionAlgorithm::SparseOptimized => {
                    self.compress_with_sparse_optimized(matrixid, rows, indptr, indices, data)?
                }
                _ => continue,
            };

            let original_size = test_blocks.iter().map(|b| b.original_size).sum::<usize>();
            let compressed_size = test_blocks
                .iter()
                .map(|b| b.compressed_data.len())
                .sum::<usize>();
            let ratio = original_size as f64 / compressed_size.max(1) as f64;

            if ratio > best_ratio {
                best_ratio = ratio;
                best_compression = Some(test_blocks);
            }
        }

        best_compression.ok_or_else(|| {
            SparseError::CompressionError("No valid compression strategy found".to_string())
        })
    }

    // Helper compression methods

    fn apply_rle_compression(&self, data: &[usize]) -> SparseResult<Vec<u8>> {
        let mut compressed = Vec::new();

        if data.is_empty() {
            return Ok(compressed);
        }

        let mut i = 0;
        while i < data.len() {
            let current_value = data[i];
            let mut run_length = 1;

            // Count consecutive occurrences
            while i + run_length < data.len() && data[i + run_length] == current_value {
                run_length += 1;
            }

            // Encode value and run length
            compressed.extend_from_slice(&current_value.to_le_bytes());
            compressed.extend_from_slice(&run_length.to_le_bytes());

            i += run_length;
        }

        Ok(compressed)
    }

    fn apply_delta_compression(&self, data: &[usize]) -> SparseResult<Vec<u8>> {
        let mut compressed = Vec::new();

        if data.is_empty() {
            return Ok(compressed);
        }

        // Store first value as-is
        compressed.extend_from_slice(&data[0].to_le_bytes());

        // Store deltas
        for i in 1..data.len() {
            let delta = data[i].wrapping_sub(data[i - 1]);
            compressed.extend_from_slice(&delta.to_le_bytes());
        }

        Ok(compressed)
    }

    fn apply_sparse_optimized_compression(&self, indices: &[usize]) -> SparseResult<Vec<u8>> {
        // Combine delta and RLE based on pattern detection
        let mut compressed = Vec::new();

        if indices.is_empty() {
            return Ok(compressed);
        }

        // First pass: detect patterns
        let has_sequential = self.has_sequential_patterns(indices);
        let has_repeating = self.has_repeating_patterns(indices);

        if has_sequential && !has_repeating {
            // Use delta compression
            compressed = self.apply_delta_compression(indices)?;
        } else if has_repeating && !has_sequential {
            // Use RLE compression
            compressed = self.apply_rle_compression(indices)?;
        } else {
            // Use hybrid approach
            compressed = self.apply_hybrid_compression(indices)?;
        }

        Ok(compressed)
    }

    fn has_sequential_patterns(&self, indices: &[usize]) -> bool {
        let mut sequential_count = 0;
        for window in indices.windows(2) {
            if window[1] == window[0] + 1 {
                sequential_count += 1;
            }
        }
        sequential_count > indices.len() / 4
    }

    fn has_repeating_patterns(&self, indices: &[usize]) -> bool {
        let mut repeating_count = 0;
        for window in indices.windows(2) {
            if window[1] == window[0] {
                repeating_count += 1;
            }
        }
        repeating_count > indices.len() / 10
    }

    fn apply_hybrid_compression(&self, indices: &[usize]) -> SparseResult<Vec<u8>> {
        // Simplified hybrid: just use delta for now
        self.apply_delta_compression(indices)
    }

    fn apply_huffman_compression(&self, data: &[usize]) -> SparseResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Build frequency table
        let mut frequency = HashMap::new();
        for &value in data {
            *frequency.entry(value).or_insert(0) += 1;
        }

        // Build Huffman tree (simplified - priority queue would be better)
        let mut huffman_codes = HashMap::new();
        let mut sorted_freq: Vec<_> = frequency.into_iter().collect();
        sorted_freq.sort_by_key(|&(_, freq)| freq);

        // Assign codes based on frequency (simplified Huffman coding)
        let mut code_length = 1u8;
        let mut codes_assigned = 0u32;
        let _total_symbols = sorted_freq.len();

        for (value, freq) in sorted_freq {
            // Simple code assignment - in practice, this would be a proper Huffman tree
            let code = codes_assigned;
            huffman_codes.insert(value, (code, code_length));
            codes_assigned += 1;

            // Increase code length when we've used half the available codes
            if codes_assigned >= (1 << code_length) / 2 && code_length < 16 {
                code_length += 1;
            }
        }

        // Encode data
        let mut compressed = Vec::new();

        // Store the huffman table size and codes first
        compressed.extend_from_slice(&huffman_codes.len().to_le_bytes());
        for (&value, &(code, length)) in &huffman_codes {
            compressed.extend_from_slice(&value.to_le_bytes());
            compressed.extend_from_slice(&code.to_le_bytes());
            compressed.push(length);
        }

        // Store original data length
        compressed.extend_from_slice(&data.len().to_le_bytes());

        // Encode actual data (simplified bit packing)
        let mut bit_buffer = 0u64;
        let mut bit_count = 0;

        for &value in data {
            if let Some(&(code, length)) = huffman_codes.get(&value) {
                bit_buffer |= (code as u64) << bit_count;
                bit_count += length as usize;

                // Flush complete bytes
                while bit_count >= 8 {
                    compressed.push((bit_buffer & 0xFF) as u8);
                    bit_buffer >>= 8;
                    bit_count -= 8;
                }
            }
        }

        // Flush remaining bits
        if bit_count > 0 {
            compressed.push((bit_buffer & 0xFF) as u8);
        }

        Ok(compressed)
    }

    fn apply_lz77_compression(&self, data: &[usize]) -> SparseResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();

        // Store original length
        compressed.extend_from_slice(&data.len().to_le_bytes());

        // LZ77 parameters
        const WINDOW_SIZE: usize = 4096;
        const LOOKAHEAD_SIZE: usize = 256;
        const MIN_MATCH_LENGTH: usize = 3;

        // Convert usize data to bytes for processing
        let mut byte_data = Vec::new();
        for &value in data {
            byte_data.extend_from_slice(&value.to_le_bytes());
        }

        let mut pos = 0;
        while pos < byte_data.len() {
            let mut best_match_length = 0;
            let mut best_match_distance = 0;

            // Search for matches in the sliding window
            let window_start = pos.saturating_sub(WINDOW_SIZE);
            let lookahead_end = (pos + LOOKAHEAD_SIZE).min(byte_data.len());

            // Find the longest match
            for search_pos in window_start..pos {
                let mut match_length = 0;
                let max_length = (lookahead_end - pos).min(pos - search_pos);

                while match_length < max_length
                    && byte_data[search_pos + match_length] == byte_data[pos + match_length]
                {
                    match_length += 1;
                }

                if match_length >= MIN_MATCH_LENGTH && match_length > best_match_length {
                    best_match_length = match_length;
                    best_match_distance = pos - search_pos;
                }
            }

            if best_match_length >= MIN_MATCH_LENGTH {
                // Encode (distance, length, next_char)
                compressed.push(1); // Flag for match
                compressed.extend_from_slice(&best_match_distance.to_le_bytes());
                compressed.extend_from_slice(&best_match_length.to_le_bytes());

                // Next character after the match
                let next_pos = pos + best_match_length;
                if next_pos < byte_data.len() {
                    compressed.push(byte_data[next_pos]);
                    pos = next_pos + 1;
                } else {
                    pos = next_pos;
                }
            } else {
                // Encode literal
                compressed.push(0); // Flag for literal
                compressed.push(byte_data[pos]);
                pos += 1;
            }
        }

        Ok(compressed)
    }

    // Decompression methods

    fn decompress_rle(&self, compressed: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        let mut decompressed = Vec::new();
        let mut pos = 0;

        while pos + 16 <= compressed.len() {
            // Read value and run length
            let value = usize::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
                compressed[pos + 4],
                compressed[pos + 5],
                compressed[pos + 6],
                compressed[pos + 7],
            ]);
            let run_length = usize::from_le_bytes([
                compressed[pos + 8],
                compressed[pos + 9],
                compressed[pos + 10],
                compressed[pos + 11],
                compressed[pos + 12],
                compressed[pos + 13],
                compressed[pos + 14],
                compressed[pos + 15],
            ]);

            // Expand the run
            for _ in 0..run_length {
                decompressed.extend_from_slice(&value.to_le_bytes());
            }

            pos += 16;
        }

        Ok(decompressed)
    }

    fn decompress_delta(&self, compressed: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed.len() < 8 {
            return Ok(Vec::new());
        }

        let mut decompressed = Vec::new();
        let mut pos = 0;

        // Read first value
        let first_value = usize::from_le_bytes([
            compressed[pos],
            compressed[pos + 1],
            compressed[pos + 2],
            compressed[pos + 3],
            compressed[pos + 4],
            compressed[pos + 5],
            compressed[pos + 6],
            compressed[pos + 7],
        ]);
        decompressed.extend_from_slice(&first_value.to_le_bytes());
        pos += 8;

        let mut current_value = first_value;

        // Read and apply deltas
        while pos + 8 <= compressed.len() {
            let delta = usize::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
                compressed[pos + 4],
                compressed[pos + 5],
                compressed[pos + 6],
                compressed[pos + 7],
            ]);

            current_value = current_value.wrapping_add(delta);
            decompressed.extend_from_slice(&current_value.to_le_bytes());
            pos += 8;
        }

        Ok(decompressed)
    }

    fn decompress_huffman(&self, compressed: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed.len() < 8 {
            return Ok(Vec::new());
        }

        let mut pos = 0;

        // Read huffman table size
        let table_size = usize::from_le_bytes([
            compressed[pos],
            compressed[pos + 1],
            compressed[pos + 2],
            compressed[pos + 3],
            compressed[pos + 4],
            compressed[pos + 5],
            compressed[pos + 6],
            compressed[pos + 7],
        ]);
        pos += 8;

        // Read huffman table
        let mut huffman_decode = HashMap::new();
        for _ in 0..table_size {
            if pos + 17 > compressed.len() {
                break;
            }

            let value = usize::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
                compressed[pos + 4],
                compressed[pos + 5],
                compressed[pos + 6],
                compressed[pos + 7],
            ]);
            let code = usize::from_le_bytes([
                compressed[pos + 8],
                compressed[pos + 9],
                compressed[pos + 10],
                compressed[pos + 11],
                compressed[pos + 12],
                compressed[pos + 13],
                compressed[pos + 14],
                compressed[pos + 15],
            ]);
            let length = compressed[pos + 16] as usize;

            huffman_decode.insert((code, length), value);
            pos += 17;
        }

        if pos + 8 > compressed.len() {
            return Ok(Vec::new());
        }

        // Read original data length
        let data_length = usize::from_le_bytes([
            compressed[pos],
            compressed[pos + 1],
            compressed[pos + 2],
            compressed[pos + 3],
            compressed[pos + 4],
            compressed[pos + 5],
            compressed[pos + 6],
            compressed[pos + 7],
        ]);
        pos += 8;

        // Decode data (simplified bit unpacking)
        let mut decompressed = Vec::new();
        let mut bit_buffer = 0u64;
        let mut bit_count = 0;
        let mut decoded_count = 0;

        while pos < compressed.len() && decoded_count < data_length {
            // Fill bit buffer
            if bit_count < 32 && pos < compressed.len() {
                bit_buffer |= (compressed[pos] as u64) << bit_count;
                bit_count += 8;
                pos += 1;
            }

            // Try to decode a symbol
            let mut found = false;
            for length in 1..=16 {
                if bit_count >= length {
                    let code = (bit_buffer & ((1 << length) - 1)) as usize;
                    if let Some(&value) = huffman_decode.get(&(code, length)) {
                        decompressed.extend_from_slice(&value.to_le_bytes());
                        decoded_count += 1;
                        bit_buffer >>= length;
                        bit_count -= length;
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                // Skip a bit if no match found (error recovery)
                if bit_count > 0 {
                    bit_buffer >>= 1;
                    bit_count -= 1;
                } else {
                    break;
                }
            }
        }

        Ok(decompressed)
    }

    fn decompress_lz77(&self, compressed: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed.len() < 8 {
            return Ok(Vec::new());
        }

        let mut pos = 0;

        // Read original length
        let original_length = usize::from_le_bytes([
            compressed[pos],
            compressed[pos + 1],
            compressed[pos + 2],
            compressed[pos + 3],
            compressed[pos + 4],
            compressed[pos + 5],
            compressed[pos + 6],
            compressed[pos + 7],
        ]);
        pos += 8;

        let mut decompressed = Vec::new();

        while pos < compressed.len() {
            if pos >= compressed.len() {
                break;
            }

            let flag = compressed[pos];
            pos += 1;

            if flag == 1 {
                // Match: read distance and length
                if pos + 16 > compressed.len() {
                    break;
                }

                let distance = usize::from_le_bytes([
                    compressed[pos],
                    compressed[pos + 1],
                    compressed[pos + 2],
                    compressed[pos + 3],
                    compressed[pos + 4],
                    compressed[pos + 5],
                    compressed[pos + 6],
                    compressed[pos + 7],
                ]);
                let length = usize::from_le_bytes([
                    compressed[pos + 8],
                    compressed[pos + 9],
                    compressed[pos + 10],
                    compressed[pos + 11],
                    compressed[pos + 12],
                    compressed[pos + 13],
                    compressed[pos + 14],
                    compressed[pos + 15],
                ]);
                pos += 16;

                // Copy from previous position
                let start_pos = decompressed.len().saturating_sub(distance);
                for i in 0..length {
                    if start_pos + i < decompressed.len() {
                        let byte_to_copy = decompressed[start_pos + i];
                        decompressed.push(byte_to_copy);
                    }
                }

                // Add next character if available
                if pos < compressed.len() {
                    decompressed.push(compressed[pos]);
                    pos += 1;
                }
            } else {
                // Literal: copy directly
                if pos < compressed.len() {
                    decompressed.push(compressed[pos]);
                    pos += 1;
                }
            }
        }

        // Convert back to usize array format and then to bytes
        let mut result = Vec::new();
        let usize_count = original_length;
        let bytes_per_usize = std::mem::size_of::<usize>();

        result.extend_from_slice(&usize_count.to_le_bytes());

        for chunk in decompressed.chunks(bytes_per_usize) {
            if chunk.len() == bytes_per_usize {
                result.extend_from_slice(chunk);
            } else {
                // Pad incomplete chunk
                let mut padded = chunk.to_vec();
                padded.resize(bytes_per_usize, 0);
                result.extend_from_slice(&padded);
            }
        }

        Ok(result)
    }

    // Serialization methods

    fn serialize_indptr(&self, indptr: &[usize]) -> SparseResult<Vec<u8>> {
        let mut serialized = Vec::new();

        // Write length
        serialized.extend_from_slice(&indptr.len().to_le_bytes());

        // Write data
        for &value in indptr {
            serialized.extend_from_slice(&value.to_le_bytes());
        }

        Ok(serialized)
    }

    fn serialize_indices(&self, indices: &[usize]) -> SparseResult<Vec<u8>> {
        let mut serialized = Vec::new();

        // Write length
        serialized.extend_from_slice(&indices.len().to_le_bytes());

        // Write data
        for &value in indices {
            serialized.extend_from_slice(&value.to_le_bytes());
        }

        Ok(serialized)
    }

    fn serialize_data<T>(&self, data: &[T]) -> SparseResult<Vec<u8>>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        let mut serialized = Vec::new();

        // Write length
        serialized.extend_from_slice(&data.len().to_le_bytes());

        // Write data (simplified - assumes f64)
        for &value in data {
            let bytes = value.to_f64().unwrap_or(0.0).to_le_bytes();
            serialized.extend_from_slice(&bytes);
        }

        Ok(serialized)
    }

    // Decompression and parsing methods (simplified implementations)

    fn decompress_block(
        &self,
        block: &CompressedBlock,
        algorithm: CompressionAlgorithm,
    ) -> SparseResult<Vec<u8>> {
        match block.compression_level {
            0 => {
                // No compression - return as-is
                Ok(block.compressed_data.clone())
            }
            1 => {
                // RLE or Delta compression
                match block.block_type {
                    BlockType::Indices => self.decompress_rle(&block.compressed_data),
                    BlockType::IndPtr => self.decompress_delta(&block.compressed_data),
                    _ => Ok(block.compressed_data.clone()),
                }
            }
            2 => {
                // Huffman compression
                match algorithm {
                    CompressionAlgorithm::Huffman => {
                        self.decompress_huffman(&block.compressed_data)
                    }
                    _ => self.decompress_delta(&block.compressed_data),
                }
            }
            3 => {
                // LZ77 compression
                self.decompress_lz77(&block.compressed_data)
            }
            _ => Ok(block.compressed_data.clone()),
        }
    }

    fn parse_indptr_data(&self, data: &[u8]) -> SparseResult<Vec<usize>> {
        if data.len() < 8 {
            return Ok(Vec::new());
        }

        let length = usize::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let mut indptr = Vec::with_capacity(length);

        let mut offset = 8;
        for _ in 0..length {
            if offset + 8 <= data.len() {
                let value = usize::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                indptr.push(value);
                offset += 8;
            }
        }

        Ok(indptr)
    }

    fn parse_indices_data(&self, data: &[u8]) -> SparseResult<Vec<usize>> {
        self.parse_indptr_data(data) // Same format
    }

    fn parse_data_values<T>(&self, data: &[u8]) -> SparseResult<Vec<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + num_traits::FromPrimitive,
    {
        if data.len() < 8 {
            return Ok(Vec::new());
        }

        let length = usize::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let mut values = Vec::with_capacity(length);

        let mut offset = 8;
        for _ in 0..length {
            if offset + 8 <= data.len() {
                let value_f64 = f64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                if let Some(value) = T::from_f64(value_f64) {
                    values.push(value);
                }
                offset += 8;
            }
        }

        Ok(values)
    }

    fn parse_combined_data<T>(&self, data: &[u8]) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<T>)>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Placeholder for combined _data parsing
        Ok((Vec::new(), Vec::new(), Vec::new()))
    }

    fn create_uncompressed_matrix<T>(
        &self,
        matrixid: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<CompressedMatrix<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let blocks = self.compress_with_none(matrixid, rows, indptr, indices, data)?;
        let total_size = std::mem::size_of_val(indptr)
            + std::mem::size_of_val(indices)
            + std::mem::size_of_val(data);

        Ok(CompressedMatrix {
            matrixid,
            original_rows: rows,
            original_cols: if !indptr.is_empty() {
                *indices.iter().max().unwrap_or(&0) + 1
            } else {
                0
            },
            compressed_blocks: blocks,
            compression_algorithm: CompressionAlgorithm::None,
            block_size: self.config.block_size,
            metadata: CompressionMetadata {
                original_size: total_size,
                compressed_size: total_size,
                compression_ratio: 1.0,
                compression_time: 0.0,
            },
            _phantom: PhantomData,
        })
    }

    fn update_compression_stats(
        &self,
        original_size: usize,
        blocks: &[CompressedBlock],
        compression_time: f64,
    ) {
        if let Ok(mut stats) = self.compression_stats.lock() {
            stats.total_blocks += blocks.len();
            stats.compressed_blocks += blocks.iter().filter(|b| b.compression_level > 0).count();
            stats.total_uncompressed_size += original_size;

            let compressed_size = blocks
                .iter()
                .map(|b| b.compressed_data.len())
                .sum::<usize>();
            stats.total_compressed_size += compressed_size;

            if stats.total_compressed_size > 0 {
                stats.compression_ratio =
                    stats.total_uncompressed_size as f64 / stats.total_compressed_size as f64;
            }

            stats.compression_time += compression_time;
        }
    }

    fn get_cache_stats(&self) -> CacheStats {
        if let Ok(cache) = self.block_cache.lock() {
            let total_accesses = cache.cache.values().map(|b| b.access_count).sum::<usize>();
            let hits = total_accesses; // Simplified
            let misses = 0; // Simplified

            CacheStats {
                hits,
                misses,
                hit_ratio: if total_accesses > 0 {
                    hits as f64 / total_accesses as f64
                } else {
                    0.0
                },
            }
        } else {
            CacheStats {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
            }
        }
    }
}

// Supporting structures and implementations

impl BlockCache {
    fn new(_maxsize: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            _maxsize,
            current_size: 0,
        }
    }
}

impl OutOfCoreManager {
    fn new(_tempdir: &str) -> SparseResult<Self> {
        std::fs::create_dir_all(_tempdir).map_err(SparseError::IoError)?;

        Ok(Self {
            temp_dir: _tempdir.to_string(),
            file_counter: AtomicUsize::new(0),
            active_files: HashMap::new(),
            memory_mapped_files: HashMap::new(),
        })
    }

    /// Write compressed block to disk
    #[allow(dead_code)]
    fn write_block_to_disk(&mut self, block: &CompressedBlock) -> SparseResult<String> {
        let file_id = self.file_counter.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("block_{}_{file_id}.dat", block.blockid);
        let filepath = Path::new(&self.temp_dir).join(&file_name);

        // Create and write to file
        let mut file = File::create(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to create file {filepath:?}: {e}")))?;

        // Write block header
        let header = BlockHeader {
            blockid: block.blockid.clone(),
            block_type: block.block_type as u8,
            original_size: block.original_size,
            compressed_size: block.compressed_data.len(),
            compression_level: block.compression_level,
        };

        file.write_all(&header.serialize())
            .map_err(|e| SparseError::Io(format!("Failed to write header: {e}")))?;

        // Write compressed data
        file.write_all(&block.compressed_data)
            .map_err(|e| SparseError::Io(format!("Failed to write data: {e}")))?;

        file.flush()
            .map_err(|e| SparseError::Io(format!("Failed to flush file: {e}")))?;

        // Track the file
        self.active_files
            .insert(block.blockid.clone(), file_name.clone());

        Ok(file_name)
    }

    /// Read compressed block from disk
    #[allow(dead_code)]
    fn read_block_from_disk(&self, blockid: BlockId) -> SparseResult<CompressedBlock> {
        let file_name = self.active_files.get(&blockid).ok_or_else(|| {
            SparseError::BlockNotFound(format!("Block {blockid} not found on disk"))
        })?;

        let filepath = Path::new(&self.temp_dir).join(file_name);
        let mut file = File::open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to open file {filepath:?}: {e}")))?;

        // Read block header
        let mut header_bytes = vec![0u8; std::mem::size_of::<BlockHeaderSerialized>()];
        file.read_exact(&mut header_bytes)
            .map_err(|e| SparseError::Io(format!("Failed to read header: {e}")))?;

        let header = BlockHeader::deserialize(&header_bytes)?;

        // Read compressed data
        let mut compressed_data = vec![0u8; header.compressed_size];
        file.read_exact(&mut compressed_data)
            .map_err(|e| SparseError::Io(format!("Failed to read data: {e}")))?;

        Ok(CompressedBlock {
            blockid: header.blockid,
            block_type: match header.block_type {
                0 => BlockType::IndPtr,
                1 => BlockType::Indices,
                2 => BlockType::Data,
                3 => BlockType::Combined,
                _ => {
                    return Err(SparseError::InvalidFormat(format!(
                        "Unknown block type: {}",
                        header.block_type
                    )))
                }
            },
            compressed_data,
            original_size: header.original_size,
            compression_level: header.compression_level,
        })
    }

    /// Create memory-mapped file for large blocks
    #[allow(dead_code)]
    fn create_memory_mapped_file(&mut self, blockid: BlockId, size: usize) -> SparseResult<String> {
        let file_id = self.file_counter.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("mmap_{blockid}_{file_id}.dat");
        let filepath = Path::new(&self.temp_dir).join(&file_name);

        let mapped_file = MemoryMappedFile::new(filepath, size)?;
        self.memory_mapped_files
            .insert(file_name.clone(), mapped_file);
        self.active_files.insert(blockid, file_name.clone());

        Ok(file_name)
    }

    /// Write data to memory-mapped file
    #[allow(dead_code)]
    fn write_to_mapped_file(
        &self,
        file_name: &str,
        offset: usize,
        data: &[u8],
    ) -> SparseResult<()> {
        if let Some(mapped_file) = self.memory_mapped_files.get(file_name) {
            mapped_file.write_at(offset, data)?;
            mapped_file.flush()?;
            Ok(())
        } else {
            Err(SparseError::Io(format!(
                "Memory-mapped file {file_name} not found"
            )))
        }
    }

    /// Read data from memory-mapped file
    #[allow(dead_code)]
    fn read_from_mapped_file(
        &self,
        file_name: &str,
        offset: usize,
        size: usize,
    ) -> SparseResult<Vec<u8>> {
        if let Some(mapped_file) = self.memory_mapped_files.get(file_name) {
            let mut buffer = vec![0u8; size];
            let bytes_read = mapped_file.read_at(offset, &mut buffer)?;
            buffer.truncate(bytes_read);
            Ok(buffer)
        } else {
            Err(SparseError::Io(format!(
                "Memory-mapped file {file_name} not found"
            )))
        }
    }

    /// Remove block from disk
    #[allow(dead_code)]
    fn remove_block(&mut self, blockid: BlockId) -> SparseResult<()> {
        if let Some(file_name) = self.active_files.remove(&blockid) {
            let filepath = Path::new(&self.temp_dir).join(&file_name);

            // Remove from memory-mapped files if it exists
            self.memory_mapped_files.remove(&file_name);

            // Remove file from disk
            std::fs::remove_file(&filepath)
                .map_err(|e| SparseError::Io(format!("Failed to remove file {filepath:?}: {e}")))?;
        }
        Ok(())
    }

    /// Get total disk usage
    #[allow(dead_code)]
    fn get_disk_usage(&self) -> SparseResult<usize> {
        let mut total_size = 0;
        for file_name in self.active_files.values() {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            if let Ok(metadata) = std::fs::metadata(&filepath) {
                total_size += metadata.len() as usize;
            }
        }
        Ok(total_size)
    }

    /// Cleanup all temporary files
    #[allow(dead_code)]
    fn cleanup(&mut self) -> SparseResult<()> {
        for file_name in self.active_files.values() {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            let _ = std::fs::remove_file(&filepath); // Ignore errors during cleanup
        }
        self.active_files.clear();
        self.memory_mapped_files.clear();
        Ok(())
    }
}

/// Block header for disk storage
#[derive(Debug)]
#[allow(dead_code)]
struct BlockHeader {
    blockid: BlockId,
    block_type: u8,
    original_size: usize,
    compressed_size: usize,
    compression_level: u8,
}

/// Serialized block header (fixed size for disk storage)
#[repr(C)]
#[allow(dead_code)]
struct BlockHeaderSerialized {
    blockid: u64,
    block_type: u8,
    original_size: u64,
    compressed_size: u64,
    compression_level: u8,
    padding: [u8; 6], // Ensure 8-byte alignment
}

impl BlockHeader {
    #[allow(dead_code)]
    fn serialize(&self) -> Vec<u8> {
        let serialized = BlockHeaderSerialized {
            blockid: self.blockid.to_u64(),
            block_type: self.block_type,
            original_size: self.original_size as u64,
            compressed_size: self.compressed_size as u64,
            compression_level: self.compression_level,
            padding: [0; 6],
        };

        // Convert to bytes
        unsafe {
            let ptr = &serialized as *const BlockHeaderSerialized as *const u8;
            std::slice::from_raw_parts(ptr, std::mem::size_of::<BlockHeaderSerialized>()).to_vec()
        }
    }

    #[allow(dead_code)]
    fn deserialize(data: &[u8]) -> SparseResult<Self> {
        if data.len() < std::mem::size_of::<BlockHeaderSerialized>() {
            return Err(SparseError::Io("Invalid header size".to_string()));
        }

        // Convert from bytes
        let serialized: BlockHeaderSerialized = unsafe {
            let ptr = data.as_ptr() as *const BlockHeaderSerialized;
            ptr.read()
        };

        Ok(BlockHeader {
            blockid: BlockId::from_u64(serialized.blockid),
            block_type: serialized.block_type,
            original_size: serialized.original_size as usize,
            compressed_size: serialized.compressed_size as usize,
            compression_level: serialized.compression_level,
        })
    }
}

// Data structures

/// Compressed sparse matrix representation
#[derive(Debug)]
pub struct CompressedMatrix<T> {
    pub matrixid: u64,
    pub original_rows: usize,
    pub original_cols: usize,
    pub compressed_blocks: Vec<CompressedBlock>,
    pub compression_algorithm: CompressionAlgorithm,
    pub block_size: usize,
    pub metadata: CompressionMetadata,
    _phantom: PhantomData<T>,
}

/// Compressed block of matrix data
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    pub blockid: BlockId,
    pub block_type: BlockType,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub compression_level: u8,
}

/// Type of data stored in a block
#[derive(Debug, Clone, Copy)]
pub enum BlockType {
    IndPtr,
    Indices,
    Data,
    Combined,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
}

/// Compression strategy
#[derive(Debug)]
#[allow(dead_code)]
struct CompressionStrategy {
    algorithm: CompressionAlgorithm,
    block_size: usize,
    hierarchical: bool,
    predicted_ratio: f64,
}

/// Sparsity pattern analysis
#[derive(Debug, Default)]
struct SparsityPatternAnalysis {
    avg_nnz_per_row: f64,
    max_nnz_per_row: usize,
    min_nnz_per_row: usize,
    sequential_patterns: usize,
    clustering_factor: f64,
    bandwidth: usize,
}

/// Access pattern information
#[derive(Debug, Default)]
struct AccessPatternInfo {
    total_accesses: usize,
    avg_temporal_locality: f64,
    avg_spatial_locality: f64,
    pattern_count: usize,
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryStats {
    pub total_memory_budget: usize,
    pub current_memory_usage: usize,
    pub memory_usage_ratio: f64,
    pub compression_stats: CompressionStats,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_ratio: f64,
    pub out_of_core_enabled: bool,
}

/// Cache statistics
#[derive(Debug)]
struct CacheStats {
    hits: usize,
    misses: usize,
    hit_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_compressor_creation() {
        let config = AdaptiveCompressionConfig::default();
        let compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        assert_eq!(compressor.memory_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_matrix_compression() {
        let config = AdaptiveCompressionConfig {
            compression_algorithm: CompressionAlgorithm::None,
            ..Default::default()
        };
        let mut compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];

        let compressed = compressor
            .compress_matrix(1, 2, &indptr, &indices, &data)
            .unwrap();

        assert_eq!(compressed.matrixid, 1);
        assert_eq!(compressed.original_rows, 2);
        assert_eq!(compressed.compressed_blocks.len(), 3);
    }

    #[test]
    fn test_compression_decompression_roundtrip() {
        let config = AdaptiveCompressionConfig {
            compression_algorithm: CompressionAlgorithm::None,
            ..Default::default()
        };
        let mut compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        let original_indptr = vec![0, 2, 4, 5];
        let original_indices = vec![0, 1, 1, 2, 0];
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let compressed = compressor
            .compress_matrix(1, 3, &original_indptr, &original_indices, &original_data)
            .unwrap();
        let (decompressed_indptr, decompressed_indices, decompressed_data) =
            compressor.decompress_matrix(&compressed).unwrap();

        assert_eq!(decompressed_indptr, original_indptr);
        assert_eq!(decompressed_indices, original_indices);
        assert_eq!(decompressed_data.len(), original_data.len());
    }

    #[test]
    fn test_memory_stats() {
        let config = AdaptiveCompressionConfig::default();
        let compressor = AdaptiveMemoryCompressor::new(config).unwrap();
        let stats = compressor.get_memory_stats();

        assert_eq!(stats.current_memory_usage, 0);
        assert!(stats.memory_usage_ratio >= 0.0);
        assert!(stats.cache_hit_ratio >= 0.0);
    }
}
