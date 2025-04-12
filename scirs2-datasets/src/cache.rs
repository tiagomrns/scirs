//! Dataset caching functionality

use crate::error::{DatasetsError, Result};
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};
use std::cell::RefCell;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::PathBuf;

/// The base directory for caching datasets
const DEFAULT_CACHE_DIR: &str = ".scirs2_data";

/// Default cache size for in-memory caching
const DEFAULT_CACHE_SIZE: usize = 100;

/// Default TTL for in-memory cache (in seconds)
const DEFAULT_CACHE_TTL: u64 = 3600; // 1 hour

/// File path wrapper for hashing
#[derive(Clone, Debug, Eq, PartialEq)]
struct FileCacheKey(String);

impl Hash for FileCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Manages caching of downloaded datasets, using both file-based and in-memory caching
///
/// This implementation uses scirs2-core::cache's TTLSizedCache for in-memory caching,
/// while maintaining the file-based persistence for long-term storage.
pub struct DatasetCache {
    /// Directory for file-based caching
    cache_dir: PathBuf,
    /// In-memory cache for frequently accessed datasets
    mem_cache: RefCell<TTLSizedCache<FileCacheKey, Vec<u8>>>,
}

impl Default for DatasetCache {
    fn default() -> Self {
        let home_dir = dirs::home_dir().expect("Could not find home directory");
        let cache_dir = home_dir.join(DEFAULT_CACHE_DIR);

        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(DEFAULT_CACHE_SIZE)
                .with_ttl(DEFAULT_CACHE_TTL)
                .build_sized_cache(),
        );

        DatasetCache {
            cache_dir,
            mem_cache,
        }
    }
}

impl DatasetCache {
    /// Create a new dataset cache with the given cache directory and default memory cache
    pub fn new(cache_dir: PathBuf) -> Self {
        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(DEFAULT_CACHE_SIZE)
                .with_ttl(DEFAULT_CACHE_TTL)
                .build_sized_cache(),
        );

        DatasetCache {
            cache_dir,
            mem_cache,
        }
    }

    /// Create a new dataset cache with custom settings
    pub fn with_config(cache_dir: PathBuf, cache_size: usize, ttl_seconds: u64) -> Self {
        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(cache_size)
                .with_ttl(ttl_seconds)
                .build_sized_cache(),
        );

        DatasetCache {
            cache_dir,
            mem_cache,
        }
    }

    /// Create the cache directory if it doesn't exist
    pub fn ensure_cache_dir(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to create cache directory: {}", e))
            })?;
        }
        Ok(())
    }

    /// Get the path to a cached file
    pub fn get_cached_path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(name)
    }

    /// Check if a file is already cached (either in memory or on disk)
    pub fn is_cached(&self, name: &str) -> bool {
        // Check memory cache first
        let key = FileCacheKey(name.to_string());
        if self.mem_cache.borrow_mut().get(&key).is_some() {
            return true;
        }

        // Then check file system
        self.get_cached_path(name).exists()
    }

    /// Read a cached file as bytes
    ///
    /// This method checks the in-memory cache first, and falls back to the file system if needed.
    /// When reading from the file system, the result is also stored in the in-memory cache.
    pub fn read_cached(&self, name: &str) -> Result<Vec<u8>> {
        // Try memory cache first
        let key = FileCacheKey(name.to_string());
        if let Some(data) = self.mem_cache.borrow_mut().get(&key) {
            return Ok(data);
        }

        // Fall back to file system cache
        let path = self.get_cached_path(name);
        if !path.exists() {
            return Err(DatasetsError::CacheError(format!(
                "Cached file does not exist: {}",
                name
            )));
        }

        let mut file = File::open(path)
            .map_err(|e| DatasetsError::CacheError(format!("Failed to open cached file: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| DatasetsError::CacheError(format!("Failed to read cached file: {}", e)))?;

        // Update memory cache
        self.mem_cache.borrow_mut().insert(key, buffer.clone());

        Ok(buffer)
    }

    /// Write data to both the file cache and memory cache
    pub fn write_cached(&self, name: &str, data: &[u8]) -> Result<()> {
        self.ensure_cache_dir()?;

        // Write to file system cache
        let path = self.get_cached_path(name);
        let mut file = File::create(path).map_err(|e| {
            DatasetsError::CacheError(format!("Failed to create cache file: {}", e))
        })?;

        file.write_all(data).map_err(|e| {
            DatasetsError::CacheError(format!("Failed to write to cache file: {}", e))
        })?;

        // Update memory cache
        let key = FileCacheKey(name.to_string());
        self.mem_cache.borrow_mut().insert(key, data.to_vec());

        Ok(())
    }

    /// Clear the entire cache (both memory and file-based)
    pub fn clear_cache(&self) -> Result<()> {
        // Clear file system cache
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| DatasetsError::CacheError(format!("Failed to clear cache: {}", e)))?;
        }

        // Clear memory cache
        self.mem_cache.borrow_mut().clear();

        Ok(())
    }

    /// Remove a specific cached file (from both memory and file system)
    pub fn remove_cached(&self, name: &str) -> Result<()> {
        // Remove from file system
        let path = self.get_cached_path(name);
        if path.exists() {
            fs::remove_file(path).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to remove cached file: {}", e))
            })?;
        }

        // Remove from memory cache
        let key = FileCacheKey(name.to_string());
        self.mem_cache.borrow_mut().remove(&key);

        Ok(())
    }

    /// Compute a hash for a filename or URL
    pub fn hash_filename(name: &str) -> String {
        let hash = blake3::hash(name.as_bytes());
        hash.to_hex().to_string()
    }
}

/// Downloads data from a URL and returns it as bytes, using the cache when possible
#[cfg(feature = "download")]
pub fn download_data(url: &str, force_download: bool) -> Result<Vec<u8>> {
    let cache = DatasetCache::default();
    let cache_key = DatasetCache::hash_filename(url);

    // Check if the data is already cached
    if !force_download && cache.is_cached(&cache_key) {
        return cache.read_cached(&cache_key);
    }

    // Download the data
    let response = reqwest::blocking::get(url).map_err(|e| {
        DatasetsError::DownloadError(format!("Failed to download from {}: {}", url, e))
    })?;

    if !response.status().is_success() {
        return Err(DatasetsError::DownloadError(format!(
            "Failed to download from {}: HTTP status {}",
            url,
            response.status()
        )));
    }

    let data = response.bytes().map_err(|e| {
        DatasetsError::DownloadError(format!("Failed to read response data: {}", e))
    })?;

    let data_vec = data.to_vec();

    // Cache the data
    cache.write_cached(&cache_key, &data_vec)?;

    Ok(data_vec)
}

// Stub for when download feature is not enabled
#[cfg(not(feature = "download"))]
/// Downloads data from a URL or retrieves it from cache
///
/// This is a stub implementation when the download feature is not enabled.
/// It returns an error informing the user to enable the download feature.
///
/// # Arguments
///
/// * `_url` - The URL to download from
/// * `_force_download` - If true, force a new download instead of using cache
///
/// # Returns
///
/// * An error indicating that the download feature is not enabled
pub fn download_data(_url: &str, _force_download: bool) -> Result<Vec<u8>> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
    ))
}
