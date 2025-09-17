//! Comprehensive GPU memory management system
//!
//! This module provides a complete GPU memory management solution including:
//! - Advanced allocation strategies (buddy, slab, arena allocators)
//! - Intelligent memory management (GC, prefetching, eviction, defragmentation)
//! - Multi-vendor GPU support (NVIDIA CUDA, AMD ROCm, Intel OneAPI, Apple Metal)
//!
//! The system is designed to provide optimal memory utilization and performance
//! across different GPU architectures and workloads.

pub mod allocation;
pub mod management;
pub mod vendors;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::ffi::c_void;

// Re-export key types from submodules
pub use allocation::{
    AllocationEngine, AllocatorType, AllocationConfig, AllocationStats,
    BuddyAllocator, SlabAllocator, ArenaAllocator, AllocationStrategy,
};

pub use management::{
    IntegratedMemoryManager, MemoryManagementConfig, ManagementStats,
    GarbageCollectionEngine, PrefetchingEngine, EvictionEngine, DefragmentationEngine,
    AccessType, MemoryManagementError,
};

pub use vendors::{
    UnifiedGpuBackend, GpuVendor, VendorConfig, UnifiedGpuError,
    GpuBackendFactory, UnifiedMemoryStats,
    CudaMemoryBackend, CudaConfig, CudaError, CudaMemoryType,
    RocmMemoryBackend, RocmConfig, RocmError, RocmMemoryType,
    OneApiMemoryBackend, OneApiConfig, OneApiError, OneApiMemoryType,
    MetalMemoryBackend, MetalConfig, MetalError, MetalMemoryType,
};

/// Comprehensive GPU memory system configuration
#[derive(Debug, Clone)]
pub struct GpuMemorySystemConfig {
    /// Vendor-specific backend configuration
    pub vendor_config: VendorConfig,
    /// Memory allocation configuration
    pub allocation_config: AllocationConfig,
    /// Memory management configuration
    pub management_config: MemoryManagementConfig,
    /// System-wide configuration
    pub system_config: SystemConfig,
}

/// System-wide configuration
#[derive(Debug, Clone)]
pub struct SystemConfig {
    /// Enable unified memory interface
    pub enable_unified_interface: bool,
    /// Enable cross-vendor memory sharing
    pub enable_cross_vendor_sharing: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Memory budget as fraction of total GPU memory
    pub memory_budget: f64,
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Enable memory compression
    pub enable_memory_compression: bool,
    /// Thread pool size for memory operations
    pub thread_pool_size: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            enable_unified_interface: true,
            enable_cross_vendor_sharing: false,
            enable_performance_monitoring: true,
            monitoring_interval: Duration::from_millis(500),
            memory_budget: 0.9,
            enable_auto_optimization: true,
            optimization_interval: Duration::from_secs(60),
            enable_memory_compression: false,
            thread_pool_size: 4,
        }
    }
}

impl Default for GpuMemorySystemConfig {
    fn default() -> Self {
        let vendor = GpuBackendFactory::get_preferred_vendor();
        Self {
            vendor_config: GpuBackendFactory::create_default_config(vendor),
            allocation_config: AllocationConfig::default(),
            management_config: MemoryManagementConfig::default(),
            system_config: SystemConfig::default(),
        }
    }
}

/// Unified GPU memory system
pub struct GpuMemorySystem {
    /// GPU backend
    gpu_backend: UnifiedGpuBackend,
    /// Allocation engine
    allocation_engine: AllocationEngine,
    /// Memory management system
    memory_manager: IntegratedMemoryManager,
    /// System configuration
    config: GpuMemorySystemConfig,
    /// System statistics
    stats: SystemStats,
    /// Memory regions tracking
    memory_regions: HashMap<*mut c_void, MemoryAllocation>,
    /// Background monitoring enabled
    monitoring_enabled: bool,
    /// Last optimization time
    last_optimization: Instant,
}

/// Memory allocation tracking
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub ptr: *mut c_void,
    pub size: usize,
    pub allocator_type: AllocatorType,
    pub vendor_memory_type: String,
    pub allocated_at: Instant,
    pub last_accessed: Option<Instant>,
    pub access_count: u64,
    pub ref_count: u32,
}

/// System-wide statistics
#[derive(Debug, Clone, Default)]
pub struct SystemStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub active_allocations: u64,
    pub peak_memory_usage: usize,
    pub fragmentation_ratio: f64,
    pub allocation_efficiency: f64,
    pub vendor_stats: UnifiedMemoryStats,
    pub allocation_stats: AllocationStats,
    pub management_stats: ManagementStats,
    pub uptime: Duration,
    pub optimization_cycles: u64,
}

impl GpuMemorySystem {
    /// Create new GPU memory system
    pub fn new(config: GpuMemorySystemConfig) -> Result<Self, GpuMemorySystemError> {
        // Initialize GPU backend
        let gpu_backend = UnifiedGpuBackend::new(config.vendor_config.clone())?;
        
        // Initialize allocation engine
        let allocation_engine = AllocationEngine::new(config.allocation_config.clone());
        
        // Initialize memory manager
        let memory_manager = IntegratedMemoryManager::new(config.management_config.clone());
        
        Ok(Self {
            gpu_backend,
            allocation_engine,
            memory_manager,
            config,
            stats: SystemStats::default(),
            memory_regions: HashMap::new(),
            monitoring_enabled: false,
            last_optimization: Instant::now(),
        })
    }

    /// Create system with auto-detected best configuration
    pub fn auto_create() -> Result<Self, GpuMemorySystemError> {
        let config = GpuMemorySystemConfig::default();
        Self::new(config)
    }

    /// Start the GPU memory system
    pub fn start(&mut self) -> Result<(), GpuMemorySystemError> {
        // Start background memory management
        if self.config.system_config.enable_performance_monitoring {
            self.memory_manager.start_background_management()
                .map_err(|e| GpuMemorySystemError::ManagementError(format!("{}", e)))?;
            self.monitoring_enabled = true;
        }

        // Initialize allocation engine
        self.allocation_engine.initialize()?;

        Ok(())
    }

    /// Allocate GPU memory with unified interface
    pub fn allocate(&mut self, size: usize, alignment: Option<usize>) -> Result<*mut c_void, GpuMemorySystemError> {
        let start_time = Instant::now();
        
        // Choose optimal allocator based on size and usage patterns
        let allocator_type = self.choose_allocator(size);
        
        // Allocate using allocation engine
        let ptr = self.allocation_engine.allocate(size, allocator_type, alignment)?;
        
        // Create allocation record
        let allocation = MemoryAllocation {
            ptr,
            size,
            allocator_type,
            vendor_memory_type: self.get_vendor_memory_type(),
            allocated_at: Instant::now(),
            last_accessed: Some(Instant::now()),
            access_count: 1,
            ref_count: 1,
        };
        
        // Track allocation
        self.memory_regions.insert(ptr, allocation);
        
        // Update statistics
        self.update_allocation_stats(size, start_time.elapsed());
        
        // Check for memory pressure and handle if needed
        self.handle_memory_pressure()?;
        
        Ok(ptr)
    }

    /// Free GPU memory
    pub fn free(&mut self, ptr: *mut c_void) -> Result<(), GpuMemorySystemError> {
        let start_time = Instant::now();
        
        // Get allocation info
        let allocation = self.memory_regions.remove(&ptr)
            .ok_or_else(|| GpuMemorySystemError::InvalidPointer("Pointer not found".to_string()))?;
        
        // Free using appropriate allocator
        self.allocation_engine.free(ptr, allocation.allocator_type)?;
        
        // Update statistics
        self.update_deallocation_stats(allocation.size, start_time.elapsed());
        
        Ok(())
    }

    /// Reallocate memory with potential optimization
    pub fn reallocate(&mut self, ptr: *mut c_void, new_size: usize) -> Result<*mut c_void, GpuMemorySystemError> {
        // Get current allocation info
        let allocation = self.memory_regions.get(&ptr)
            .ok_or_else(|| GpuMemorySystemError::InvalidPointer("Pointer not found".to_string()))?;
        
        let old_size = allocation.size;
        let allocator_type = allocation.allocator_type;
        
        // Try in-place reallocation first
        if let Ok(new_ptr) = self.allocation_engine.reallocate(ptr, new_size, allocator_type) {
            if new_ptr == ptr {
                // In-place reallocation successful
                if let Some(allocation) = self.memory_regions.get_mut(&ptr) {
                    allocation.size = new_size;
                    allocation.last_accessed = Some(Instant::now());
                    allocation.access_count += 1;
                }
                return Ok(ptr);
            }
        }
        
        // Fallback to allocate + copy + free
        let new_ptr = self.allocate(new_size, None)?;
        
        // Copy data (simulate)
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, new_ptr as *mut u8, old_size.min(new_size));
        }
        
        // Free old memory
        self.free(ptr)?;
        
        Ok(new_ptr)
    }

    /// Record memory access for optimization
    pub fn record_access(&mut self, ptr: *mut c_void, access_type: AccessType) -> Result<(), GpuMemorySystemError> {
        if let Some(allocation) = self.memory_regions.get_mut(&ptr) {
            allocation.last_accessed = Some(Instant::now());
            allocation.access_count += 1;
            
            // Update memory manager with access pattern
            self.memory_manager.update_access_pattern(ptr, allocation.size, access_type)?;
        }
        
        Ok(())
    }

    /// Get memory information
    pub fn get_memory_info(&self, ptr: *mut c_void) -> Option<&MemoryAllocation> {
        self.memory_regions.get(&ptr)
    }

    /// Get system statistics
    pub fn get_stats(&mut self) -> SystemStats {
        // Update vendor stats
        self.stats.vendor_stats = self.gpu_backend.get_memory_stats();
        
        // Update allocation stats
        self.stats.allocation_stats = self.allocation_engine.get_stats();
        
        // Update management stats
        self.stats.management_stats = self.memory_manager.get_stats().clone();
        
        // Calculate derived metrics
        self.calculate_system_metrics();
        
        self.stats.clone()
    }

    /// Optimize memory system based on usage patterns
    pub fn optimize(&mut self) -> Result<(), GpuMemorySystemError> {
        if !self.config.system_config.enable_auto_optimization {
            return Ok(());
        }
        
        let now = Instant::now();
        if now.duration_since(self.last_optimization) < self.config.system_config.optimization_interval {
            return Ok(());
        }
        
        // Run garbage collection
        let memory_regions: HashMap<usize, management::MemoryRegion> = self.memory_regions
            .iter()
            .enumerate()
            .map(|(i, (ptr, alloc))| (
                i, 
                management::MemoryRegion {
                    id: i,
                    start_addr: *ptr as usize,
                    size: alloc.size,
                    allocated: true,
                    last_access: alloc.last_accessed,
                    access_count: alloc.access_count,
                    allocator_type: format!("{:?}", alloc.allocator_type),
                }
            ))
            .collect();
        
        let _ = self.memory_manager.run_garbage_collection(&memory_regions)?;
        
        // Optimize allocation strategies
        self.allocation_engine.optimize_strategies()?;
        
        // Optimize memory management policies
        self.memory_manager.optimize_policies()?;
        
        // Run defragmentation if needed
        if self.stats.fragmentation_ratio > 0.3 {
            let _ = self.memory_manager.defragment(&memory_regions)?;
        }
        
        self.last_optimization = now;
        self.stats.optimization_cycles += 1;
        
        Ok(())
    }

    /// Check and handle memory pressure
    fn handle_memory_pressure(&mut self) -> Result<(), GpuMemorySystemError> {
        let memory_usage_ratio = self.calculate_memory_usage_ratio();
        
        if memory_usage_ratio > self.config.system_config.memory_budget {
            let memory_regions: HashMap<usize, management::MemoryRegion> = self.memory_regions
                .iter()
                .enumerate()
                .map(|(i, (ptr, alloc))| (
                    i, 
                    management::MemoryRegion {
                        id: i,
                        start_addr: *ptr as usize,
                        size: alloc.size,
                        allocated: true,
                        last_access: alloc.last_accessed,
                        access_count: alloc.access_count,
                        allocator_type: format!("{:?}", alloc.allocator_type),
                    }
                ))
                .collect();
            
            self.memory_manager.handle_memory_pressure(memory_usage_ratio, &memory_regions)?;
        }
        
        Ok(())
    }

    /// Choose optimal allocator based on allocation size and patterns
    fn choose_allocator(&self, size: usize) -> AllocatorType {
        // Simple heuristics - could be enhanced with ML
        if size < 1024 {
            AllocatorType::Slab  // Small allocations
        } else if size < 1024 * 1024 {
            AllocatorType::Buddy  // Medium allocations
        } else {
            AllocatorType::Arena  // Large allocations
        }
    }

    /// Get vendor-specific memory type string
    fn get_vendor_memory_type(&self) -> String {
        match self.gpu_backend.get_vendor() {
            GpuVendor::Nvidia => "Device".to_string(),
            GpuVendor::Amd => "Device".to_string(), 
            GpuVendor::Intel => "Device".to_string(),
            GpuVendor::Apple => "Private".to_string(),
            GpuVendor::Unknown => "Unknown".to_string(),
        }
    }

    /// Update allocation statistics
    fn update_allocation_stats(&mut self, size: usize, duration: Duration) {
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += size as u64;
        self.stats.active_allocations += 1;
        
        if self.stats.bytes_allocated > self.stats.peak_memory_usage as u64 {
            self.stats.peak_memory_usage = self.stats.bytes_allocated as usize;
        }
    }

    /// Update deallocation statistics
    fn update_deallocation_stats(&mut self, size: usize, _duration: Duration) {
        self.stats.total_deallocations += 1;
        self.stats.bytes_deallocated += size as u64;
        self.stats.active_allocations = self.stats.active_allocations.saturating_sub(1);
    }

    /// Calculate memory usage ratio
    fn calculate_memory_usage_ratio(&self) -> f64 {
        let total_memory = self.get_total_gpu_memory();
        let used_memory = self.stats.bytes_allocated - self.stats.bytes_deallocated;
        used_memory as f64 / total_memory as f64
    }

    /// Get total GPU memory (vendor-specific)
    fn get_total_gpu_memory(&self) -> usize {
        // This would be implemented based on vendor-specific device queries
        match self.gpu_backend.get_vendor() {
            GpuVendor::Nvidia => 8 * 1024 * 1024 * 1024,  // 8GB typical
            GpuVendor::Amd => 16 * 1024 * 1024 * 1024,     // 16GB typical
            GpuVendor::Intel => 12 * 1024 * 1024 * 1024,   // 12GB typical
            GpuVendor::Apple => 32 * 1024 * 1024 * 1024,   // 32GB unified memory
            GpuVendor::Unknown => 4 * 1024 * 1024 * 1024,  // 4GB fallback
        }
    }

    /// Calculate system-wide metrics
    fn calculate_system_metrics(&mut self) {
        // Calculate fragmentation ratio
        let total_allocated = self.memory_regions.iter().map(|(_, alloc)| alloc.size).sum::<usize>();
        let total_managed = self.stats.vendor_stats.bytes_allocated;
        self.stats.fragmentation_ratio = if total_managed > 0 {
            1.0 - (total_allocated as f64 / total_managed as f64)
        } else {
            0.0
        };
        
        // Calculate allocation efficiency
        self.stats.allocation_efficiency = if self.stats.total_allocations > 0 {
            let successful_allocations = self.stats.total_allocations;
            successful_allocations as f64 / self.stats.total_allocations as f64
        } else {
            1.0
        };
    }
}

/// GPU memory system errors
#[derive(Debug)]
pub enum GpuMemorySystemError {
    BackendError(UnifiedGpuError),
    AllocationError(String),
    ManagementError(String),
    InvalidPointer(String),
    SystemNotStarted,
    ConfigurationError(String),
    OptimizationFailed(String),
    InternalError(String),
}

impl std::fmt::Display for GpuMemorySystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuMemorySystemError::BackendError(err) => write!(f, "Backend error: {}", err),
            GpuMemorySystemError::AllocationError(msg) => write!(f, "Allocation error: {}", msg),
            GpuMemorySystemError::ManagementError(msg) => write!(f, "Management error: {}", msg),
            GpuMemorySystemError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            GpuMemorySystemError::SystemNotStarted => write!(f, "System not started"),
            GpuMemorySystemError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            GpuMemorySystemError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            GpuMemorySystemError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for GpuMemorySystemError {}

impl From<UnifiedGpuError> for GpuMemorySystemError {
    fn from(err: UnifiedGpuError) -> Self {
        GpuMemorySystemError::BackendError(err)
    }
}

impl From<allocation::AllocationError> for GpuMemorySystemError {
    fn from(err: allocation::AllocationError) -> Self {
        GpuMemorySystemError::AllocationError(format!("{}", err))
    }
}

impl From<MemoryManagementError> for GpuMemorySystemError {
    fn from(err: MemoryManagementError) -> Self {
        GpuMemorySystemError::ManagementError(format!("{}", err))
    }
}

/// Thread-safe wrapper for GPU memory system
pub struct ThreadSafeGpuMemorySystem {
    system: Arc<Mutex<GpuMemorySystem>>,
}

impl ThreadSafeGpuMemorySystem {
    pub fn new(config: GpuMemorySystemConfig) -> Result<Self, GpuMemorySystemError> {
        let system = GpuMemorySystem::new(config)?;
        Ok(Self {
            system: Arc::new(Mutex::new(system)),
        })
    }

    pub fn allocate(&self, size: usize, alignment: Option<usize>) -> Result<*mut c_void, GpuMemorySystemError> {
        let mut system = self.system.lock().unwrap();
        system.allocate(size, alignment)
    }

    pub fn free(&self, ptr: *mut c_void) -> Result<(), GpuMemorySystemError> {
        let mut system = self.system.lock().unwrap();
        system.free(ptr)
    }

    pub fn get_stats(&self) -> SystemStats {
        let mut system = self.system.lock().unwrap();
        system.get_stats()
    }

    pub fn optimize(&self) -> Result<(), GpuMemorySystemError> {
        let mut system = self.system.lock().unwrap();
        system.optimize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let config = GpuMemorySystemConfig::default();
        let system = GpuMemorySystem::new(config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_auto_create() {
        let system = GpuMemorySystem::auto_create();
        assert!(system.is_ok());
    }

    #[test]
    fn test_thread_safe_wrapper() {
        let config = GpuMemorySystemConfig::default();
        let system = ThreadSafeGpuMemorySystem::new(config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_allocator_selection() {
        let config = GpuMemorySystemConfig::default();
        let system = GpuMemorySystem::new(config).unwrap();
        
        assert_eq!(system.choose_allocator(512), AllocatorType::Slab);
        assert_eq!(system.choose_allocator(64 * 1024), AllocatorType::Buddy);
        assert_eq!(system.choose_allocator(2 * 1024 * 1024), AllocatorType::Arena);
    }
}