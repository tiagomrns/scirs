//! GPU backend implementations for different hardware platforms
//!
//! This module contains implementations for various GPU backends including
//! CUDA, OpenCL, ROCm, and others. Each backend provides a consistent
//! interface for GPU-accelerated linear algebra operations.

use super::{GpuBackend, GpuBuffer, GpuContext, GpuContextAlloc, GpuDeviceInfo, GpuDeviceType};
use crate::error::{LinalgError, LinalgResult};
use std::collections::HashMap;

/// Comprehensive CUDA backend with cuBLAS integration
#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use std::ptr;

    // CUDA runtime types and constants (would normally come from cuda-sys crate)
    type CudaResult = i32;
    type CudaDevice = i32;
    type CudaDeviceProperties = [u8; 352]; // Approximate size
    type CudaStream = *mut std::ffi::c_void;
    type CudaEvent = *mut std::ffi::c_void;
    type CublasHandle = *mut std::ffi::c_void;
    type CusolverDnHandle = *mut std::ffi::c_void;

    const CUDA_SUCCESS: CudaResult = 0;
    const CUDA_ERROR_NO_DEVICE: CudaResult = 38;

    // Mock CUDA functions (in real implementation, these would be extern "C" bindings)
    fn cuda_get_device_count() -> (CudaResult, i32) {
        // Mock implementation - would use cudart sys bindings
        (CUDA_SUCCESS, 0) // Return 0 devices for safety in this mock
    }

    fn cuda_get_device_properties(
        _props: &mut CudaDeviceProperties,
        device: CudaDevice,
    ) -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_set_device(device: CudaDevice) -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_device_synchronize() -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_free(ptr: *mut std::ffi::c_void) -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_memcpy(
        _dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        _count: usize,
        kind: i32,
    ) -> CudaResult {
        CUDA_SUCCESS
    }

    fn cuda_mem_get_info() -> (CudaResult, usize, usize) {
        (
            CUDA_SUCCESS,
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
        ) // Mock 8GB free, 16GB total
    }

    /// Comprehensive CUDA backend with advanced features
    pub struct CudaBackend {
        initialized: bool,
        devices: Vec<CudaDeviceInfo>,
        driver_version: u32,
        runtime_version: u32,
    }

    #[derive(Debug, Clone)]
    struct CudaDeviceInfo {
        device_id: i32,
        properties: CudaDeviceProperties,
        compute_capability: (i32, i32),
        max_threads_per_block: i32,
        max_block_dim: [i32; 3],
        max_grid_dim: [i32; 3],
        shared_memory_per_block: usize,
        total_constant_memory: usize,
        warpsize: i32,
        max_pitch: usize,
        max_registers_per_block: i32,
        clock_rate: i32,
        texture_alignment: usize,
        concurrent_kernels: bool,
        integrated: bool,
        can_map_host_memory: bool,
        compute_mode: i32,
        maxtexture_1d: i32,
        maxtexture_2d: [i32; 2],
        maxtexture_3d: [i32; 3],
        pci_bus_id: String,
        pci_device_id: String,
        unified_addressing: bool,
        memory_clock_rate: i32,
        memory_bus_width: i32,
        l2_cachesize: usize,
        max_threads_per_multiprocessor: i32,
        stream_priorities_supported: bool,
        global_l1_cache_supported: bool,
        local_l1_cache_supported: bool,
        managed_memory: bool,
        multi_gpu_board: bool,
        multi_gpu_board_group_id: i32,
    }

    impl CudaBackend {
        pub fn new() -> LinalgResult<Self> {
            // Initialize CUDA runtime
            let (result, device_count) = cuda_get_device_count();
            if result != CUDA_SUCCESS {
                if result == CUDA_ERROR_NO_DEVICE {
                    return Err(LinalgError::ComputationError(
                        "No CUDA-capable devices found".to_string(),
                    ));
                }
                return Err(LinalgError::ComputationError(format!(
                    "Failed to initialize CUDA runtime: error code {}",
                    result
                )));
            }

            let mut devices = Vec::with_capacity(device_count as usize);

            // Enumerate all CUDA devices
            for device_id in 0..device_count {
                let mut properties = [0u8; 352];
                let result = cuda_get_device_properties(&mut properties, device_id);
                if result != CUDA_SUCCESS {
                    return Err(LinalgError::ComputationError(format!(
                        "Failed to get device properties for device {}: error code {}",
                        device_id, result
                    )));
                }

                // Parse device properties (simplified for mock implementation)
                let device_info = CudaDeviceInfo {
                    device_id,
                    properties,
                    compute_capability: (7, 5), // Mock Turing architecture
                    max_threads_per_block: 1024,
                    max_block_dim: [1024, 1024, 64],
                    max_grid_dim: [2147483647, 65535, 65535],
                    shared_memory_per_block: 49152,
                    total_constant_memory: 65536,
                    warpsize: 32,
                    max_pitch: 2147483647,
                    max_registers_per_block: 65536,
                    clock_rate: 1590000, // 1.59 GHz
                    texture_alignment: 512,
                    concurrent_kernels: true,
                    integrated: false,
                    can_map_host_memory: true,
                    compute_mode: 0, // Default compute mode
                    maxtexture_1d: 131072,
                    maxtexture_2d: [131072, 65536],
                    maxtexture_3d: [16384, 16384, 16384],
                    pci_bus_id: format!("0000:{:02x}:00.0", device_id),
                    pci_device_id: format!("10de:1b80"), // Mock RTX 2080 Ti
                    unified_addressing: true,
                    memory_clock_rate: 7000000, // 7 GHz effective
                    memory_bus_width: 352,
                    l2_cachesize: 5767168, // 5.5 MB
                    max_threads_per_multiprocessor: 1024,
                    stream_priorities_supported: true,
                    global_l1_cache_supported: true,
                    local_l1_cache_supported: true,
                    managed_memory: true,
                    multi_gpu_board: false,
                    multi_gpu_board_group_id: 0,
                };

                devices.push(device_info);
            }

            Ok(Self {
                initialized: true,
                devices,
                driver_version: 47_057, // Mock driver version
                runtime_version: 114,   // Mock CUDA 11.4
            })
        }

        /// Get CUDA driver version
        pub fn driver_version(&self) -> u32 {
            self.driver_version
        }

        /// Get CUDA runtime version
        pub fn runtime_version(&self) -> u32 {
            self.runtime_version
        }

        /// Check if unified memory is supported
        pub fn supports_unified_memory(&self) -> bool {
            self.devices.iter().all(|d| d.managed_memory)
        }

        /// Get maximum compute capability across all devices
        pub fn max_compute_capability(&self) -> (i32, i32) {
            self.devices
                .iter()
                .map(|d| d.compute_capability)
                .max_by_key(|&(major, minor)| major * 10 + minor)
                .unwrap_or((0, 0))
        }
    }

    impl GpuBackend for CudaBackend {
        fn name(&self) -> &str {
            "CUDA"
        }

        fn is_available(&self) -> bool {
            self.initialized && !self.devices.is_empty()
        }

        fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
            if !self.initialized {
                return Err(LinalgError::ComputationError(
                    "CUDA backend not initialized".to_string(),
                ));
            }

            let devices = self
                .devices
                .iter()
                .map(|cuda_device| {
                    // Calculate memory bandwidth (simplified calculation)
                    let memory_bandwidth = (cuda_device.memory_clock_rate as f64
                        * 2.0
                        * cuda_device.memory_bus_width as f64)
                        / 8.0
                        / 1_000_000.0;

                    GpuDeviceInfo {
                        device_type: GpuDeviceType::Cuda,
                        name: format!(
                            "CUDA Device {} (Compute {}.{})",
                            cuda_device.device_id,
                            cuda_device.compute_capability.0,
                            cuda_device.compute_capability.1
                        ),
                        total_memory: 11 * 1024 * 1024 * 1024, // Mock 11GB VRAM
                        compute_units: 68,                     // Mock SM count for RTX 2080 Ti
                        clock_frequency: (cuda_device.clock_rate / 1000) as u32, // Convert to MHz
                        supports_fp64: cuda_device.compute_capability.0 >= 2, // Fermi and later
                        supports_fp16: cuda_device.compute_capability.0 >= 5
                            || (cuda_device.compute_capability.0 == 5
                                && cuda_device.compute_capability.1 >= 3), // Maxwell and later
                        max_work_groupsize: cuda_device.max_threads_per_block as usize,
                        memory_bandwidth,
                        l2_cachesize: cuda_device.l2_cachesize,
                        shared_memory_per_block: cuda_device.shared_memory_per_block,
                        registers_per_block: cuda_device.max_registers_per_block as u32,
                        warpsize: cuda_device.warpsize as u32,
                        max_threads_per_mp: cuda_device.max_threads_per_multiprocessor as u32,
                        multiprocessor_count: 68, // Mock SM count
                        supports_tensor_cores: cuda_device.compute_capability.0 >= 7, // Volta and later
                        supports_mixed_precision: cuda_device.compute_capability.0 >= 5, // Maxwell and later
                        vendor: "NVIDIA".to_string(),
                    }
                })
                .collect();

            Ok(devices)
        }

        fn create_context(&self, deviceid: usize) -> LinalgResult<Box<dyn GpuContext>> {
            if !self.initialized {
                return Err(LinalgError::ComputationError(
                    "CUDA backend not initialized".to_string(),
                ));
            }

            if device_id >= self.devices.len() {
                return Err(LinalgError::ComputationError(format!(
                    "Invalid device ID: {} (available devices: {})",
                    device_id,
                    self.devices.len()
                )));
            }

            let cuda_device = &self.devices[device_id];

            // Set CUDA device
            let result = cuda_set_device(cuda_device.device_id);
            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to set CUDA device {}: error code {}",
                    device_id, result
                )));
            }

            // Create CUDA context
            let context = CudaContext::new(cuda_device.clone())?;
            Ok(Box::new(context))
        }
    }

    /// Comprehensive CUDA context with cuBLAS and cuSOLVER integration
    #[derive(Debug)]
    pub struct CudaContext {
        device_info: CudaDeviceInfo,
        device_id: i32,
        cublas_handle: Option<CublasHandle>,
        cusolver_handle: Option<CusolverDnHandle>,
        streams: Vec<CudaStream>,
        events: Vec<CudaEvent>,
        memory_pool: CudaMemoryPool,
        performance_stats: CudaPerformanceStats,
    }

    // SAFETY: CUDA handles are thread-safe and can be shared across threads
    // These are opaque handles managed by the CUDA runtime
    unsafe impl Send for CudaContext {}
    unsafe impl Sync for CudaContext {}

    impl CudaContext {
        fn new(_deviceinfo: CudaDeviceInfo) -> LinalgResult<Self> {
            let device_id = device_info.device_id;

            // Initialize cuBLAS (mock)
            let cublas_handle = None; // Would create cuBLAS handle

            // Initialize cuSOLVER (mock)
            let cusolver_handle = None; // Would create cuSOLVER handle

            // Create default streams
            let streams = Vec::new(); // Would create CUDA streams

            // Create events for timing
            let events = Vec::new(); // Would create CUDA events

            // Initialize memory pool
            let memory_pool = CudaMemoryPool::new(device_id)?;

            // Initialize performance statistics
            let performance_stats = CudaPerformanceStats::new();

            Ok(Self {
                device_info,
                device_id,
                cublas_handle,
                cusolver_handle,
                streams,
                events,
                memory_pool,
                performance_stats,
            })
        }

        /// Get cuBLAS handle
        pub fn cublas_handle(&self) -> Option<CublasHandle> {
            self.cublas_handle
        }

        /// Get cuSOLVER handle
        pub fn cusolver_handle(&self) -> Option<CusolverDnHandle> {
            self.cusolver_handle
        }

        /// Create a new CUDA stream
        pub fn create_stream(&mut self) -> LinalgResult<CudaStream> {
            // Would create CUDA stream
            let stream = ptr::null_mut();
            self.streams.push(stream);
            Ok(stream)
        }

        /// Get performance statistics
        pub fn performance_stats(&self) -> &CudaPerformanceStats {
            &self.performance_stats
        }
    }

    impl GpuContext for CudaContext {
        fn device_info(&self) -> &GpuDeviceInfo {
            // Convert CudaDeviceInfo to GpuDeviceInfo
            static mut CACHED_INFO: Option<GpuDeviceInfo> = None;

            unsafe {
                if CACHED_INFO.is_none() {
                    let memory_bandwidth = (self.device_info.memory_clock_rate as f64
                        * 2.0
                        * self.device_info.memory_bus_width as f64)
                        / 8.0
                        / 1_000_000.0;

                    CACHED_INFO = Some(GpuDeviceInfo {
                        device_type: GpuDeviceType::Cuda,
                        name: format!(
                            "CUDA Device {} (Compute {}.{})",
                            self.device_info.device_id,
                            self.device_info.compute_capability.0,
                            self.device_info.compute_capability.1
                        ),
                        total_memory: 11 * 1024 * 1024 * 1024,
                        compute_units: 68,
                        clock_frequency: (self.device_info.clock_rate / 1000) as u32,
                        supports_fp64: self.device_info.compute_capability.0 >= 2,
                        supports_fp16: self.device_info.compute_capability.0 >= 5
                            || (self.device_info.compute_capability.0 == 5
                                && self.device_info.compute_capability.1 >= 3),
                        max_work_groupsize: self.device_info.max_threads_per_block as usize,
                        memory_bandwidth,
                        l2_cachesize: self.device_info.l2_cachesize,
                        shared_memory_per_block: self.device_info.shared_memory_per_block,
                        registers_per_block: self.device_info.max_registers_per_block as u32,
                        warpsize: self.device_info.warpsize as u32,
                        max_threads_per_mp: self.device_info.max_threads_per_multiprocessor as u32,
                        multiprocessor_count: 68,
                        supports_tensor_cores: self.device_info.compute_capability.0 >= 7,
                        supports_mixed_precision: self.device_info.compute_capability.0 >= 5,
                        vendor: "NVIDIA".to_string(),
                    });
                }

                CACHED_INFO.as_ref().unwrap()
            }
        }

        fn synchronize(&self) -> LinalgResult<()> {
            let result = cuda_device_synchronize();
            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "CUDA synchronization failed: error code {}",
                    result
                )));
            }
            Ok(())
        }

        fn available_memory(&self) -> LinalgResult<usize> {
            let (result, free_mem_total_mem) = cuda_mem_get_info();
            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to get memory info: error code {}",
                    result
                )));
            }
            Ok(free_mem)
        }
    }

    impl GpuContextAlloc for CudaContext {
        fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
            &self,
            size: usize,
        ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
            let buffer = CudaBuffer::new(size, self.device_id)?;
            Ok(Box::new(buffer))
        }
    }

    /// Advanced CUDA memory pool for efficient allocation
    #[derive(Debug)]
    struct CudaMemoryPool {
        device_id: i32,
        total_allocated: usize,
        peak_usage: usize,
        allocation_count: usize,
        free_blocks: HashMap<usize, Vec<*mut std::ffi::c_void>>,
    }

    impl CudaMemoryPool {
        fn new(_deviceid: i32) -> LinalgResult<Self> {
            Ok(Self {
                device_id,
                total_allocated: 0,
                peak_usage: 0,
                allocation_count: 0,
                free_blocks: HashMap::new(),
            })
        }

        fn allocate(&mut self, size: usize) -> LinalgResult<*mut std::ffi::c_void> {
            // Try to reuse existing block of same size
            if let Some(blocks) = self.free_blocks.get_mut(&size) {
                if let Some(ptr) = blocks.pop() {
                    return Ok(ptr);
                }
            }

            // Allocate new memory
            let mut ptr = ptr::null_mut();
            let result = cuda_malloc(&mut ptr, size);
            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "CUDA memory allocation failed: error code {}",
                    result
                )));
            }

            self.total_allocated += size;
            self.peak_usage = self.peak_usage.max(self.total_allocated);
            self.allocation_count += 1;

            Ok(ptr)
        }

        #[allow(dead_code)]
        fn deallocate(&mut self, ptr: *mut std::ffi::c_void, size: usize) {
            // Add to free blocks for reuse
            self.free_blocks.entry(size).or_default().push(ptr);
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }
    }

    /// CUDA buffer with advanced memory management
    #[derive(Debug)]
    struct CudaBuffer<T> {
        device_ptr: *mut std::ffi::c_void,
        size: usize,
        device_id: i32,
        is_pinned: bool,
        _phantom: std::marker::PhantomData<T>,
    }

    // SAFETY: CUDA device pointers are thread-safe and can be shared across threads
    // The CUDA runtime handles thread synchronization for device memory
    unsafe impl<T> Send for CudaBuffer<T> {}
    unsafe impl<T> Sync for CudaBuffer<T> {}

    impl<T: Clone + Send + Sync + Copy> CudaBuffer<T> {
        fn new(size: usize, deviceid: i32) -> LinalgResult<Self> {
            let bytesize = size * std::mem::size_of::<T>();
            let mut device_ptr = ptr::null_mut();

            let result = cuda_malloc(&mut device_ptr, bytesize);
            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to allocate CUDA buffer: error code {}",
                    result
                )));
            }

            Ok(Self {
                device_ptr,
                size,
                device_id,
                is_pinned: false,
                _phantom: std::marker::PhantomData,
            })
        }

        /// Enable pinned memory for faster transfers
        #[allow(dead_code)]
        pub fn enable_pinned_memory(&mut self) -> LinalgResult<()> {
            // Would configure pinned memory
            self.is_pinned = true;
            Ok(())
        }

        /// Get device pointer as typed pointer
        pub fn device_ptr_typed(&self) -> *mut T {
            self.device_ptr as *mut T
        }
    }

    impl<T: Clone + Send + Sync + Copy + std::fmt::Debug> GpuBuffer<T> for CudaBuffer<T> {
        fn len(&self) -> usize {
            self.size
        }

        fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            let bytesize = data.len() * std::mem::size_of::<T>();
            let result = cuda_memcpy(
                self.device_ptr,
                data.as_ptr() as *const std::ffi::c_void,
                bytesize,
                1, // cudaMemcpyHostToDevice
            );

            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "CUDA host-to-device copy failed: error code {}",
                    result
                )));
            }

            Ok(())
        }

        fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            let bytesize = data.len() * std::mem::size_of::<T>();
            let result = cuda_memcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.device_ptr,
                bytesize,
                2, // cudaMemcpyDeviceToHost
            );

            if result != CUDA_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "CUDA device-to-host copy failed: error code {}",
                    result
                )));
            }

            Ok(())
        }

        fn device_ptr(&self) -> *mut std::ffi::c_void {
            self.device_ptr
        }
    }

    impl<T> Drop for CudaBuffer<T> {
        fn drop(&mut self) {
            if !self.device_ptr.is_null() {
                let _ = cuda_free(self.device_ptr);
            }
        }
    }

    /// Performance statistics for CUDA operations
    #[derive(Debug, Clone)]
    pub struct CudaPerformanceStats {
        pub kernel_launches: usize,
        pub memory_transfers: usize,
        pub total_compute_time_ms: f64,
        pub total_transfer_time_ms: f64,
        pub peak_memory_usage: usize,
        pub average_occupancy: f64,
    }

    impl CudaPerformanceStats {
        fn new() -> Self {
            Self {
                kernel_launches: 0,
                memory_transfers: 0,
                total_compute_time_ms: 0.0,
                total_transfer_time_ms: 0.0,
                peak_memory_usage: 0,
                average_occupancy: 0.0,
            }
        }

        pub fn compute_efficiency(&self) -> f64 {
            if self.total_compute_time_ms + self.total_transfer_time_ms == 0.0 {
                return 0.0;
            }
            self.total_compute_time_ms / (self.total_compute_time_ms + self.total_transfer_time_ms)
        }
    }
}

/// Comprehensive OpenCL backend with clBLAS integration
#[cfg(feature = "opencl")]
pub mod opencl {
    use super::*;
    use std::ptr;
    use std::sync::Arc;

    // OpenCL types and constants (would normally come from opencl-sys crate)
    type ClInt = i32;
    type ClUInt = u32;
    type ClULong = u64;
    type ClBool = u32;

    // Thread-safe wrapper for OpenCL raw pointers
    #[derive(Debug, Clone, Copy)]
    struct SafeClPtr(*mut std::ffi::c_void);

    // SAFETY: In a real implementation, OpenCL handles are thread-safe
    // These are mock implementations for testing purposes
    unsafe impl Send for SafeClPtr {}
    unsafe impl Sync for SafeClPtr {}

    impl SafeClPtr {
        fn new(ptr: *mut std::ffi::c_void) -> Self {
            Self(_ptr)
        }

        fn as_ptr(self) -> *mut std::ffi::c_void {
            self.0
        }
    }

    type ClPlatformId = SafeClPtr;
    type ClDeviceId = SafeClPtr;
    type ClContext = SafeClPtr;
    type ClCommandQueue = SafeClPtr;
    type ClProgram = SafeClPtr;
    type ClKernel = SafeClPtr;
    type ClMem = SafeClPtr;
    type ClEvent = SafeClPtr;

    const CL_SUCCESS: ClInt = 0;
    const CL_DEVICE_NOT_FOUND: ClInt = -1;
    const CL_DEVICE_TYPE_GPU: ClULong = 1 << 2;
    const CL_DEVICE_TYPE_CPU: ClULong = 1 << 1;
    const CL_DEVICE_TYPE_ALL: ClULong = 0xFFFFFFFF;
    const CL_MEM_READ_WRITE: ClULong = 1 << 0;
    const CL_MEM_COPY_HOST_PTR: ClULong = 1 << 2;

    // Mock OpenCL functions
    fn cl_get_platform_ids() -> (ClInt, Vec<ClPlatformId>) {
        (CL_SUCCESS, vec![]) // Mock empty platforms for safety
    }

    fn cl_get_device_ids(
        _platform: ClPlatformId_device,
        r#type: ClULong,
    ) -> (ClInt, Vec<ClDeviceId>) {
        (CL_SUCCESS, vec![])
    }

    fn cl_get_device_info(device: ClDeviceId_param, name: ClUInt) -> (ClInt, Vec<u8>) {
        (CL_SUCCESS, vec![0; 256])
    }

    fn cl_get_platform_info(platform: ClPlatformId_param, name: ClUInt) -> (ClInt, String) {
        (CL_SUCCESS, "Mock Platform".to_string())
    }

    fn cl_create_context(devices: &[ClDeviceId]) -> (ClInt, ClContext) {
        (CL_SUCCESS, SafeClPtr::new(ptr::null_mut()))
    }

    fn cl_create_command_queue(context: ClContext, device: ClDeviceId) -> (ClInt, ClCommandQueue) {
        (CL_SUCCESS, SafeClPtr::new(ptr::null_mut()))
    }

    fn cl_create_buffer(_context: ClContext, flags: ClULong, size: usize) -> (ClInt, ClMem) {
        (CL_SUCCESS, SafeClPtr::new(ptr::null_mut()))
    }

    fn cl_enqueue_write_buffer(
        _queue: ClCommandQueue,
        buffer: ClMem,
        _blocking: ClBool,
        offset: usize,
        size: usize,
        ptr: *const std::ffi::c_void,
    ) -> ClInt {
        CL_SUCCESS
    }

    fn cl_enqueue_read_buffer(
        _queue: ClCommandQueue,
        buffer: ClMem,
        _blocking: ClBool,
        offset: usize,
        size: usize,
        ptr: *mut std::ffi::c_void,
    ) -> ClInt {
        CL_SUCCESS
    }

    fn cl_finish(queue: ClCommandQueue) -> ClInt {
        CL_SUCCESS
    }

    fn cl_release_mem_object(memobj: ClMem) -> ClInt {
        CL_SUCCESS
    }

    /// Comprehensive OpenCL backend with cross-platform GPU support
    pub struct OpenClBackend {
        platforms: Vec<OpenClPlatform>,
        devices: Vec<OpenClDeviceInfo>,
        context_cache: HashMap<usize, Arc<OpenClContextData>>,
        opencl_version: String,
        extensions: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OpenClPlatform {
        platform_id: ClPlatformId,
        name: String,
        vendor: String,
        version: String,
        profile: String,
        extensions: Vec<String>,
        devices: Vec<usize>, // Indices into global device list
    }

    #[derive(Debug, Clone)]
    struct OpenClDeviceInfo {
        device_id: ClDeviceId,
        platform_index: usize,
        device_type: ClULong,
        name: String,
        vendor: String,
        driver_version: String,
        device_version: String,
        opencl_c_version: String,
        max_compute_units: ClUInt,
        max_work_groupsize: usize,
        max_work_item_dimensions: ClUInt,
        max_work_itemsizes: Vec<usize>,
        preferred_vector_width_char: ClUInt,
        preferred_vector_width_short: ClUInt,
        preferred_vector_width_int: ClUInt,
        preferred_vector_width_long: ClUInt,
        preferred_vector_width_float: ClUInt,
        preferred_vector_width_double: ClUInt,
        max_clock_frequency: ClUInt,
        address_bits: ClUInt,
        max_mem_allocsize: ClULong,
        image_support: ClBool,
        max_read_image_args: ClUInt,
        max_write_image_args: ClUInt,
        image2d_max_width: usize,
        image2d_max_height: usize,
        image3d_max_width: usize,
        image3d_max_height: usize,
        image3d_max_depth: usize,
        max_samplers: ClUInt,
        max_parametersize: usize,
        mem_base_addr_align: ClUInt,
        min_data_type_alignsize: ClUInt,
        single_fp_config: ClULong,
        global_mem_cache_type: ClUInt,
        global_mem_cachelinesize: ClUInt,
        global_mem_cachesize: ClULong,
        global_memsize: ClULong,
        max_constant_buffersize: ClULong,
        max_constant_args: ClUInt,
        local_mem_type: ClUInt,
        local_memsize: ClULong,
        error_correction_support: ClBool,
        profiling_timer_resolution: usize,
        endian_little: ClBool,
        available: ClBool,
        compiler_available: ClBool,
        execution_capabilities: ClULong,
        queue_properties: ClULong,
        platform_id: ClPlatformId,
    }

    #[derive(Debug)]
    struct OpenClContextData {
        context: ClContext,
        device_id: ClDeviceId,
        command_queue: ClCommandQueue,
        device_info: OpenClDeviceInfo,
        kernel_cache: std::collections::HashMap<String, ClKernel>,
        program_cache: std::collections::HashMap<String, ClProgram>,
    }

    impl OpenClBackend {
        pub fn new() -> LinalgResult<Self> {
            // Get available OpenCL platforms
            let (result, platform_ids) = cl_get_platform_ids();
            if result != CL_SUCCESS {
                if result == CL_DEVICE_NOT_FOUND {
                    return Err(LinalgError::ComputationError(
                        "No OpenCL platforms found".to_string(),
                    ));
                }
                return Err(LinalgError::ComputationError(format!(
                    "Failed to get OpenCL platforms: error code {}",
                    result
                )));
            }

            let mut platforms = Vec::new();
            let mut all_devices = Vec::new();

            // Enumerate all platforms and their devices
            for (platform_idx, &platform_id) in platform_ids.iter().enumerate() {
                // Get platform information
                let (result, platform_name) = cl_get_platform_info(platform_id, 0x0902); // CL_PLATFORM_NAME
                if result != CL_SUCCESS {
                    continue;
                }

                let (result, platform_vendor) = cl_get_platform_info(platform_id, 0x0903); // CL_PLATFORM_VENDOR
                if result != CL_SUCCESS {
                    continue;
                }

                let (result, platform_version) = cl_get_platform_info(platform_id, 0x0901); // CL_PLATFORM_VERSION
                if result != CL_SUCCESS {
                    continue;
                }

                // Get devices for this platform
                let (result, device_ids) = cl_get_device_ids(platform_id, CL_DEVICE_TYPE_ALL);
                if result != CL_SUCCESS {
                    continue;
                }

                let mut platform_device_indices = Vec::new();

                for device_id in device_ids {
                    let device_info = Self::get_device_info(device_id, platform_idx, platform_id)?;
                    platform_device_indices.push(all_devices.len());
                    all_devices.push(device_info);
                }

                platforms.push(OpenClPlatform {
                    platform_id,
                    name: platform_name,
                    vendor: platform_vendor,
                    version: platform_version,
                    profile: "FULL_PROFILE".to_string(), // Mock
                    extensions: vec!["cl_khr_fp64".to_string(), "cl_khr_fp16".to_string()], // Mock common extensions
                    devices: platform_device_indices,
                });
            }

            Ok(Self {
                platforms,
                devices: all_devices,
                context_cache: HashMap::new(),
                opencl_version: "OpenCL 2.1".to_string(), // Mock version
                extensions: vec![
                    "cl_khr_fp64".to_string(),
                    "cl_khr_fp16".to_string(),
                    "cl_khr_global_int32_base_atomics".to_string(),
                    "cl_khr_global_int32_extended_atomics".to_string(),
                ],
            })
        }

        fn get_device_info(
            device_id: ClDeviceId,
            platform_index: usize,
            platform_id: ClPlatformId,
        ) -> LinalgResult<OpenClDeviceInfo> {
            // Mock device information (in real implementation, query actual device properties)
            Ok(OpenClDeviceInfo {
                device_id,
                platform_index,
                device_type: CL_DEVICE_TYPE_GPU,
                name: "Mock OpenCL GPU Device".to_string(),
                vendor: "Mock Vendor".to_string(),
                driver_version: "1.0.0".to_string(),
                device_version: "OpenCL 2.1".to_string(),
                opencl_c_version: "OpenCL C 2.0".to_string(),
                max_compute_units: 32,
                max_work_groupsize: 1024,
                max_work_item_dimensions: 3,
                max_work_itemsizes: vec![1024, 1024, 64],
                preferred_vector_width_char: 16,
                preferred_vector_width_short: 8,
                preferred_vector_width_int: 4,
                preferred_vector_width_long: 2,
                preferred_vector_width_float: 4,
                preferred_vector_width_double: 2,
                max_clock_frequency: 1500,
                address_bits: 64,
                max_mem_allocsize: 2 * 1024 * 1024 * 1024, // 2GB
                image_support: 1,
                max_read_image_args: 128,
                max_write_image_args: 64,
                image2d_max_width: 16384,
                image2d_max_height: 16384,
                image3d_max_width: 2048,
                image3d_max_height: 2048,
                image3d_max_depth: 2048,
                max_samplers: 16,
                max_parametersize: 1024,
                mem_base_addr_align: 1024,
                min_data_type_alignsize: 128,
                single_fp_config: 0x3F,   // Mock FP config
                global_mem_cache_type: 2, // CL_READ_WRITE_CACHE
                global_mem_cachelinesize: 64,
                global_mem_cachesize: 2 * 1024 * 1024,  // 2MB
                global_memsize: 8 * 1024 * 1024 * 1024, // 8GB
                max_constant_buffersize: 64 * 1024,     // 64KB
                max_constant_args: 8,
                local_mem_type: 1,        // CL_LOCAL
                local_memsize: 48 * 1024, // 48KB
                error_correction_support: 0,
                profiling_timer_resolution: 1,
                endian_little: 1,
                available: 1,
                compiler_available: 1,
                execution_capabilities: 1, // CL_EXEC_KERNEL
                queue_properties: 2,       // CL_QUEUE_PROFILING_ENABLE
                platform_id,
            })
        }

        /// Get all available platforms
        pub fn platforms(&self) -> &[OpenClPlatform] {
            &self.platforms
        }

        /// Get platform by index
        pub fn platform(&self, index: usize) -> Option<&OpenClPlatform> {
            self.platforms.get(index)
        }

        /// Check if double precision is supported
        pub fn supports_double_precision(&self) -> bool {
            self.extensions.contains(&"cl_khr_fp64".to_string())
        }

        /// Check if half precision is supported
        pub fn supports_half_precision(&self) -> bool {
            self.extensions.contains(&"cl_khr_fp16".to_string())
        }

        /// Get OpenCL version
        pub fn opencl_version(&self) -> &str {
            &self.opencl_version
        }
    }

    impl GpuBackend for OpenClBackend {
        fn name(&self) -> &str {
            "OpenCL"
        }

        fn is_available(&self) -> bool {
            !self.platforms.is_empty() && !self.devices.is_empty()
        }

        fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
            let devices = self
                .devices
                .iter()
                .map(|opencl_device| {
                    // Calculate memory bandwidth (estimated)
                    let memory_bandwidth =
                        (opencl_device.max_clock_frequency as f64 * 256.0) / 1000.0; // Rough estimate

                    let device_type = match opencl_device.device_type {
                        CL_DEVICE_TYPE_GPU => GpuDeviceType::OpenCl,
                        CL_DEVICE_TYPE_CPU => GpuDeviceType::OpenCl,
                        _ => GpuDeviceType::OpenCl,
                    };

                    GpuDeviceInfo {
                        device_type,
                        name: format!("{} ({})", opencl_device.name, opencl_device.vendor),
                        total_memory: opencl_device.global_memsize as usize,
                        compute_units: opencl_device.max_compute_units,
                        clock_frequency: opencl_device.max_clock_frequency,
                        supports_fp64: self.supports_double_precision(),
                        supports_fp16: self.supports_half_precision(),
                        max_work_groupsize: opencl_device.max_work_groupsize,
                        memory_bandwidth,
                        l2_cachesize: opencl_device.global_mem_cachesize as usize,
                        shared_memory_per_block: opencl_device.local_memsize as usize,
                        registers_per_block: 0, // OpenCL doesn't expose this directly
                        warpsize: opencl_device.preferred_vector_width_float, // Approximate
                        max_threads_per_mp: opencl_device.max_work_groupsize as u32,
                        multiprocessor_count: opencl_device.max_compute_units,
                        supports_tensor_cores: false, // Most OpenCL devices don't have tensor cores
                        supports_mixed_precision: self.supports_half_precision(),
                        vendor: opencl_device.vendor.clone(),
                    }
                })
                .collect();

            Ok(devices)
        }

        fn create_context(&self, deviceid: usize) -> LinalgResult<Box<dyn GpuContext>> {
            if device_id >= self.devices.len() {
                return Err(LinalgError::ComputationError(format!(
                    "Invalid device ID: {} (available devices: {})",
                    device_id,
                    self.devices.len()
                )));
            }

            let device_info = &self.devices[device_id];

            // Create OpenCL context
            let (result, context) = cl_create_context(&[device_info.device_id]);
            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create OpenCL context: error code {}",
                    result
                )));
            }

            // Create command queue
            let (result, command_queue) = cl_create_command_queue(context, device_info.device_id);
            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create OpenCL command queue: error code {}",
                    result
                )));
            }

            let context_data = Arc::new(OpenClContextData {
                context,
                device_id: device_info.device_id,
                command_queue,
                device_info: device_info.clone(),
                kernel_cache: HashMap::new(),
                program_cache: HashMap::new(),
            });

            Ok(Box::new(OpenClContext::new(context_data, device_id)))
        }
    }

    /// OpenCL context with clBLAS integration and kernel caching
    #[derive(Debug)]
    pub struct OpenClContext {
        context_data: Arc<OpenClContextData>,
        device_index: usize,
        memory_pool: OpenClMemoryPool,
        performance_stats: OpenClPerformanceStats,
        kernel_compilation_cache: HashMap<String, String>, // Source hash -> compiled binary
    }

    impl OpenClContext {
        fn new(_context_data: Arc<OpenClContextData>, deviceindex: usize) -> Self {
            let memory_pool = OpenClMemoryPool::new(_context_data.context);
            let performance_stats = OpenClPerformanceStats::new();

            Self {
                context_data,
                device_index,
                memory_pool,
                performance_stats,
                kernel_compilation_cache: HashMap::new(),
            }
        }

        /// Get OpenCL context
        pub fn cl_context(&self) -> ClContext {
            self.context_data.context
        }

        /// Get command queue
        pub fn command_queue(&self) -> ClCommandQueue {
            self.context_data.command_queue
        }

        /// Compile and cache a kernel
        pub fn compile_kernel(
            &mut self,
            _kernel_name: &str,
            _source: &str,
        ) -> LinalgResult<ClKernel> {
            // In a real implementation, this would compile OpenCL kernel _source
            // For now, return a null pointer as mock
            Ok(SafeClPtr(ptr::null_mut()))
        }

        /// Get performance statistics
        pub fn performance_stats(&self) -> &OpenClPerformanceStats {
            &self.performance_stats
        }
    }

    impl GpuContext for OpenClContext {
        fn device_info(&self) -> &GpuDeviceInfo {
            // Convert OpenClDeviceInfo to GpuDeviceInfo
            static mut CACHED_INFO: Option<GpuDeviceInfo> = None;

            unsafe {
                if CACHED_INFO.is_none() {
                    let opencl_device = &self.context_data.device_info;
                    let memory_bandwidth =
                        (opencl_device.max_clock_frequency as f64 * 256.0) / 1000.0;

                    CACHED_INFO = Some(GpuDeviceInfo {
                        device_type: GpuDeviceType::OpenCl,
                        name: format!("{} ({})", opencl_device.name, opencl_device.vendor),
                        total_memory: opencl_device.global_memsize as usize,
                        compute_units: opencl_device.max_compute_units,
                        clock_frequency: opencl_device.max_clock_frequency,
                        supports_fp64: true, // Mock - would check extensions
                        supports_fp16: true, // Mock - would check extensions
                        max_work_groupsize: opencl_device.max_work_groupsize,
                        memory_bandwidth,
                        l2_cachesize: opencl_device.global_mem_cachesize as usize,
                        shared_memory_per_block: opencl_device.local_memsize as usize,
                        registers_per_block: 0,
                        warpsize: opencl_device.preferred_vector_width_float,
                        max_threads_per_mp: opencl_device.max_work_groupsize as u32,
                        multiprocessor_count: opencl_device.max_compute_units,
                        supports_tensor_cores: false,
                        supports_mixed_precision: true,
                        vendor: opencl_device.vendor.clone(),
                    });
                }

                CACHED_INFO.as_ref().unwrap()
            }
        }

        fn synchronize(&self) -> LinalgResult<()> {
            let result = cl_finish(self.context_data.command_queue);
            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "OpenCL synchronization failed: error code {}",
                    result
                )));
            }
            Ok(())
        }

        fn available_memory(&self) -> LinalgResult<usize> {
            // Mock implementation - would query actual available memory
            Ok(self.context_data.device_info.global_memsize as usize / 2)
        }
    }

    impl GpuContextAlloc for OpenClContext {
        fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
            &self,
            size: usize,
        ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
            let buffer = OpenClBuffer::new(
                size,
                self.context_data.context,
                self.context_data.command_queue,
            )?;
            Ok(Box::new(buffer))
        }
    }

    /// OpenCL memory pool for efficient buffer management
    #[derive(Debug)]
    struct OpenClMemoryPool {
        context: ClContext,
        total_allocated: usize,
        peak_usage: usize,
        free_buffers: HashMap<usize, Vec<ClMem>>,
    }

    impl OpenClMemoryPool {
        fn new(context: ClContext) -> Self {
            Self {
                context,
                total_allocated: 0,
                peak_usage: 0,
                free_buffers: HashMap::new(),
            }
        }

        #[allow(dead_code)]
        fn allocate(&mut self, size: usize) -> LinalgResult<ClMem> {
            // Try to reuse existing buffer
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    return Ok(buffer);
                }
            }

            // Allocate new buffer
            let (result, buffer) = cl_create_buffer(self.context, CL_MEM_READ_WRITE, size);
            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "OpenCL buffer allocation failed: error code {}",
                    result
                )));
            }

            self.total_allocated += size;
            self.peak_usage = self.peak_usage.max(self.total_allocated);

            Ok(buffer)
        }

        #[allow(dead_code)]
        fn deallocate(&mut self, buffer: ClMem, size: usize) {
            self.free_buffers.entry(size).or_default().push(buffer);
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }
    }

    /// OpenCL buffer implementation
    #[derive(Debug)]
    struct OpenClBuffer<T> {
        buffer: ClMem,
        size: usize,
        context: ClContext,
        command_queue: ClCommandQueue,
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T: Clone + Send + Sync + Copy> OpenClBuffer<T> {
        fn new(
            size: usize,
            context: ClContext,
            command_queue: ClCommandQueue,
        ) -> LinalgResult<Self> {
            let bytesize = size * std::mem::size_of::<T>();

            let (result, buffer) = cl_create_buffer(context, CL_MEM_READ_WRITE, bytesize);
            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create OpenCL buffer: error code {}",
                    result
                )));
            }

            Ok(Self {
                buffer,
                size,
                context,
                command_queue_phantom: std::marker::PhantomData,
            })
        }

        /// Get OpenCL memory object
        pub fn cl_mem(&self) -> ClMem {
            self.buffer
        }
    }

    impl<T: Clone + Send + Sync + Copy + std::fmt::Debug> GpuBuffer<T> for OpenClBuffer<T> {
        fn len(&self) -> usize {
            self.size
        }

        fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            let bytesize = data.len() * std::mem::size_of::<T>();
            let result = cl_enqueue_write_buffer(
                self.command_queue,
                self.buffer,
                1, // blocking
                0, // offset
                bytesize,
                data.as_ptr() as *const std::ffi::c_void,
            );

            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "OpenCL host-to-device copy failed: error code {}",
                    result
                )));
            }

            Ok(())
        }

        fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            let bytesize = data.len() * std::mem::size_of::<T>();
            let result = cl_enqueue_read_buffer(
                self.command_queue,
                self.buffer,
                1, // blocking
                0, // offset
                bytesize,
                data.as_mut_ptr() as *mut std::ffi::c_void,
            );

            if result != CL_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "OpenCL device-to-host copy failed: error code {}",
                    result
                )));
            }

            Ok(())
        }

        fn device_ptr(&self) -> *mut std::ffi::c_void {
            self.buffer.as_ptr()
        }
    }

    impl<T> Drop for OpenClBuffer<T> {
        fn drop(&mut self) {
            if !self.buffer.0.is_null() {
                let _ = cl_release_mem_object(self.buffer);
            }
        }
    }

    /// Performance statistics for OpenCL operations
    #[derive(Debug, Clone)]
    pub struct OpenClPerformanceStats {
        pub kernel_executions: usize,
        pub buffer_operations: usize,
        pub total_kernel_time_ms: f64,
        pub total_transfer_time_ms: f64,
        pub compilation_time_ms: f64,
        pub cache_hits: usize,
        pub cache_misses: usize,
    }

    impl OpenClPerformanceStats {
        fn new() -> Self {
            Self {
                kernel_executions: 0,
                buffer_operations: 0,
                total_kernel_time_ms: 0.0,
                total_transfer_time_ms: 0.0,
                compilation_time_ms: 0.0,
                cache_hits: 0,
                cache_misses: 0,
            }
        }

        pub fn kernel_efficiency(&self) -> f64 {
            if self.total_kernel_time_ms + self.total_transfer_time_ms == 0.0 {
                return 0.0;
            }
            self.total_kernel_time_ms / (self.total_kernel_time_ms + self.total_transfer_time_ms)
        }

        pub fn cache_hit_rate(&self) -> f64 {
            let total_accesses = self.cache_hits + self.cache_misses;
            if total_accesses == 0 {
                return 0.0;
            }
            self.cache_hits as f64 / total_accesses as f64
        }
    }
}

/// Placeholder ROCm backend (requires ROCm feature and runtime)
#[cfg(feature = "rocm")]
pub mod rocm {
    use super::*;

    pub struct RocmBackend {
        #[allow(dead_code)]
        devices: Vec<String>,
    }

    impl RocmBackend {
        pub fn new() -> LinalgResult<Self> {
            // In a real implementation, this would initialize ROCm/HIP
            Ok(Self {
                devices: Vec::new(),
            })
        }
    }

    impl GpuBackend for RocmBackend {
        fn name(&self) -> &str {
            "ROCm"
        }

        fn is_available(&self) -> bool {
            // In a real implementation, check ROCm availability
            false
        }

        fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
            // In a real implementation, enumerate ROCm devices
            Ok(vec![])
        }

        fn create_context(&self_deviceid: usize) -> LinalgResult<Box<dyn GpuContext>> {
            Err(LinalgError::ComputationError(
                "ROCm backend not fully implemented".to_string(),
            ))
        }
    }
}

/// Placeholder Metal backend (requires Metal feature - macOS/iOS only)
#[cfg(feature = "metal")]
pub mod metal {
    use super::*;

    pub struct MetalBackend {
        #[allow(dead_code)]
        device_registry: HashMap<String, String>,
    }

    impl MetalBackend {
        pub fn new() -> LinalgResult<Self> {
            // In a real implementation, this would initialize Metal
            Ok(Self {
                device_registry: HashMap::new(),
            })
        }
    }

    impl GpuBackend for MetalBackend {
        fn name(&self) -> &str {
            "Metal"
        }

        fn is_available(&self) -> bool {
            // In a real implementation, check Metal availability (macOS/iOS only)
            cfg!(target_os = "macos") || cfg!(target_os = "ios")
        }

        fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
            // In a real implementation, enumerate Metal devices
            Ok(vec![])
        }

        fn create_context(&self_deviceid: usize) -> LinalgResult<Box<dyn GpuContext>> {
            Err(LinalgError::ComputationError(
                "Metal backend not fully implemented".to_string(),
            ))
        }
    }
}

/// CPU fallback backend that implements GPU traits using CPU operations
pub struct CpuFallbackBackend {
    device_info: GpuDeviceInfo,
}

impl Default for CpuFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFallbackBackend {
    pub fn new() -> Self {
        Self {
            device_info: GpuDeviceInfo {
                device_type: GpuDeviceType::OpenCl, // Use OpenCL as generic type
                name: "CPU Fallback".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB estimate
                compute_units: num_cpus::get() as u32,
                clock_frequency: 3000, // 3GHz estimate
                supports_fp64: true,
                supports_fp16: false,
                max_work_groupsize: 1024,
                memory_bandwidth: 100.0, // CPU memory bandwidth estimate
                l2_cachesize: 32 * 1024 * 1024, // 32MB L2 cache estimate
                shared_memory_per_block: 0, // No shared memory concept for CPU
                registers_per_block: 0,
                warpsize: 1, // No SIMD grouping for CPU
                max_threads_per_mp: 1,
                multiprocessor_count: num_cpus::get() as u32,
                supports_tensor_cores: false,
                supports_mixed_precision: false,
                vendor: "CPU".to_string(),
            },
        }
    }
}

impl GpuBackend for CpuFallbackBackend {
    fn name(&self) -> &str {
        "CPU Fallback"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
        Ok(vec![self.device_info.clone()])
    }

    fn create_context(&self, deviceid: usize) -> LinalgResult<Box<dyn GpuContext>> {
        if device_id != 0 {
            return Err(LinalgError::ComputationError(
                "CPU fallback only has one device".to_string(),
            ));
        }

        Ok(Box::new(CpuFallbackContext {
            device_info: self.device_info.clone(),
        }))
    }
}

/// CPU fallback context implementation
#[derive(Debug)]
struct CpuFallbackContext {
    device_info: GpuDeviceInfo,
}

impl GpuContext for CpuFallbackContext {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    fn synchronize(&self) -> LinalgResult<()> {
        // CPU operations are always synchronous
        Ok(())
    }

    fn available_memory(&self) -> LinalgResult<usize> {
        // Return a reasonable estimate for available system memory
        Ok(self.device_info.total_memory / 2)
    }
}

impl GpuContextAlloc for CpuFallbackContext {
    fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
        &self,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
        Ok(Box::new(CpuBuffer::new(size)))
    }
}

/// CPU buffer implementation that just wraps a Vec
#[derive(Debug)]
struct CpuBuffer<T> {
    data: Vec<T>,
}

impl<T: Clone + Send + Sync> CpuBuffer<T> {
    fn new(size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
        }
    }
}

impl<T: Clone + Send + Sync + Copy + std::fmt::Debug> GpuBuffer<T> for CpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()> {
        self.data.clear();
        self.data.extend_from_slice(data);
        Ok(())
    }

    fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()> {
        if data.len() != self.data.len() {
            return Err(LinalgError::ShapeError("Buffer size mismatch".to_string()));
        }
        data.copy_from_slice(&self.data);
        Ok(())
    }

    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.data.as_ptr() as *mut std::ffi::c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_backend() {
        let backend = CpuFallbackBackend::new();
        assert_eq!(backend.name(), "CPU Fallback");
        assert!(backend.is_available());

        let devices = backend.list_devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "CPU Fallback");
    }

    #[test]
    fn test_cpu_fallback_context() {
        let backend = CpuFallbackBackend::new();
        let context = backend.create_context(0).unwrap();

        assert_eq!(context.device_info().name, "CPU Fallback");
        assert!(context.available_memory().unwrap() > 0);
        assert!(context.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_buffer() {
        let backend = CpuFallbackBackend::new();
        let device_info = backend.device_info.clone();

        // Create context directly to access allocate_buffer method
        let cpu_context = CpuFallbackContext { device_info };
        let mut buffer = cpu_context.allocate_buffer::<f32>(10).unwrap();
        assert_eq!(buffer.len(), 0); // Initially empty

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.copy_from_host(&data).unwrap();
        assert_eq!(buffer.len(), 5);

        let mut output = vec![0.0; 5];
        buffer.copy_to_host(&mut output).unwrap();
        assert_eq!(output, data);
    }
}
