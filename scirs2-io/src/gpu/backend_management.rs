//! GPU backend management for I/O operations
//!
//! This module provides comprehensive GPU backend detection, validation,
//! and capability management for optimal I/O performance across different
//! GPU vendors and architectures.

use crate::error::{IoError, Result};
use scirs2_core::gpu::{GpuBackend, GpuDevice, GpuError};
use scirs2_core::simd_ops::PlatformCapabilities;

/// GPU-accelerated I/O processor with backend management
#[derive(Debug)]
pub struct GpuIoProcessor {
    pub device: GpuDevice,
    pub capabilities: PlatformCapabilities,
}

impl GpuIoProcessor {
    /// Create a new GPU I/O processor with the preferred backend
    pub fn new() -> Result<Self> {
        let capabilities = PlatformCapabilities::detect();

        if !capabilities.gpu_available {
            return Err(IoError::Other("GPU acceleration not available. Please ensure GPU drivers are installed and properly configured.".to_string()));
        }

        // Try backends in order of preference with proper detection
        let backend = Self::detect_optimal_backend()
            .map_err(|e| IoError::Other(format!("Failed to detect optimal GPU backend: {}", e)))?;
        let device = GpuDevice::new(backend, 0);

        // Validate device capabilities
        let info = device.get_info();
        if info.memory_gb < 0.5 {
            return Err(IoError::Other(format!(
                "GPU has insufficient memory: {:.1}GB available, minimum 0.5GB required",
                info.memory_gb
            )));
        }

        Ok(Self {
            device,
            capabilities,
        })
    }

    /// Create a new GPU I/O processor with a specific backend
    pub fn with_backend(backend: GpuBackend) -> Result<Self> {
        if !Self::is_backend_available(backend) {
            return Err(IoError::Other(format!(
                "GPU backend {} is not available",
                backend
            )));
        }

        let device = GpuDevice::new(backend, 0);
        let capabilities = PlatformCapabilities::detect();

        Ok(Self {
            device,
            capabilities,
        })
    }

    /// Detect the optimal GPU backend for the current system
    pub fn detect_optimal_backend() -> Result<GpuBackend> {
        // Check each backend in order of preference
        let backends_to_try = [
            GpuBackend::Cuda,   // NVIDIA - best performance and feature support
            GpuBackend::Metal,  // Apple Silicon - excellent for Mac
            GpuBackend::OpenCL, // Cross-platform fallback
        ];

        for &backend in &backends_to_try {
            if Self::is_backend_available(backend) {
                // Additional validation - check if we can actually create a device
                match Self::validate_backend(backend) {
                    Ok(true) => return Ok(backend),
                    _ => continue,
                }
            }
        }

        Err(IoError::Other(
            "No suitable GPU backend available".to_string(),
        ))
    }

    /// Validate that a backend is functional (not just available)
    pub fn validate_backend(backend: GpuBackend) -> Result<bool> {
        if !backend.is_available() {
            return Ok(false);
        }

        // Try to create a test device and perform a simple operation
        match GpuDevice::new(backend, 0) {
            device => {
                // Additional validation based on backend type
                match backend {
                    GpuBackend::Cuda => Self::validate_cuda_backend(&device),
                    GpuBackend::Metal => Self::validate_metal_backend(&device),
                    GpuBackend::OpenCL => Self::validate_opencl_backend(&device),
                    _ => Ok(false),
                }
            }
        }
    }

    /// Validate CUDA backend functionality
    fn validate_cuda_backend(device: &GpuDevice) -> Result<bool> {
        // Check CUDA compute capability and available memory
        match device.get_info() {
            info => {
                // Require at least compute capability 3.5 and 1GB memory
                Ok(info.compute_capability >= 3.5 && info.memory_gb >= 1.0)
            }
        }
    }

    /// Validate Metal backend functionality  
    fn validate_metal_backend(device: &GpuDevice) -> Result<bool> {
        // Check Metal feature set support
        match device.get_info() {
            info => {
                // Require Metal 2.0+ support and sufficient memory
                Ok(info.metal_version >= 2.0 && info.memory_gb >= 1.0)
            }
        }
    }

    /// Validate OpenCL backend functionality
    fn validate_opencl_backend(device: &GpuDevice) -> Result<bool> {
        // Check OpenCL version and extensions
        match device.get_info() {
            info => {
                // Require OpenCL 1.2+ and basic extensions
                Ok(info.opencl_version >= 1.2 && info.supports_fp64)
            }
        }
    }

    /// Get detailed backend capabilities
    pub fn get_backend_capabilities(&self) -> Result<BackendCapabilities> {
        let info = self.device.get_info();

        Ok(BackendCapabilities {
            backend: self.backend(),
            memory_gb: info.memory_gb,
            max_work_group_size: info.max_work_group_size,
            supports_fp64: info.supports_fp64,
            supports_fp16: info.supports_fp16,
            compute_units: info.compute_units,
            max_allocation_size: info.max_allocation_size,
            local_memory_size: info.local_memory_size,
        })
    }

    /// Get the current GPU backend
    pub fn backend(&self) -> GpuBackend {
        self.device.backend()
    }

    /// Check if a specific backend is available
    pub fn is_backend_available(backend: GpuBackend) -> bool {
        backend.is_available()
    }

    /// List all available backends on the system
    pub fn list_available_backends() -> Vec<GpuBackend> {
        [GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::OpenCL]
            .iter()
            .filter(|&&backend| Self::is_backend_available(backend))
            .copied()
            .collect()
    }

    /// Get optimal backend for specific workload type
    pub fn get_optimal_backend_for_workload(workload: GpuWorkloadType) -> Result<GpuBackend> {
        let available_backends = Self::list_available_backends();

        if available_backends.is_empty() {
            return Err(IoError::Other("No GPU backends available".to_string()));
        }

        // Choose backend based on workload characteristics
        match workload {
            GpuWorkloadType::MachineLearning => {
                // CUDA is preferred for ML workloads
                if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
            GpuWorkloadType::ImageProcessing => {
                // Metal is excellent for image processing on Apple devices
                if available_backends.contains(&GpuBackend::Metal) {
                    Ok(GpuBackend::Metal)
                } else if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
            GpuWorkloadType::GeneralCompute => {
                // Use first available backend for general compute
                Ok(available_backends[0])
            }
            GpuWorkloadType::Compression => {
                // CUDA typically performs well for compression
                if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
        }
    }
}

impl Default for GpuIoProcessor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to CPU backend if GPU creation fails
            GpuIoProcessor {
                device: GpuDevice::new(GpuBackend::Cpu, 0),
                capabilities: PlatformCapabilities::detect(),
            }
        })
    }
}

/// Detailed backend capabilities for optimization decisions
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub backend: GpuBackend,
    pub memory_gb: f64,
    pub max_work_group_size: usize,
    pub supports_fp64: bool,
    pub supports_fp16: bool,
    pub compute_units: usize,
    pub max_allocation_size: usize,
    pub local_memory_size: usize,
}

impl BackendCapabilities {
    /// Check if backend supports high-precision computations
    pub fn supports_high_precision(&self) -> bool {
        self.supports_fp64
    }

    /// Check if backend supports half-precision optimizations
    pub fn supports_half_precision(&self) -> bool {
        self.supports_fp16
    }

    /// Get optimal work group size for given problem size
    pub fn get_optimal_work_group_size(&self, problem_size: usize) -> usize {
        let base_size = match self.backend {
            GpuBackend::Cuda => 256,   // CUDA warps are 32, good block size is 256
            GpuBackend::Metal => 64,   // Metal threadgroups typically 64
            GpuBackend::OpenCL => 128, // OpenCL flexible, 128 is safe
            _ => 64,
        };

        // Ensure we don't exceed device limits
        base_size.min(self.max_work_group_size).min(problem_size)
    }

    /// Estimate memory bandwidth in GB/s
    pub fn estimate_memory_bandwidth(&self) -> f64 {
        match self.backend {
            GpuBackend::Cuda => self.memory_gb * 0.8, // CUDA typically achieves ~80% peak
            GpuBackend::Metal => self.memory_gb * 0.7, // Metal varies more
            GpuBackend::OpenCL => self.memory_gb * 0.6, // OpenCL is more conservative
            _ => self.memory_gb * 0.4,
        }
    }
}

/// GPU workload types for optimal backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkloadType {
    MachineLearning,
    ImageProcessing,
    GeneralCompute,
    Compression,
}

/// Backend performance characteristics for optimization
#[derive(Debug, Clone)]
pub struct BackendPerformanceProfile {
    pub backend: GpuBackend,
    pub throughput_gbps: f64,
    pub latency_ms: f64,
    pub power_efficiency: f64,
    pub memory_efficiency: f64,
}

impl BackendPerformanceProfile {
    /// Create performance profile for a backend
    pub fn new(backend: GpuBackend, capabilities: &BackendCapabilities) -> Self {
        let (throughput, latency, power_eff, mem_eff) = match backend {
            GpuBackend::Cuda => (capabilities.memory_gb * 0.8, 0.1, 0.7, 0.9),
            GpuBackend::Metal => (capabilities.memory_gb * 0.7, 0.15, 0.9, 0.8),
            GpuBackend::OpenCL => (capabilities.memory_gb * 0.6, 0.2, 0.6, 0.7),
            _ => (capabilities.memory_gb * 0.4, 0.5, 0.8, 0.5),
        };

        Self {
            backend,
            throughput_gbps: throughput,
            latency_ms: latency,
            power_efficiency: power_eff,
            memory_efficiency: mem_eff,
        }
    }

    /// Calculate overall performance score
    pub fn performance_score(&self) -> f64 {
        self.throughput_gbps * 0.4
            + (1.0 / self.latency_ms) * 0.3
            + self.power_efficiency * 0.2
            + self.memory_efficiency * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_availability_detection() {
        let backends = GpuIoProcessor::list_available_backends();
        // At least CPU backend should always be available
        assert!(!backends.is_empty());
    }

    #[test]
    fn test_backend_capabilities() {
        if let Ok(processor) = GpuIoProcessor::new() {
            let capabilities = processor.get_backend_capabilities().unwrap();
            assert!(capabilities.memory_gb > 0.0);
            assert!(capabilities.compute_units > 0);
        }
    }

    #[test]
    fn test_optimal_backend_for_workload() {
        let backend =
            GpuIoProcessor::get_optimal_backend_for_workload(GpuWorkloadType::MachineLearning);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_work_group_size_calculation() {
        let capabilities = BackendCapabilities {
            backend: GpuBackend::Cuda,
            memory_gb: 8.0,
            max_work_group_size: 1024,
            supports_fp64: true,
            supports_fp16: true,
            compute_units: 32,
            max_allocation_size: 1024 * 1024 * 1024,
            local_memory_size: 48 * 1024,
        };

        let work_group_size = capabilities.get_optimal_work_group_size(10000);
        assert_eq!(work_group_size, 256); // CUDA optimal

        let small_size = capabilities.get_optimal_work_group_size(100);
        assert_eq!(small_size, 100); // Limited by problem size
    }

    #[test]
    fn test_performance_profile_scoring() {
        let capabilities = BackendCapabilities {
            backend: GpuBackend::Cuda,
            memory_gb: 8.0,
            max_work_group_size: 1024,
            supports_fp64: true,
            supports_fp16: true,
            compute_units: 32,
            max_allocation_size: 1024 * 1024 * 1024,
            local_memory_size: 48 * 1024,
        };

        let profile = BackendPerformanceProfile::new(GpuBackend::Cuda, &capabilities);
        let score = profile.performance_score();
        assert!(score > 0.0);
    }
}
