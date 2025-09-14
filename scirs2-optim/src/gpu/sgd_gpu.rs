//! GPU-accelerated SGD optimizer implementation

use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::Result as OptimResult;
use crate::gpu::{GpuOptimError, GpuOptimizerConfig, GpuOptimizerMemory};
use crate::optimizers::{Optimizer, SGD};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuKernelHandle;

/// GPU-accelerated SGD optimizer
#[allow(dead_code)]
pub struct SGDGpu<A: Float + ScalarOperand + Debug> {
    /// CPU SGD optimizer for fallback
    cpu_optimizer: SGD<A>,

    /// GPU memory manager
    gpu_memory: Option<GpuOptimizerMemory<A>>,

    /// GPU kernel handle
    kernel_handle: Option<Arc<GpuKernelHandle>>,

    /// GPU kernel handle for momentum variant
    momentum_kernel_handle: Option<Arc<GpuKernelHandle>>,

    /// Whether optimizer is on GPU
    on_gpu: bool,

    /// Momentum buffer on GPU
    momentum_gpu: Option<scirs2_core::gpu::GpuBuffer<A>>,
}

impl<A: Float + ScalarOperand + Debug> SGDGpu<A> {
    /// Create a new GPU-accelerated SGD optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            cpu_optimizer: SGD::new(learning_rate),
            gpu_memory: None,
            kernel_handle: None,
            momentum_kernel_handle: None,
            on_gpu: false,
            momentum_gpu: None,
        }
    }

    /// Create with momentum
    pub fn with_momentum(learning_rate: A, momentum: A) -> Self {
        Self {
            cpu_optimizer: SGD::with_momentum(learning_rate, momentum),
            gpu_memory: None,
            kernel_handle: None,
            momentum_kernel_handle: None,
            on_gpu: false,
            momentum_gpu: None,
        }
    }

    /// Initialize GPU resources
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_gpu(
        &mut self,
        size: usize,
        config: GpuOptimizerConfig,
    ) -> Result<(), GpuOptimError> {
        // Create GPU memory manager
        let mut gpu_memory = GpuOptimizerMemory::new(size, config)?;
        gpu_memory.allocate()?;

        // Load SGD kernel
        #[cfg(feature = "gpu")]
        {
            let kernel_name = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                if self.cpu_optimizer.momentum != A::zero() {
                    "sgd_momentum_update_f32"
                } else {
                    "sgd_update_f32"
                }
            } else if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f64>() {
                if self.cpu_optimizer.momentum != A::zero() {
                    "sgd_momentum_update_f64"
                } else {
                    "sgd_update_f64"
                }
            } else {
                return Err(GpuOptimError::UnsupportedOperation(
                    "Unsupported data type for GPU SGD".to_string(),
                ));
            };

            let kernel_handle = gpu_memory.context().get_kernel(kernel_name)?;
            self.kernel_handle = Some(Arc::new(kernel_handle));

            // Allocate momentum buffer if needed
            if self.cpu_optimizer.momentum != A::zero() {
                self.momentum_gpu = Some(gpu_memory.context().create_buffer::<A>(size));

                // Load tensor core optimized kernel for large batches
                let tc_kernel_name = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>()
                {
                    "sgd_momentum_tensor_core_f16"
                } else {
                    kernel_name // Fall back to regular kernel
                };

                if let Ok(tc_kernel) = gpu_memory.context().get_kernel(tc_kernel_name) {
                    self.momentum_kernel_handle = Some(Arc::new(tc_kernel));
                }
            }
        }

        self.gpu_memory = Some(gpu_memory);
        Ok(())
    }

    /// Move optimizer state to GPU
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        if self.gpu_memory.is_none() {
            return Err(GpuOptimError::NotInitialized);
        }

        self.on_gpu = true;
        Ok(())
    }

    /// Move optimizer state back to CPU
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.on_gpu = false;
        Ok(())
    }

    /// Perform optimization step on GPU
    pub fn step_gpu<S1, S2, D>(
        &mut self,
        params: &mut ArrayBase<S1, D>,
        gradients: &ArrayBase<S2, D>,
    ) -> Result<(), GpuOptimError>
    where
        S1: DataMut<Elem = A>,
        S2: Data<Elem = A>,
        D: Dimension,
    {
        if !self.on_gpu {
            return Err(GpuOptimError::InvalidState(
                "Optimizer not on GPU".to_string(),
            ));
        }

        let gpu_memory = self
            .gpu_memory
            .as_mut()
            .ok_or(GpuOptimError::NotInitialized)?;

        let kernel = self
            .kernel_handle
            .as_ref()
            .ok_or(GpuOptimError::NotInitialized)?;

        // Copy data to GPU
        gpu_memory.copy_params_to_gpu(params)?;

        // Copy gradients to GPU
        if let Some(ref grads_gpu) = gpu_memory.grads_gpu {
            let grads_slice = gradients.as_slice().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;
            grads_gpu.copy_from_host(grads_slice);
        }

        // Set kernel parameters
        #[cfg(feature = "gpu")]
        {
            kernel.set_buffer("params", gpu_memory.params_gpu.as_ref().unwrap());
            kernel.set_buffer("grads", gpu_memory.grads_gpu.as_ref().unwrap());

            if self.cpu_optimizer.momentum != A::zero() {
                kernel.set_buffer("momentum", self.momentum_gpu.as_ref().unwrap());

                if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                    kernel.set_f32(
                        "lr",
                        self.cpu_optimizer.get_learning_rate().to_f32().unwrap(),
                    );
                    kernel.set_f32(
                        "momentum_factor",
                        self.cpu_optimizer.momentum.to_f32().unwrap(),
                    );
                    kernel.set_f32(
                        "weight_decay",
                        self.cpu_optimizer.weight_decay.to_f32().unwrap(),
                    );
                    if self.cpu_optimizer.nesterov {
                        kernel.set_i32("nesterov", 1);
                    } else {
                        kernel.set_i32("nesterov", 0);
                    }
                } else {
                    kernel.set_f64(
                        "lr",
                        self.cpu_optimizer.get_learning_rate().to_f64().unwrap(),
                    );
                    kernel.set_f64(
                        "momentum_factor",
                        self.cpu_optimizer.momentum.to_f64().unwrap(),
                    );
                    kernel.set_f64(
                        "weight_decay",
                        self.cpu_optimizer.weight_decay.to_f64().unwrap(),
                    );
                    if self.cpu_optimizer.nesterov {
                        kernel.set_i32("nesterov", 1);
                    } else {
                        kernel.set_i32("nesterov", 0);
                    }
                }
            } else {
                // Simple SGD without momentum
                if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                    kernel.set_f32(
                        "lr",
                        self.cpu_optimizer.get_learning_rate().to_f32().unwrap(),
                    );
                    kernel.set_f32(
                        "weight_decay",
                        self.cpu_optimizer.weight_decay.to_f32().unwrap(),
                    );
                } else {
                    kernel.set_f64(
                        "lr",
                        self.cpu_optimizer.get_learning_rate().to_f64().unwrap(),
                    );
                    kernel.set_f64(
                        "weight_decay",
                        self.cpu_optimizer.weight_decay.to_f64().unwrap(),
                    );
                }
            }

            kernel.set_i32("n", params.len() as i32);

            // Calculate grid and block dimensions
            let (grid_size, block_size) =
                crate::gpu::utils::calculate_block_size(params.len(), 256);

            // Launch kernel
            kernel.dispatch([grid_size as u32, 1, 1]);
        }

        // Copy results back to CPU
        gpu_memory.copy_params_from_gpu(params)?;

        Ok(())
    }
}

// Implement standard Optimizer trait with CPU fallback
impl<A, D> Optimizer<A, D> for SGDGpu<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> OptimResult<Array<A, D>> {
        if self.on_gpu {
            // Try GPU execution
            let mut params_mut = params.clone();
            match self.step_gpu(&mut params_mut, gradients) {
                Ok(()) => Ok(params_mut),
                Err(e) => {
                    // Fall back to CPU on error
                    eprintln!("GPU execution failed, falling back to CPU: {}", e);
                    self.cpu_optimizer.step(params, gradients)
                }
            }
        } else {
            // Use CPU implementation
            self.cpu_optimizer.step(params, gradients)
        }
    }

    fn get_learning_rate(&self) -> A {
        self.cpu_optimizer.get_learning_rate()
    }

    fn set_learning_rate(&mut self, learningrate: A) {
        self.cpu_optimizer.set_learning_rate(learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_sgd_gpu_creation() {
        let optimizer = SGDGpu::<f32>::new(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);
        assert!(!optimizer.on_gpu);
    }

    #[test]
    fn test_sgd_gpu_with_momentum() {
        let optimizer = SGDGpu::<f32>::with_momentum(0.01, 0.9);
        assert_eq!(optimizer.get_learning_rate(), 0.01);
        assert_eq!(optimizer.cpu_optimizer.momentum, 0.9);
    }

    #[test]
    fn test_sgd_gpu_cpu_fallback() {
        let mut optimizer = SGDGpu::new(0.01);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Should use CPU implementation when not on GPU
        let result = optimizer.step(&params, &grads);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 3);

        // Check that parameters moved in negative gradient direction
        assert!(updated[0] < params[0]);
        assert!(updated[1] < params[1]);
        assert!(updated[2] < params[2]);
    }
}
