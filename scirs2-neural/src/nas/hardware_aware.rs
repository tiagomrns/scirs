//! Hardware-Aware Neural Architecture Search
//!
//! This module provides hardware-aware optimization capabilities that consider
//! deployment constraints such as latency, memory usage, energy consumption,
//! and platform-specific optimizations.

use crate::error::Result;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding,
    search_space::{Architecture, LayerType},
};
use std::collections::HashMap;
use std::sync::Arc;
/// Hardware platform types
#[derive(Debug, Clone, PartialEq)]
pub enum HardwarePlatform {
    /// Desktop/Server CPU
    CPU,
    /// NVIDIA GPU with CUDA
    GPU,
    /// Mobile ARM CPU
    MobileARM,
    /// Edge TPU
    EdgeTPU,
    /// FPGA
    FPGA,
    /// Custom accelerator
    Custom(String),
}
/// Hardware constraints for deployment
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Target platform
    pub platform: HardwarePlatform,
    /// Maximum inference latency in milliseconds
    pub max_latency_ms: Option<f64>,
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<f64>,
    /// Maximum energy consumption in mJ per inference
    pub max_energy_mj: Option<f64>,
    /// Maximum model size in MB
    pub max_model_size_mb: Option<f64>,
    /// Minimum throughput (inferences per second)
    pub min_throughput: Option<f64>,
    /// Available compute units (cores, CUs, etc.)
    pub compute_units: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: Option<f64>,
    /// Quantization support
    pub quantization_support: QuantizationSupport,
impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            platform: HardwarePlatform::CPU,
            max_latency_ms: Some(100.0),
            max_memory_mb: Some(512.0),
            max_energy_mj: Some(100.0),
            max_model_size_mb: Some(50.0),
            min_throughput: Some(10.0),
            compute_units: 4,
            memory_bandwidth_gbps: Some(25.6),
            quantization_support: QuantizationSupport::Int8,
        }
    }
/// Quantization support levels
pub enum QuantizationSupport {
    /// No quantization support
    None,
    /// 8-bit integer quantization
    Int8,
    /// 16-bit float quantization
    Float16,
    /// Mixed precision (INT8 + FP16)
    Mixed,
    /// Custom quantization schemes
    Custom(Vec<String>),
/// Latency prediction model
pub struct LatencyPredictor {
    /// Platform-specific lookup tables
    operation_latencies: HashMap<String, f64>,
    /// Memory access costs
    memory_costs: HashMap<String, f64>,
    /// Parallelization factors
    parallelization_factors: HashMap<String, f64>,
    /// Platform characteristics
    platform: HardwarePlatform,
impl LatencyPredictor {
    /// Create a new latency predictor for a platform
    pub fn new(platform: HardwarePlatform) -> Self {
        let mut predictor = Self {
            operation_latencies: HashMap::new(),
            memory_costs: HashMap::new(),
            parallelization_factors: HashMap::new(),
            platform,
        };
        predictor.initialize_platform_characteristics();
        predictor
    /// Initialize platform-specific characteristics
    fn initialize_platform_characteristics(&mut self) {
        match self.platform {
            HardwarePlatform::CPU => {
                self.operation_latencies.insert("dense".to_string(), 0.1);
                self.operation_latencies.insert("conv2d".to_string(), 0.5);
                self.operation_latencies.insert("pooling".to_string(), 0.05);
                self.operation_latencies
                    .insert("activation".to_string(), 0.01);
                    .insert("batchnorm".to_string(), 0.02);
                self.memory_costs.insert("weight_load".to_string(), 0.001);
                self.memory_costs
                    .insert("activation_store".to_string(), 0.0005);
                self.parallelization_factors
                    .insert("dense".to_string(), 0.8);
                    .insert("conv2d".to_string(), 0.9);
            }
            HardwarePlatform::GPU => {
                self.operation_latencies.insert("dense".to_string(), 0.02);
                self.operation_latencies.insert("conv2d".to_string(), 0.1);
                self.operation_latencies.insert("pooling".to_string(), 0.01);
                    .insert("activation".to_string(), 0.005);
                    .insert("batchnorm".to_string(), 0.01);
                self.memory_costs.insert("weight_load".to_string(), 0.0001);
                    .insert("activation_store".to_string(), 0.00005);
                    .insert("dense".to_string(), 0.95);
                    .insert("conv2d".to_string(), 0.98);
            HardwarePlatform::MobileARM => {
                self.operation_latencies.insert("dense".to_string(), 0.3);
                self.operation_latencies.insert("conv2d".to_string(), 1.0);
                self.operation_latencies.insert("pooling".to_string(), 0.1);
                    .insert("activation".to_string(), 0.02);
                    .insert("batchnorm".to_string(), 0.05);
                self.memory_costs.insert("weight_load".to_string(), 0.002);
                    .insert("activation_store".to_string(), 0.001);
                    .insert("dense".to_string(), 0.6);
                    .insert("conv2d".to_string(), 0.7);
            HardwarePlatform::EdgeTPU => {
                self.operation_latencies.insert("dense".to_string(), 0.05);
                self.operation_latencies.insert("conv2d".to_string(), 0.2);
                self.operation_latencies.insert("pooling".to_string(), 0.02);
                    .insert("batchnorm".to_string(), 0.015);
                self.memory_costs.insert("weight_load".to_string(), 0.0002);
                    .insert("activation_store".to_string(), 0.0001);
                    .insert("dense".to_string(), 0.9);
                    .insert("conv2d".to_string(), 0.95);
            _ => {
                // Default values for unknown platforms
                self.operation_latencies.insert("dense".to_string(), 0.2);
                self.operation_latencies.insert("conv2d".to_string(), 0.8);
                self.operation_latencies.insert("pooling".to_string(), 0.08);
                    .insert("batchnorm".to_string(), 0.04);
    /// Predict latency for an architecture
    pub fn predict_latency(
        &self,
        architecture: &Architecture,
        inputshape: &[usize],
    ) -> Result<f64> {
        let mut total_latency = 0.0;
        let mut currentshape = inputshape.to_vec();
        for layer in &architecture.layers {
            let layer_latency = self.predict_layer_latency(layer, &currentshape)?;
            total_latency += layer_latency;
            // Update shape for next layer
            currentshape = self.compute_outputshape(layer, &currentshape)?;
        // Add overhead for model loading and initialization
        total_latency += self.compute_initialization_overhead(architecture)?;
        Ok(total_latency)
    /// Predict latency for a single layer
    fn predict_layer_latency(&self, layer: &LayerType, inputshape: &[usize]) -> Result<f64> {
        let base_latency = match layer {
            LayerType::Dense(units) => {
                let input_size: usize = inputshape.iter().product();
                let ops = input_size * units;
                let base = self
                    .operation_latencies
                    .get("dense")
                    .copied()
                    .unwrap_or(0.1);
                base * (ops as f64 / 1e6) // Normalize by million operations
            LayerType::Conv2D {
                filters,
                kernel_size,
                stride,
            } => {
                if inputshape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Conv2D requires 3D input".to_string(),
                    ));
                }
                let h = inputshape[0];
                let w = inputshape[1];
                let c = inputshape[2];
                let output_h = (h - kernel_size.0) / stride.0 + 1;
                let output_w = (w - kernel_size.1) / stride.1 + 1;
                let ops = output_h * output_w * filters * kernel_size.0 * kernel_size.1 * c;
                    .get("conv2d")
                    .unwrap_or(0.5);
                base * (ops as f64 / 1e6)
            LayerType::MaxPool2D { pool_size, stride }
            | LayerType::AvgPool2D { pool_size, stride } => {
                    return Ok(0.0);
                let output_h = (h - pool_size.0) / stride.0 + 1;
                let output_w = (w - pool_size.1) / stride.1 + 1;
                let ops = output_h * output_w * c * pool_size.0 * pool_size.1;
                    .get("pooling")
                    .unwrap_or(0.05);
            LayerType::Activation(_) => {
                let ops: usize = inputshape.iter().product();
                    .get("activation")
                    .unwrap_or(0.01);
            LayerType::BatchNorm | LayerType::LayerNorm => {
                    .get("batchnorm")
                    .unwrap_or(0.02);
            LayerType::Dropout(_) => 0.001, // Minimal cost
            _ => 0.01,                      // Default small cost for unknown layers
        // Apply parallelization factor
        let parallelization = match layer {
            LayerType::Dense(_) => self
                .parallelization_factors
                .get("dense")
                .copied()
                .unwrap_or(1.0),
            LayerType::Conv2D { .. } => self
                .get("conv2d")
            _ => 1.0,
        Ok(base_latency * parallelization)
    /// Compute output shape after a layer
    fn compute_outputshape(&self, layer: &LayerType, inputshape: &[usize]) -> Result<Vec<usize>> {
        match layer {
            LayerType::Dense(units) => Ok(vec![*units]),
                let h = (inputshape[0] - kernel_size.0) / stride.0 + 1;
                let w = (inputshape[1] - kernel_size.1) / stride.1 + 1;
                Ok(vec![h, w, *filters])
                    return Ok(inputshape.to_vec());
                let h = (inputshape[0] - pool_size.0) / stride.0 + 1;
                let w = (inputshape[1] - pool_size.1) / stride.1 + 1;
                Ok(vec![h, w, inputshape[2]])
            LayerType::Flatten => {
                let total_size: usize = inputshape.iter().product();
                Ok(vec![total_size])
            _ => Ok(inputshape.to_vec()), // Most layers preserve shape
    /// Compute initialization overhead
    fn compute_initialization_overhead(&self, architecture: &Architecture) -> Result<f64> {
        let num_layers = architecture.layers.len() as f64;
        let base_overhead = match self.platform {
            HardwarePlatform::CPU => 5.0,
            HardwarePlatform::GPU => 20.0,
            HardwarePlatform::MobileARM => 10.0,
            HardwarePlatform::EdgeTPU => 15.0_ => 8.0,
        Ok(base_overhead + num_layers * 0.5)
/// Memory usage predictor
pub struct MemoryPredictor {
impl MemoryPredictor {
    /// Create a new memory predictor
        Self { platform }
    /// Predict memory usage for an architecture
    pub fn predict_memory_usage(
        let mut total_memory = 0.0;
        // Model weights memory
        total_memory += self.compute_weights_memory(architecture)?;
        // Activations memory (peak usage)
        let mut max_activation_memory = 0.0;
            let activation_memory = self.compute_activation_memory(&currentshape)?;
            max_activation_memory = max_activation_memory.max(activation_memory);
        total_memory += max_activation_memory;
        // Add overhead for framework and OS
        total_memory += self.compute_memory_overhead()?;
        Ok(total_memory)
    /// Compute memory usage for model weights
    fn compute_weights_memory(&self, architecture: &Architecture) -> Result<f64> {
        let mut weights_memory = 0.0;
            let layer_params = match layer {
                LayerType::Dense(units) => {
                    // Assume previous layer had 1024 units as approximation
                    1024 * units + units // weights + bias
                LayerType::Conv2D {
                    filters,
                    kernel_size,
                    ..
                } => {
                    // Assume input has 64 channels as approximation
                    filters * kernel_size.0 * kernel_size.1 * 64 + filters
                LayerType::BatchNorm => 128 * 4, // gamma, beta, mean, variance
                LayerType::Embedding {
                    vocab_size,
                    embedding_dim,
                } => vocab_size * embedding_dim_ => 0, // No parameters for other layers
            };
            // 4 bytes per float32 parameter
            weights_memory += layer_params as f64 * 4.0;
        // Convert to MB
        Ok(weights_memory / (1024.0 * 1024.0))
    /// Compute memory usage for activations
    fn compute_activation_memory(&self, shape: &[usize]) -> Result<f64> {
        let elements: usize = shape.iter().product();
        // 4 bytes per float32 activation
        let memory_bytes = elements as f64 * 4.0;
        Ok(memory_bytes / (1024.0 * 1024.0))
    /// Compute memory overhead
    fn compute_memory_overhead(&self) -> Result<f64> {
        let overhead_mb = match self.platform {
            HardwarePlatform::CPU => 50.0,
            HardwarePlatform::GPU => 100.0,
            HardwarePlatform::MobileARM => 25.0,
            HardwarePlatform::EdgeTPU => 30.0_ => 40.0,
        Ok(overhead_mb)
    /// Compute output shape (simplified version)
            _ => Ok(inputshape.to_vec()),
/// Energy consumption predictor
pub struct EnergyPredictor {
    power_characteristics: HashMap<String, f64>, // mW per operation
impl EnergyPredictor {
    /// Create a new energy predictor
            power_characteristics: HashMap::new(),
        predictor.initialize_power_characteristics();
    /// Initialize platform-specific power characteristics
    fn initialize_power_characteristics(&mut self) {
                self.power_characteristics
                    .insert("dense".to_string(), 100.0);
                    .insert("conv2d".to_string(), 150.0);
                    .insert("memory".to_string(), 50.0);
                    .insert("dense".to_string(), 200.0);
                    .insert("conv2d".to_string(), 300.0);
                    .insert("memory".to_string(), 80.0);
                self.power_characteristics.insert("dense".to_string(), 30.0);
                    .insert("conv2d".to_string(), 50.0);
                    .insert("memory".to_string(), 15.0);
                self.power_characteristics.insert("dense".to_string(), 75.0);
                    .insert("conv2d".to_string(), 100.0);
                    .insert("memory".to_string(), 30.0);
    /// Predict energy consumption for an architecture
    pub fn predict_energy(&self, architecture: &Architecture, latencyms: f64) -> Result<f64> {
        let mut total_energy = 0.0;
        // Compute dynamic energy (computation)
            let layer_energy = self.compute_layer_energy(layer)?;
            total_energy += layer_energy;
        // Add static energy (idle power during inference)
        let static_power = self.get_static_power();
        total_energy += static_power * latency_ms;
        Ok(total_energy)
    /// Compute energy for a single layer
    fn compute_layer_energy(&self, layer: &LayerType) -> Result<f64> {
        let power = match layer {
                .power_characteristics
                .unwrap_or(100.0),
                .unwrap_or(150.0, _ => 10.0, // Low energy for other operations
        // Assume 1ms operation time for energy calculation
        Ok(power * 1.0)
    /// Get static power consumption
    fn get_static_power(&self) -> f64 {
            HardwarePlatform::CPU => 1000.0,      // 1W
            HardwarePlatform::GPU => 5000.0,      // 5W
            HardwarePlatform::MobileARM => 500.0, // 0.5W
            HardwarePlatform::EdgeTPU => 2000.0,  // 2W
            _ => 1500.0,                          // 1.5W default
/// Hardware-aware search implementation
pub struct HardwareAwareSearch {
    constraints: HardwareConstraints,
    latency_predictor: LatencyPredictor,
    memory_predictor: MemoryPredictor,
    energy_predictor: EnergyPredictor,
    constraint_weights: HashMap<String, f64>,
    violation_history: Vec<HashMap<String, f64>>,
impl HardwareAwareSearch {
    /// Create a new hardware-aware search
    pub fn new(constraints: HardwareConstraints) -> Self {
        let latency_predictor = LatencyPredictor::new(_constraints.platform.clone());
        let memory_predictor = MemoryPredictor::new(_constraints.platform.clone());
        let energy_predictor = EnergyPredictor::new(_constraints.platform.clone());
        let mut constraint_weights = HashMap::new();
        constraintweights.insert("latency".to_string(), 0.3);
        constraintweights.insert("memory".to_string(), 0.25);
        constraintweights.insert("energy".to_string(), 0.2);
        constraintweights.insert("model_size".to_string(), 0.15);
        constraintweights.insert("throughput".to_string(), 0.1);
            constraints,
            latency_predictor,
            memory_predictor,
            energy_predictor,
            constraint_weights,
            violation_history: Vec::new(),
    /// Evaluate hardware _constraints for an architecture
    pub fn evaluate_constraints(
        &mut self,
        architecture: &Arc<dyn ArchitectureEncoding>,
    ) -> Result<HashMap<String, f64>> {
        let arch = architecture.to_architecture()?;
        let mut violations = HashMap::new();
        // Predict latency
        let predicted_latency = self.latency_predictor.predict_latency(&arch, inputshape)?;
        if let Some(max_latency) = self._constraints.max_latency_ms {
            let violation = (predicted_latency - max_latency).max(0.0) / max_latency;
            violations.insert("latency".to_string(), violation);
        // Predict memory usage
        let predicted_memory = self
            .memory_predictor
            .predict_memory_usage(&arch, inputshape)?;
        if let Some(max_memory) = self._constraints.max_memory_mb {
            let violation = (predicted_memory - max_memory).max(0.0) / max_memory;
            violations.insert("memory".to_string(), violation);
        // Predict energy consumption
        let predicted_energy = self
            .energy_predictor
            .predict_energy(&arch, predicted_latency)?;
        if let Some(max_energy) = self._constraints.max_energy_mj {
            let violation = (predicted_energy - max_energy).max(0.0) / max_energy;
            violations.insert("energy".to_string(), violation);
        // Estimate model size (simplified)
        let model_size = self.estimate_model_size(&arch)?;
        if let Some(max_size) = self._constraints.max_model_size_mb {
            let violation = (model_size - max_size).max(0.0) / max_size;
            violations.insert("model_size".to_string(), violation);
        // Estimate throughput
        let throughput = 1000.0 / predicted_latency; // inferences per second
        if let Some(min_throughput) = self._constraints.min_throughput {
            let violation = (min_throughput - throughput).max(0.0) / min_throughput;
            violations.insert("throughput".to_string(), violation);
        // Store violation history
        self.violation_history.push(violations.clone());
        Ok(violations)
    /// Compute weighted constraint violation score
    pub fn compute_constraint_score(&self, violations: &HashMap<String, f64>) -> f64 {
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        for (constraint, &violation) in violations {
            if let Some(&weight) = self.constraintweights.get(constraint) {
                weighted_score += violation * weight;
                total_weight += weight;
        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
    /// Check if architecture satisfies all constraints
    pub fn satisfies_constraints(&self, violations: &HashMap<String, f64>) -> bool {
        violations.values().all(|&violation| violation <= 0.0)
    /// Get constraint satisfaction rate
    pub fn get_satisfaction_rate(&self) -> f64 {
        if self.violation_history.is_empty() {
            return 1.0;
        let satisfied_count = self
            .violation_history
            .iter()
            .filter(|violations| self.satisfies_constraints(violations))
            .count();
        satisfied_count as f64 / self.violation_history.len() as f64
    /// Generate optimization suggestions
    pub fn generate_optimization_suggestions(
        violations: &HashMap<String, f64>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
            if violation > 0.1 {
                // Significant violation
                match constraint.as_str() {
                    "latency" => {
                        suggestions.push("Consider reducing model depth or width".to_string());
                        suggestions.push("Use depthwise separable convolutions".to_string());
                        suggestions.push("Apply model pruning or quantization".to_string());
                    }
                    "memory" => {
                        suggestions.push("Reduce batch size or model parameters".to_string());
                        suggestions.push("Use gradient checkpointing".to_string());
                        suggestions.push("Apply weight sharing techniques".to_string());
                    "energy" => {
                        suggestions.push("Reduce computational complexity".to_string());
                        suggestions.push("Use low-power layer types".to_string());
                        suggestions.push("Optimize data movement patterns".to_string());
                    "model_size" => {
                        suggestions.push("Apply model compression techniques".to_string());
                        suggestions.push("Use knowledge distillation".to_string());
                        suggestions.push("Reduce parameter precision".to_string());
                    "throughput" => {
                        suggestions.push("Optimize for batch processing".to_string());
                        suggestions.push("Use pipeline parallelism".to_string());
                        suggestions.push("Consider model ensemble reduction".to_string());
                    _ => {}
        suggestions
    /// Estimate model size in MB
    fn estimate_model_size(&self, architecture: &Architecture) -> Result<f64> {
        let mut total_params = 0;
                    // Assume previous layer had 1024 units
                    1024 * units + units
                    // Assume 64 input channels
                _ => 0,
            total_params += layer_params;
        // 4 bytes per float32 parameter, convert to MB
        Ok(total_params as f64 * 4.0 / (1024.0 * 1024.0))
    /// Update constraint weights based on violation history
    pub fn adapt_constraint_weights(&mut self) {
        if self.violation_history.len() < 10 {
            return; // Need enough history
        // Analyze recent violations
        let recent_violations = &self.violation_history[self.violation_history.len() - 10..];
        let mut avg_violations = HashMap::new();
        for violations in recent_violations {
            for (constraint, &violation) in violations {
                *avg_violations.entry(constraint.clone()).or_insert(0.0) += violation / 10.0;
        // Increase weights for frequently violated constraints
        for (constraint, avg_violation) in avg_violations {
            if avg_violation > 0.1 {
                if let Some(weight) = self.constraintweights.get_mut(&constraint) {
                    *weight = (*weight * 1.1).min(0.5); // Cap at 50%
        // Normalize weights
        let total_weight: f64 = self.constraintweights.values().sum();
            for weight in self.constraintweights.values_mut() {
                *weight /= total_weight;
    /// Get platform characteristics summary
    pub fn get_platform_summary(&self) -> String {
        format!(
            "Platform: {:?}\nMax Latency: {:?} ms\nMax Memory: {:?} MB\nMax Energy: {:?} mJ\nCompute Units: {}",
            self.constraints.platform,
            self.constraints.max_latency_ms,
            self.constraints.max_memory_mb,
            self.constraints.max_energy_mj,
            self.constraints.compute_units
        )
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search__space::Architecture;
    #[test]
    fn test_hardware_constraints_default() {
        let constraints = HardwareConstraints::default();
        assert_eq!(constraints.platform, HardwarePlatform::CPU);
        assert!(constraints.max_latency_ms.is_some());
    fn test_latency_predictor() {
        let predictor = LatencyPredictor::new(HardwarePlatform::CPU);
        let architecture = Architecture {
            layers: vec![
                LayerType::Dense(128),
                LayerType::Activation("relu".to_string()),
                LayerType::Dense(10),
            ],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        let latency = predictor
            .predict_latency(&architecture, &[32, 32, 3])
            .unwrap();
        assert!(latency > 0.0);
    fn test_memory_predictor() {
        let predictor = MemoryPredictor::new(HardwarePlatform::GPU);
            layers: vec![LayerType::Dense(256), LayerType::Dense(10)],
        let memory = predictor
            .predict_memory_usage(&architecture, &[784])
        assert!(memory > 0.0);
    fn test_hardware_aware_search() {
        let mut search = HardwareAwareSearch::new(constraints);
        let arch = Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
            vec![LayerType::Dense(128), LayerType::Dense(10)],
        ));
        let violations = search.evaluate_constraints(&arch, &[784]).unwrap();
        assert!(!violations.is_empty());
        let score = search.compute_constraint_score(&violations);
        assert!(score >= 0.0);
