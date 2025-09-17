//! Progressive Neural Architecture Search
//!
//! This module implements progressive search strategies that adaptively expand
//! the search space and improve search efficiency over time.

use crate::error::Result;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding, search_space::LayerType, SearchResult,
    SearchSpace, SearchSpaceConfig,
};
use std::sync::Arc;
/// Configuration for progressive search
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Initial search space configuration
    pub initial_search_space: SearchSpaceConfig,
    /// Number of stages in progressive search
    pub num_stages: usize,
    /// Architectures to evaluate per stage
    pub architectures_per_stage: usize,
    /// Expansion strategy for search space
    pub expansion_strategy: ExpansionStrategy,
    /// Threshold for advancing to next stage
    pub advancement_threshold: f64,
    /// Maximum complexity increase per stage
    pub max_complexity_increase: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}
impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            initial_search_space: SearchSpaceConfig::default(),
            num_stages: 5,
            architectures_per_stage: 50,
            expansion_strategy: ExpansionStrategy::AdaptiveComplexity,
            advancement_threshold: 0.02,  // 2% improvement to advance
            max_complexity_increase: 0.5, // 50% complexity increase per stage
            early_stopping_patience: 2,
        }
    }
/// Strategies for expanding the search space
pub enum ExpansionStrategy {
    /// Gradually increase model complexity
    AdaptiveComplexity,
    /// Add new layer types progressively
    LayerTypeExpansion,
    /// Increase depth and width multipliers
    ScaleExpansion,
    /// Add skip connections gradually
    ConnectionExpansion,
    /// Composite strategy combining multiple approaches
    Composite(Vec<ExpansionStrategy>),
/// Progressive search implementation
pub struct ProgressiveSearch {
    config: ProgressiveConfig,
    current_stage: usize,
    search_spaces: Vec<SearchSpace>,
    stage_results: Vec<Vec<SearchResult>>,
    best_per_stage: Vec<Option<Arc<dyn ArchitectureEncoding>>>,
    complexity_history: Vec<f64>,
    performance_history: Vec<f64>,
    stagnation_counter: usize,
impl ProgressiveSearch {
    /// Create a new progressive search
    pub fn new(config: ProgressiveConfig) -> Result<Self> {
        let initial_space = SearchSpace::new(_config.initial_search_space.clone())?;
        Ok(Self {
            config,
            current_stage: 0,
            search_spaces: vec![initial_space],
            stage_results: vec![Vec::new()],
            best_per_stage: vec![None],
            complexity_history: Vec::new(),
            performance_history: Vec::new(),
            stagnation_counter: 0,
        })
    /// Get current search space
    pub fn current_search_space(&self) -> &SearchSpace {
        &self.search_spaces[self.current_stage]
    /// Add results from current stage
    pub fn add_stage_results(&mut self, results: Vec<SearchResult>) -> Result<()> {
        // Ensure we have enough stages
        while self.stage_results.len() <= self.current_stage {
            self.stage_results.push(Vec::new());
        // Add results to current stage
        self.stage_results[self.current_stage].extend(results);
        // Update best architecture for current stage
        if let Some(best_result) = self.get_best_result_in_stage(self.current_stage) {
            self.best_per_stage[self.current_stage] = Some(best_result.architecture.clone());
            // Update performance history
            let performance =
                best_result.metrics.values().sum::<f64>() / best_result.metrics.len() as f64;
            self.performance_history.push(performance);
            // Update complexity history
            let complexity = self.estimate_architecture_complexity(&best_result.architecture)?;
            self.complexity_history.push(complexity);
        Ok(())
    /// Check if ready to advance to next stage
    pub fn should_advance_stage(&self) -> bool {
        if self.current_stage >= self.config.num_stages - 1 {
            return false; // Already at final stage
        // Check if we have enough evaluations in current stage
        let current_evaluations = self
            .stage_results
            .get(self.current_stage)
            .map(|results| results.len())
            .unwrap_or(0);
        if current_evaluations < self.config.architectures_per_stage {
            return false;
        // Check for performance improvement
        if self.performance_history.len() >= 2 {
            let current_performance = self.performance_history.last().copied().unwrap_or(0.0);
            let previous_performance = self.performance_history[self.performance_history.len() - 2];
            let improvement =
                (current_performance - previous_performance) / previous_performance.abs();
            if improvement >= self.config.advancement_threshold {
                return true;
            }
        // Check for stagnation
        self.stagnation_counter >= self.config.early_stopping_patience
    /// Advance to the next stage
    pub fn advance_stage(&mut self) -> Result<()> {
            return Ok(()); // Already at final stage
        self.current_stage += 1;
        // Create expanded search space for new stage
        let expanded_space = self.expand_search_space()?;
        self.search_spaces.push(expanded_space);
        // Initialize containers for new stage
        self.stage_results.push(Vec::new());
        self.best_per_stage.push(None);
        // Reset stagnation counter
        self.stagnation_counter = 0;
    /// Expand search space according to strategy
    fn expand_search_space(&self) -> Result<SearchSpace> {
        let mut expanded_config = self.config.initial_search_space.clone();
        // Apply expansion based on current stage and strategy
        match &self.config.expansion_strategy {
            ExpansionStrategy::AdaptiveComplexity => {
                self.expand_by_complexity(&mut expanded_config)?;
            ExpansionStrategy::LayerTypeExpansion => {
                self.expand_by_layer_types(&mut expanded_config)?;
            ExpansionStrategy::ScaleExpansion => {
                self.expand_by_scale(&mut expanded_config)?;
            ExpansionStrategy::ConnectionExpansion => {
                self.expand_by_connections(&mut expanded_config)?;
            ExpansionStrategy::Composite(strategies) => {
                for strategy in strategies {
                    match strategy {
                        ExpansionStrategy::AdaptiveComplexity => {
                            self.expand_by_complexity(&mut expanded_config)?
                        }
                        ExpansionStrategy::LayerTypeExpansion => {
                            self.expand_by_layer_types(&mut expanded_config)?
                        ExpansionStrategy::ScaleExpansion => {
                            self.expand_by_scale(&mut expanded_config)?
                        ExpansionStrategy::ConnectionExpansion => {
                            self.expand_by_connections(&mut expanded_config)?
                        _ => {} // Avoid infinite recursion
                    }
                }
        SearchSpace::new(expanded_config)
    /// Expand search space by increasing complexity
    fn expand_by_complexity(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        let complexity_factor = 1.0
            + (self.current_stage as f64 * self.config.max_complexity_increase
                / self.config.num_stages as f64);
        // Increase layer size options
        let mut new_layer_types = Vec::new();
        for layer_type in &config.layer_types {
            match layer_type {
                LayerType::Dense(units) => {
                    let new_units = (*units as f64 * complexity_factor) as usize;
                    new_layer_types.push(LayerType::Dense(new_units));
                LayerType::Conv2D {
                    filters,
                    kernel_size,
                    stride,
                } => {
                    let new_filters = (*filters as f64 * complexity_factor) as usize;
                    new_layer_types.push(LayerType::Conv2D {
                        filters: new_filters,
                        kernel_size: *kernel_size,
                        stride: *stride,
                    });
                other => new_layer_types.push(other.clone()),
        // Add original layer types too for diversity
        new_layer_types.extend(config.layer_types.clone());
        config.layer_types = new_layer_types;
        // Increase maximum layers
        config.max_layers = (config.max_layers as f64 * complexity_factor) as usize;
    /// Expand search space by adding new layer types
    fn expand_by_layer_types(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        // Add more advanced layer types based on stage
        match self.current_stage {
            1 => {
                // Add attention and normalization layers
                config.layer_types.push(LayerType::Attention {
                    num_heads: 4,
                    key_dim: 64,
                });
                config.layer_types.push(LayerType::LayerNorm);
            2 => {
                // Add recurrent layers
                config.layer_types.push(LayerType::LSTM {
                    units: 128,
                    return_sequences: false,
                config.layer_types.push(LayerType::GRU {
            3 => {
                // Add more complex convolutions
                config.layer_types.push(LayerType::Conv2D {
                    filters: 128,
                    kernel_size: (5, 5),
                    stride: (1, 1),
                    filters: 256,
                    kernel_size: (7, 7, _ => {
                // Add embedding and reshape layers
                config.layer_types.push(LayerType::Embedding {
                    vocab_size: 10000,
                    embedding_dim: 128,
                config.layer_types.push(LayerType::Reshape(vec![-1, 64]));
    /// Expand search space by scaling factors
    fn expand_by_scale(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        let scale_factor = 1.0 + (self.current_stage as f32 * 0.25);
        // Add new width multipliers
        for &base_mult in &[0.5, 0.75, 1.0, 1.25, 1.5] {
            let new_mult = base_mult * scale_factor;
            if !config.width_multipliers.contains(&new_mult) {
                config.width_multipliers.push(new_mult);
        // Add new depth multipliers
            if !config.depth_multipliers.contains(&new_mult) {
                config.depth_multipliers.push(new_mult);
    /// Expand search space by enabling more connections
    fn expand_by_connections(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        // Enable branches if not already enabled
        config.allow_branches = true;
        // Increase skip connection probability
        config.skip_connection_prob = (config.skip_connection_prob + 0.1).min(0.8);
        // Increase maximum branches
        config.max_branches = (config.max_branches + 1).min(5);
    /// Get best result in a specific stage
    fn get_best_result_in_stage(&self, stage: usize) -> Option<&SearchResult> {
        self.stage_results.get(stage)?.iter().max_by(|a, b| {
            let a_score = a.metrics.values().sum::<f64>() / a.metrics.len() as f64;
            let b_score = b.metrics.values().sum::<f64>() / b.metrics.len() as f64;
            a_score.partial_cmp(&b_score).unwrap()
    /// Estimate architecture complexity
    fn estimate_architecture_complexity(
        &self,
        architecture: &Arc<dyn ArchitectureEncoding>,
    ) -> Result<f64> {
        // Convert to vector and use its magnitude as complexity measure
        let encoding_vector = architecture.to_vector();
        let complexity = encoding_vector.iter().map(|x| x.abs()).sum::<f64>();
        // Normalize by vector length
        Ok(complexity / encoding_vector.len() as f64)
    /// Get current stage number
    pub fn current_stage(&self) -> usize {
        self.current_stage
    /// Get total number of stages
    pub fn total_stages(&self) -> usize {
        self.config.num_stages
    /// Get all stage results
    pub fn get_all_results(&self) -> &[Vec<SearchResult>] {
        &self.stage_results
    /// Get best architecture from all stages
    pub fn get_global_best(&self) -> Option<&Arc<dyn ArchitectureEncoding>> {
        let mut best_performance = f64::NEG_INFINITY;
        let mut best_arch = None;
        for stage_results in &self.stage_results {
            if let Some(best_in_stage) = stage_results.iter().max_by(|a, b| {
                let a_score = a.metrics.values().sum::<f64>() / a.metrics.len() as f64;
                let b_score = b.metrics.values().sum::<f64>() / b.metrics.len() as f64;
                a_score.partial_cmp(&b_score).unwrap()
            }) {
                let performance = best_in_stage.metrics.values().sum::<f64>()
                    / best_in_stage.metrics.len() as f64;
                if performance > best_performance {
                    best_performance = performance;
                    best_arch = Some(&best_in_stage.architecture);
        best_arch
    /// Get performance trend across stages
    pub fn get_performance_trend(&self) -> &[f64] {
        &self.performance_history
    /// Get complexity trend across stages
    pub fn get_complexity_trend(&self) -> &[f64] {
        &self.complexity_history
    /// Check if search has converged
    pub fn has_converged(&self) -> bool {
        if self.performance_history.len() < 3 {
        // Check if performance improvement has stagnated
        let recent_improvements: Vec<f64> = self
            .performance_history
            .windows(2)
            .take(3)
            .map(|window| (window[1] - window[0]) / window[0].abs())
            .collect();
        let avg_improvement =
            recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64;
        avg_improvement < self.config.advancement_threshold / 2.0
    /// Generate search space evolution report
    pub fn generate_evolution_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Progressive Search Evolution Report\n\n");
        report.push_str(&format!(
            "Current Stage: {}/{}\n",
            self.current_stage + 1,
            self.config.num_stages
        ));
            "Total Evaluations: {}\n",
            self.stage_results.iter().map(|r| r.len()).sum::<usize>()
        report.push_str("\n## Performance Evolution\n");
        for (stage, &performance) in self.performance_history.iter().enumerate() {
            report.push_str(&format!("Stage {}: {:.4}\n", stage + 1, performance));
        report.push_str("\n## Complexity Evolution\n");
        for (stage, &complexity) in self.complexity_history.iter().enumerate() {
            report.push_str(&format!("Stage {}: {:.4}\n", stage + 1, complexity));
        if let Some(best_arch) = self.get_global_best() {
            report.push_str("\n## Best Architecture Found\n");
            report.push_str(&format!("{}\n", best_arch.to_string()));
        report.push_str(&format!("\n## Convergence Status\n"));
        report.push_str(&format!("Converged: {}\n", self.has_converged()));
            "Stagnation Counter: {}\n",
            self.stagnation_counter
        report
/// Progressive search builder for easier configuration
pub struct ProgressiveSearchBuilder {
impl ProgressiveSearchBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
            config: ProgressiveConfig::default(),
    /// Set number of stages
    pub fn stages(mut self, numstages: usize) -> Self {
        self.config.num_stages = num_stages;
        self
    /// Set architectures per stage
    pub fn architectures_per_stage(mut self, count: usize) -> Self {
        self.config.architectures_per_stage = count;
    /// Set expansion strategy
    pub fn expansion_strategy(mut self, strategy: ExpansionStrategy) -> Self {
        self.config.expansion_strategy = strategy;
    /// Set advancement threshold
    pub fn advancement_threshold(mut self, threshold: f64) -> Self {
        self.config.advancement_threshold = threshold;
    /// Build the progressive search
    pub fn build(self) -> Result<ProgressiveSearch> {
        ProgressiveSearch::new(self.config)
impl Default for ProgressiveSearchBuilder {
        Self::new()
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_progressive_config_default() {
        let config = ProgressiveConfig::default();
        assert_eq!(config.num_stages, 5);
        assert_eq!(config.architectures_per_stage, 50);
    fn test_progressive_search_creation() {
        let search = ProgressiveSearch::new(config).unwrap();
        assert_eq!(search.current_stage(), 0);
        assert_eq!(search.total_stages(), 5);
    fn test_builder_pattern() {
        let search = ProgressiveSearchBuilder::new()
            .stages(3)
            .architectures_per_stage(25)
            .advancement_threshold(0.05)
            .build()
            .unwrap();
        assert_eq!(search.total_stages(), 3);
