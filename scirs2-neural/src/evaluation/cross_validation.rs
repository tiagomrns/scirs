//! Cross-validation utilities
//!
//! This module provides utilities for cross-validation of models.

use super::{EvaluationConfig, Evaluator, MetricType, ModelBuilder};
use crate::data::Dataset;
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use rand::SeedableRng;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
/// Cross-validation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold(usize),
    /// Stratified K-fold cross-validation
    StratifiedKFold(usize),
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Leave-P-out cross-validation
    LeavePOut(usize),
    /// Random shuffling cross-validation
    ShuffleSplit(usize, f64),
}
/// Configuration for cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Cross-validation strategy
    pub strategy: CrossValidationStrategy,
    /// Whether to shuffle the data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_seed: Option<u64>,
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Metrics to compute during evaluation
    pub metrics: Vec<MetricType>,
    /// Verbosity level
    pub verbose: usize,
impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            strategy: CrossValidationStrategy::KFold(5),
            shuffle: true,
            random_seed: None,
            batch_size: 32,
            num_workers: 0,
            metrics: vec![MetricType::Loss, MetricType::Accuracy],
            verbose: 1,
        }
    }
/// Cross-validation fold
#[derive(Debug)]
pub struct CrossValidationFold {
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Validation indices
    pub val_indices: Vec<usize>,
/// Cross-validator for model evaluation
pub struct CrossValidator<
    F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync,
> {
    /// Configuration for cross-validation
    pub config: CrossValidationConfig,
    /// Evaluator for validation metrics
    evaluator: Evaluator<F>,
impl<F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync>
    CrossValidator<F>
{
    /// Create a new cross-validator
    pub fn new(config: CrossValidationConfig) -> Result<Self> {
        // Create evaluator
        let eval_config = EvaluationConfig {
            batch_size: config.batch_size,
            shuffle: false, // We handle shuffling during fold creation
            num_workers: config.num_workers,
            metrics: config.metrics.clone(),
            steps: None,
            verbose: config.verbose,
        };
        let evaluator = Evaluator::new(eval_config)?;
        Ok(Self { config, evaluator })
    /// Generate cross-validation folds
    pub fn create_folds(&self, dataset: &dyn Dataset<F>) -> Result<Vec<CrossValidationFold>> {
        let n_samples = dataset.len();
        match self.config.strategy {
            CrossValidationStrategy::KFold(k) => {
                if k < 2 {
                    return Err(NeuralError::ValidationError(
                        "k must be at least 2".to_string(),
                    ));
                }
                if n_samples < k {
                    return Err(NeuralError::ValidationError(format!(
                        "Dataset size ({}) must be at least equal to k ({})",
                        n_samples, k
                    )));
                // Create indices
                let mut indices: Vec<usize> = (0..n_samples).collect();
                // Shuffle if required
                if self.config.shuffle {
                    use rand::seq::SliceRandom;
                    if let Some(seed) = self.config.random_seed {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                        indices.shuffle(&mut rng);
                    } else {
                        let mut rng = rng();
                    }
                // Calculate fold sizes
                let fold_size = n_samples / k;
                let remainder = n_samples % k;
                // Create folds
                let mut folds = Vec::with_capacity(k);
                let mut start = 0;
                for i in 0..k {
                    // Adjust fold size to distribute remainder
                    let fold_size_adjusted = if i < remainder {
                        fold_size + 1
                        fold_size
                    };
                    let end = start + fold_size_adjusted;
                    // Validation indices for this fold
                    let val_indices = indices[start..end].to_vec();
                    // Training indices (all except validation)
                    let mut train_indices = Vec::with_capacity(n_samples - val_indices.len());
                    for &idx in &indices[0..start] {
                        train_indices.push(idx);
                    for &idx in &indices[end..] {
                    folds.push(CrossValidationFold {
                        train_indices,
                        val_indices,
                    });
                    start = end;
                Ok(folds)
            }
            CrossValidationStrategy::StratifiedKFold(k) => {
                // For stratified k-fold, we need class labels
                // This is a simplified implementation that assumes targets are class indices
                // Get class labels for each sample
                let mut class_indices = HashMap::new();
                for i in 0..n_samples {
                    let (_, target) = dataset.get(i)?;
                    // Extract class index (assuming target is a class index)
                    let class_idx = if target.ndim() > 1 && target.shape()[1] > 1 {
                        // One-hot encoded: find max index
                        let mut max_idx = 0;
                        let mut max_val = target[[0, 0]];
                        for j in 1..target.shape()[1] {
                            if target[[0, j]] > max_val {
                                max_idx = j;
                                max_val = target[[0, j]];
                            }
                        }
                        max_idx
                        // Direct class index
                        target[[0]].to_usize().unwrap_or(0)
                    class_indices
                        .entry(class_idx)
                        .or_insert_with(Vec::new)
                        .push(i);
                // Create folds with stratification
                for _ in 0..k {
                        train_indices: Vec::new(),
                        val_indices: Vec::new(),
                // Distribute indices by class
                for (_, mut indices) in class_indices {
                    // Shuffle indices within class
                    if self.config.shuffle {
                        use rand::seq::SliceRandom;
                        if let Some(seed) = self.config.random_seed {
                            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                            indices.shuffle(&mut rng);
                        } else {
                            let mut rng = rng();
                    // Distribute indices to folds
                    for (i, &idx) in indices.iter().enumerate() {
                        let fold_idx = i % k;
                        folds[fold_idx].val_indices.push(idx);
                // Fill in training indices
                for fold in folds.iter_mut().take(k) {
                    let val_indices = &fold.val_indices;
                    for i in 0..n_samples {
                        if !val_indices.contains(&i) {
                            train_indices.push(i);
                    fold.train_indices = train_indices;
            CrossValidationStrategy::LeaveOneOut => {
                // Leave-one-out: each sample becomes a fold
                let mut folds = Vec::with_capacity(n_samples);
                    let val_indices = vec![i];
                    let mut train_indices = Vec::with_capacity(n_samples - 1);
                    for j in 0..n_samples {
                        if j != i {
                            train_indices.push(j);
            CrossValidationStrategy::LeavePOut(p) => {
                if p >= n_samples {
                        "p ({}) must be less than dataset size ({})",
                        p, n_samples
                // This is a simplified implementation that doesn't generate all possible folds
                // as that would be combinatorial and potentially too large
                // Create n-p folds with p samples each
                let n_folds = n_samples / p;
                let mut folds = Vec::with_capacity(n_folds);
                for i in 0..n_folds {
                    let start = i * p;
                    let end = (i + 1) * p;
                    let mut train_indices = Vec::with_capacity(n_samples - p);
                    for (j, &idx) in indices.iter().enumerate().take(n_samples) {
                        if j < start || j >= end {
                            train_indices.push(idx);
            CrossValidationStrategy::ShuffleSplit(n_splits, test_size) => {
                if test_size <= 0.0 || test_size >= 1.0 {
                        "test_size must be between 0 and 1".to_string(),
                let indices: Vec<usize> = (0..n_samples).collect();
                // Calculate test set size
                let test_count = (n_samples as f64 * test_size).ceil() as usize;
                if test_count >= n_samples {
                        "test_size too large for dataset".to_string(),
                let mut folds = Vec::with_capacity(n_splits);
                let rng_with_seed = self
                    .config
                    .random_seed
                    .map(rand::rngs::StdRng::from_seed);
                for _ in 0..n_splits {
                    // Shuffle indices
                    let mut shuffled = indices.clone();
                        if let Some(mut rng) = rng_with_seed.clone() {
                            shuffled.shuffle(&mut rng);
                    // Split into train and validation
                    let val_indices = shuffled[0..test_count].to_vec();
                    let train_indices = shuffled[test_count..].to_vec();
    /// Perform cross-validation on a model builder and dataset
    pub fn cross_validate<L: Layer<F> + Clone>(
        &mut self,
        model_builder: &dyn ModelBuilder<F, Model = L>,
        dataset: &dyn Dataset<F>,
        loss_fn: Option<&dyn crate::losses::Loss<F>>,
    ) -> Result<HashMap<String, Vec<F>>> {
        // Generate folds
        let folds = self.create_folds(dataset)?;
        // Initialize results
        let metrics = &self.config.metrics;
        let mut results: HashMap<String, Vec<F>> = metrics
            .iter()
            .map(|m| {
                let name = match m {
                    MetricType::Loss => "loss".to_string(),
                    MetricType::Accuracy => "accuracy".to_string(),
                    MetricType::Precision => "precision".to_string(),
                    MetricType::Recall => "recall".to_string(),
                    MetricType::F1Score => "f1_score".to_string(),
                    MetricType::MeanSquaredError => "mse".to_string(),
                    MetricType::MeanAbsoluteError => "mae".to_string(),
                    MetricType::RSquared => "r2".to_string(),
                    MetricType::AUC => "auc".to_string(),
                    MetricType::Custom(name) => name.clone(),
                };
                (name, Vec::with_capacity(folds.len()))
            })
            .collect();
        // Perform cross-validation
        for (fold_idx, fold) in folds.iter().enumerate() {
            if self.config.verbose > 0 {
                println!("Fold {}/{}", fold_idx + 1, folds.len());
            // Create train and validation datasets
            // Instead of using a DatasetView with references, we'll create an owned version
            struct DatasetSubset<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> {
                data: Vec<(
                    ndarray::Array<F, ndarray::IxDyn>,
                )>,
            impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> DatasetSubset<F> {
                fn new(dataset: &dyn Dataset<F>, indices: &[usize]) -> Result<Self> {
                    let mut data = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        let (input, target) = dataset.get(idx)?;
                        data.push((input, target));
                    Ok(Self { data })
            impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> Dataset<F>
                for DatasetSubset<F>
            {
                fn len(&self) -> usize {
                    self.data.len()
                fn get(
                    &self,
                    idx: usize,
                ) -> Result<(
                )> {
                    if idx >= self.data.len() {
                        return Err(crate::error::NeuralError::InferenceError(format!(
                            "Index out of bounds: {} >= {}",
                            idx,
                            self.data.len()
                        )));
                    Ok((self.data[idx].0.clone(), self.data[idx].1.clone()))
                fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
                    // Create a new DatasetSubset with cloned data
                    let cloned_data = self.data.clone();
                    Box::new(Self { data: cloned_data })
            let _train_dataset = DatasetSubset::new(dataset, &fold.train_indices)?;
            let val_dataset = DatasetSubset::new(dataset, &fold.val_indices)?;
            // Build model
            let model = model_builder.build()?;
            // Evaluate on validation set
            let fold_metrics = self.evaluator.evaluate(&model, &val_dataset, loss_fn)?;
            // Store results
            for (name, value) in fold_metrics {
                if let Some(values) = results.get_mut(&name) {
                    values.push(value);
        // Calculate mean and std of metrics
        for (name, values) in &results {
            if !values.is_empty() {
                // Calculate mean
                let sum = values.iter().fold(F::zero(), |acc, &x| acc + x);
                let mean = sum / F::from(values.len()).unwrap();
                // Calculate std
                let variance_sum = values
                    .iter()
                    .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean));
                let std = (variance_sum / F::from(values.len()).unwrap()).sqrt();
                if self.config.verbose > 0 {
                    println!("{}: {:.4} ± {:.4}", name, mean, std);
        Ok(results)
