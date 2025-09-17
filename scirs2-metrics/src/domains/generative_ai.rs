//! Advanced generative AI and deep learning metrics
//!
//! This module provides comprehensive evaluation metrics for modern deep learning
//! approaches including:
//! - **GAN Evaluation**: Inception Score, FID, KID, IS, LPIPS
//! - **Contrastive Learning**: Uniformity, Alignment, InfoNCE
//! - **Self-Supervised Learning**: Linear probing, clustering metrics, representation quality
//! - **Foundation Models**: Zero-shot evaluation, few-shot performance
//! - **Multimodal Models**: Cross-modal retrieval, alignment metrics

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;
use std::iter::Sum;

/// Comprehensive generative AI metrics suite
pub struct GenerativeAISuite<F: Float> {
    /// GAN evaluation metrics
    pub gan_metrics: GANEvaluationMetrics<F>,
    /// Contrastive learning metrics
    pub contrastive_metrics: ContrastiveLearningMetrics<F>,
    /// Self-supervised learning metrics
    pub ssl_metrics: SelfSupervisedMetrics<F>,
    /// Foundation model metrics
    pub foundation_metrics: FoundationModelMetrics<F>,
    /// Multimodal metrics
    pub multimodal_metrics: MultimodalMetrics<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for GenerativeAISuite<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> GenerativeAISuite<F> {
    /// Create new generative AI metrics suite
    pub fn new() -> Self {
        Self {
            gan_metrics: GANEvaluationMetrics::new(),
            contrastive_metrics: ContrastiveLearningMetrics::new(),
            ssl_metrics: SelfSupervisedMetrics::new(),
            foundation_metrics: FoundationModelMetrics::new(),
            multimodal_metrics: MultimodalMetrics::new(),
        }
    }

    /// Get GAN evaluation metrics
    pub fn gan_metrics(&self) -> &GANEvaluationMetrics<F> {
        &self.gan_metrics
    }

    /// Get contrastive learning metrics
    pub fn contrastive_metrics(&self) -> &ContrastiveLearningMetrics<F> {
        &self.contrastive_metrics
    }

    /// Get self-supervised learning metrics
    pub fn ssl_metrics(&self) -> &SelfSupervisedMetrics<F> {
        &self.ssl_metrics
    }

    /// Get foundation model metrics
    pub fn foundation_metrics(&self) -> &FoundationModelMetrics<F> {
        &self.foundation_metrics
    }

    /// Get multimodal metrics
    pub fn multimodal_metrics(&self) -> &MultimodalMetrics<F> {
        &self.multimodal_metrics
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> DomainMetrics
    for GenerativeAISuite<F>
{
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Generative AI & Deep Learning"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "inception_score",
            "fid_score",
            "kid_score",
            "lpips_distance",
            "uniformity",
            "alignment",
            "infonce_loss",
            "linear_probing_accuracy",
            "clustering_nmi",
            "representation_rank",
            "zero_shot_accuracy",
            "few_shot_accuracy",
            "cross_modal_retrieval",
            "multimodal_alignment",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "inception_score",
            "Inception Score for evaluating GAN quality",
        );
        descriptions.insert(
            "fid_score",
            "Fréchet Inception Distance for distribution comparison",
        );
        descriptions.insert("kid_score", "Kernel Inception Distance for sample quality");
        descriptions.insert(
            "lpips_distance",
            "Learned Perceptual Image Patch Similarity",
        );
        descriptions.insert("uniformity", "Uniformity of learned representations");
        descriptions.insert(
            "alignment",
            "Alignment between positive pairs in contrastive learning",
        );
        descriptions.insert("infonce_loss", "InfoNCE contrastive loss value");
        descriptions.insert(
            "linear_probing_accuracy",
            "Linear probe classification accuracy",
        );
        descriptions.insert(
            "clustering_nmi",
            "Normalized Mutual Information for clustering",
        );
        descriptions.insert(
            "representation_rank",
            "Effective rank of representation matrix",
        );
        descriptions.insert("zero_shot_accuracy", "Zero-shot classification accuracy");
        descriptions.insert("few_shot_accuracy", "Few-shot learning performance");
        descriptions.insert("cross_modal_retrieval", "Cross-modal retrieval performance");
        descriptions.insert(
            "multimodal_alignment",
            "Multimodal representation alignment",
        );
        descriptions
    }
}

/// GAN evaluation metrics
pub struct GANEvaluationMetrics<F: Float> {
    /// Number of inception features to use
    pub n_inception_features: usize,
    /// Number of samples for KID estimation
    pub n_kid_samples: usize,
    /// Enable LPIPS computation
    pub enable_lpips: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for GANEvaluationMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> GANEvaluationMetrics<F> {
    /// Create new GAN evaluation metrics
    pub fn new() -> Self {
        Self {
            n_inception_features: 2048,
            n_kid_samples: 10000,
            enable_lpips: true,
            random_seed: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set inception features dimension
    pub fn with_inception_features(mut self, n: usize) -> Self {
        self.n_inception_features = n;
        self
    }

    /// Set KID sample size
    pub fn with_kid_samples(mut self, n: usize) -> Self {
        self.n_kid_samples = n;
        self
    }

    /// Enable/disable LPIPS
    pub fn with_lpips(mut self, enable: bool) -> Self {
        self.enable_lpips = enable;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute Inception Score (IS)
    pub fn inception_score(
        &self,
        features: &Array2<F>,
        splits: usize,
    ) -> Result<InceptionScoreResult<F>> {
        if features.is_empty() || splits == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty features or zero splits".to_string(),
            ));
        }

        let n_samples = features.nrows();
        let split_size = n_samples / splits;
        let mut scores = Vec::with_capacity(splits);

        for i in 0..splits {
            let start_idx = i * split_size;
            let end_idx = if i == splits - 1 {
                n_samples
            } else {
                (i + 1) * split_size
            };

            let split_features = features.slice(ndarray::s![start_idx..end_idx, ..]);

            // Convert features to probabilities (assuming they're logits)
            let probabilities = split_features.mapv(|x| F::one() / (F::one() + (-x).exp()));

            // Compute marginal probability
            let marginal = probabilities.mean_axis(Axis(0)).unwrap();

            // Compute KL divergence for each sample
            let mut kl_sum = F::zero();
            let mut valid_samples = 0;

            for sample_idx in 0..probabilities.nrows() {
                let sample_probs = probabilities.row(sample_idx);
                let mut sample_kl = F::zero();
                let mut valid_probs = 0;

                for (&p_sample, &p_marginal) in sample_probs.iter().zip(marginal.iter()) {
                    if p_sample > F::zero() && p_marginal > F::zero() {
                        sample_kl = sample_kl + p_sample * (p_sample / p_marginal).ln();
                        valid_probs += 1;
                    }
                }

                if valid_probs > 0 {
                    kl_sum = kl_sum + sample_kl;
                    valid_samples += 1;
                }
            }

            if valid_samples > 0 {
                let mean_kl = kl_sum / F::from(valid_samples).unwrap();
                scores.push(mean_kl.exp());
            } else {
                scores.push(F::one());
            }
        }

        let mean_score = scores.iter().copied().sum::<F>() / F::from(scores.len()).unwrap();
        let std_score = {
            let variance = scores
                .iter()
                .map(|&x| {
                    let diff = x - mean_score;
                    diff * diff
                })
                .sum::<F>()
                / F::from(scores.len()).unwrap();
            variance.sqrt()
        };

        Ok(InceptionScoreResult {
            mean_score,
            std_score,
            split_scores: scores,
        })
    }

    /// Compute Fréchet Inception Distance (FID)
    pub fn frechet_inception_distance(
        &self,
        real_features: &Array2<F>,
        fake_features: &Array2<F>,
    ) -> Result<F> {
        if real_features.is_empty() || fake_features.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty feature arrays".to_string(),
            ));
        }

        if real_features.ncols() != fake_features.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        // Compute means
        let mu_real = real_features.mean_axis(Axis(0)).unwrap();
        let mu_fake = fake_features.mean_axis(Axis(0)).unwrap();

        // Compute covariances
        let cov_real = self.compute_covariance_matrix(real_features)?;
        let cov_fake = self.compute_covariance_matrix(fake_features)?;

        // Compute squared L2 distance between means
        let mean_diff = &mu_real - &mu_fake;
        let mean_dist_sq = mean_diff.mapv(|x| x * x).sum();

        // Compute trace of covariances
        let trace_cov_real: F = (0..cov_real.nrows()).map(|i| cov_real[[i, i]]).sum();
        let trace_cov_fake: F = (0..cov_fake.nrows()).map(|i| cov_fake[[i, i]]).sum();

        // Compute product of covariances (simplified to diagonal approximation for efficiency)
        let mut trace_product = F::zero();
        for i in 0..cov_real.nrows() {
            for j in 0..cov_fake.ncols() {
                if i == j {
                    trace_product = trace_product + (cov_real[[i, j]] * cov_fake[[i, j]]).sqrt();
                }
            }
        }
        let trace_product_sqrt = trace_product;

        // FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
        let fid = mean_dist_sq + trace_cov_real + trace_cov_fake
            - F::from(2.0).unwrap() * trace_product_sqrt;

        Ok(fid)
    }

    /// Compute Kernel Inception Distance (KID)
    pub fn kernel_inception_distance(
        &self,
        real_features: &Array2<F>,
        fake_features: &Array2<F>,
        degree: usize,
        gamma: Option<F>,
    ) -> Result<KIDResult<F>> {
        if real_features.is_empty() || fake_features.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty feature arrays".to_string(),
            ));
        }

        let n_real = real_features.nrows().min(self.n_kid_samples);
        let n_fake = fake_features.nrows().min(self.n_kid_samples);

        // Subsample for efficiency
        let real_sub = real_features.slice(ndarray::s![0..n_real, ..]);
        let fake_sub = fake_features.slice(ndarray::s![0..n_fake, ..]);

        // Compute polynomial kernel matrices
        let gamma_val = gamma.unwrap_or_else(|| F::one() / F::from(real_features.ncols()).unwrap());

        let k_rr = self.compute_polynomial_kernel(&real_sub, &real_sub, degree, gamma_val)?;
        let k_ff = self.compute_polynomial_kernel(&fake_sub, &fake_sub, degree, gamma_val)?;
        let k_rf = self.compute_polynomial_kernel(&real_sub, &fake_sub, degree, gamma_val)?;

        // Compute KID estimate
        let term1 = k_rr.sum() / F::from(n_real * n_real).unwrap();
        let term2 = k_ff.sum() / F::from(n_fake * n_fake).unwrap();
        let term3 = k_rf.sum() / F::from(n_real * n_fake).unwrap();

        let kid_estimate = term1 + term2 - F::from(2.0).unwrap() * term3;

        // Compute bias correction
        let bias_correction = self.compute_kid_bias_correction(n_real, n_fake, &k_rr, &k_ff)?;
        let kid_corrected = kid_estimate - bias_correction;

        Ok(KIDResult {
            kid_estimate,
            kid_corrected,
            bias_correction,
            n_samples_real: n_real,
            n_samples_fake: n_fake,
        })
    }

    /// Compute covariance matrix
    fn compute_covariance_matrix(&self, features: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples for covariance".to_string(),
            ));
        }

        // Center the data
        let mean = features.mean_axis(Axis(0)).unwrap();
        let centered = features - &mean.insert_axis(Axis(0));

        // Compute covariance matrix: (1/(n-1)) * X^T * X
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = F::zero();
                for k in 0..n_samples {
                    sum = sum + centered[[k, i]] * centered[[k, j]];
                }
                let cov_val = sum / F::from(n_samples - 1).unwrap();
                cov[[i, j]] = cov_val;
                if i != j {
                    cov[[j, i]] = cov_val; // Symmetric
                }
            }
        }

        Ok(cov)
    }

    /// Compute polynomial kernel matrix
    fn compute_polynomial_kernel(
        &self,
        x1: &ArrayView2<F>,
        x2: &ArrayView2<F>,
        degree: usize,
        gamma: F,
    ) -> Result<Array2<F>> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                // Compute dot product
                let mut dot_product = F::zero();
                for k in 0..x1.ncols() {
                    dot_product = dot_product + x1[[i, k]] * x2[[j, k]];
                }

                // Polynomial kernel: (gamma * <x1, x2> + 1)^degree
                let kernel_val = (gamma * dot_product + F::one()).powf(F::from(degree).unwrap());
                kernel[[i, j]] = kernel_val;
            }
        }

        Ok(kernel)
    }

    /// Compute KID bias correction
    fn compute_kid_bias_correction(
        &self,
        n_real: usize,
        n_fake: usize,
        k_rr: &Array2<F>,
        k_ff: &Array2<F>,
    ) -> Result<F> {
        // Simplified bias correction (diagonal terms)
        let diag_rr = (0..n_real).map(|i| k_rr[[i, i]]).sum::<F>() / F::from(n_real).unwrap();
        let diag_ff = (0..n_fake).map(|i| k_ff[[i, i]]).sum::<F>() / F::from(n_fake).unwrap();

        let bias = (diag_rr / F::from(n_real).unwrap()) + (diag_ff / F::from(n_fake).unwrap());
        Ok(bias)
    }
}

/// Contrastive learning metrics
pub struct ContrastiveLearningMetrics<F: Float> {
    /// Temperature parameter for contrastive loss
    pub temperature: F,
    /// Number of negative samples
    pub n_negatives: usize,
    /// Enable hard negative mining
    pub enable_hard_negatives: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for ContrastiveLearningMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand>
    ContrastiveLearningMetrics<F>
{
    /// Create new contrastive learning metrics
    pub fn new() -> Self {
        Self {
            temperature: F::from(0.1).unwrap(),
            n_negatives: 1024,
            enable_hard_negatives: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set temperature parameter
    pub fn with_temperature(mut self, temp: F) -> Self {
        self.temperature = temp;
        self
    }

    /// Set number of negatives
    pub fn with_negatives(mut self, n: usize) -> Self {
        self.n_negatives = n;
        self
    }

    /// Enable hard negative mining
    pub fn with_hard_negatives(mut self, enable: bool) -> Self {
        self.enable_hard_negatives = enable;
        self
    }

    /// Compute uniformity of representations
    pub fn uniformity(&self, representations: &Array2<F>, t: F) -> Result<F> {
        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        let n_samples = representations.nrows();
        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Normalize representations to unit sphere
        let normalized = self.l2_normalize(representations)?;

        let mut sum_exp = F::zero();
        let mut count = 0;

        // Compute pairwise similarities and uniformity
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let similarity = self.cosine_similarity(
                    &normalized.row(i).to_owned(),
                    &normalized.row(j).to_owned(),
                )?;

                sum_exp = sum_exp + (t * similarity).exp();
                count += 1;
            }
        }

        if count == 0 {
            return Ok(F::zero());
        }

        let uniformity = (sum_exp / F::from(count).unwrap()).ln() / t;
        Ok(uniformity)
    }

    /// Compute alignment between positive pairs
    pub fn alignment(
        &self,
        anchor_representations: &Array2<F>,
        positive_representations: &Array2<F>,
        alpha: F,
    ) -> Result<F> {
        if anchor_representations.nrows() != positive_representations.nrows() {
            return Err(MetricsError::InvalidInput(
                "Mismatched number of pairs".to_string(),
            ));
        }

        if anchor_representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty _representations".to_string(),
            ));
        }

        // Normalize _representations
        let anchor_norm = self.l2_normalize(anchor_representations)?;
        let positive_norm = self.l2_normalize(positive_representations)?;

        let mut sum_distance = F::zero();
        let n_pairs = anchor_norm.nrows();

        for i in 0..n_pairs {
            let distance_sq = self.squared_euclidean_distance(
                &anchor_norm.row(i).to_owned(),
                &positive_norm.row(i).to_owned(),
            )?;

            sum_distance = sum_distance + distance_sq.powf(alpha);
        }

        let alignment = sum_distance / F::from(n_pairs).unwrap();
        Ok(alignment)
    }

    /// Compute InfoNCE loss
    pub fn infonce_loss(
        &self,
        anchor_representations: &Array2<F>,
        positive_representations: &Array2<F>,
        negative_representations: &Array2<F>,
    ) -> Result<InfoNCEResult<F>> {
        if anchor_representations.nrows() != positive_representations.nrows() {
            return Err(MetricsError::InvalidInput(
                "Mismatched anchor-positive pairs".to_string(),
            ));
        }

        let n_pairs = anchor_representations.nrows();
        let n_negatives = negative_representations.nrows();

        if n_pairs == 0 || n_negatives == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty _representations".to_string(),
            ));
        }

        // Normalize all _representations
        let anchor_norm = self.l2_normalize(anchor_representations)?;
        let positive_norm = self.l2_normalize(positive_representations)?;
        let negative_norm = self.l2_normalize(negative_representations)?;

        let mut total_loss = F::zero();
        let mut correct_predictions = 0;

        for i in 0..n_pairs {
            // Compute positive similarity
            let pos_sim = self.cosine_similarity(
                &anchor_norm.row(i).to_owned(),
                &positive_norm.row(i).to_owned(),
            )?;
            let pos_logit = pos_sim / self.temperature;

            // Compute negative similarities
            let mut neg_logits = Vec::with_capacity(n_negatives);
            for j in 0..n_negatives {
                let neg_sim = self.cosine_similarity(
                    &anchor_norm.row(i).to_owned(),
                    &negative_norm.row(j).to_owned(),
                )?;
                neg_logits.push(neg_sim / self.temperature);
            }

            // Compute softmax denominator
            let mut exp_sum = pos_logit.exp();
            for &neg_logit in &neg_logits {
                exp_sum = exp_sum + neg_logit.exp();
            }

            // InfoNCE loss for this sample
            let sample_loss = -pos_logit + exp_sum.ln();
            total_loss = total_loss + sample_loss;

            // Check if positive is the highest scoring
            let max_neg_logit = neg_logits
                .iter()
                .copied()
                .fold(neg_logits[0], |a, b| a.max(b));
            if pos_logit > max_neg_logit {
                correct_predictions += 1;
            }
        }

        let mean_loss = total_loss / F::from(n_pairs).unwrap();
        let accuracy = F::from(correct_predictions).unwrap() / F::from(n_pairs).unwrap();

        Ok(InfoNCEResult {
            loss: mean_loss,
            accuracy,
            n_pairs,
            temperature: self.temperature,
        })
    }

    /// L2 normalize representations
    fn l2_normalize(&self, representations: &Array2<F>) -> Result<Array2<F>> {
        let mut normalized = representations.clone();

        for mut row in normalized.rows_mut() {
            let norm = (row.mapv(|x| x * x).sum()).sqrt();
            if norm > F::zero() {
                for val in row.iter_mut() {
                    *val = *val / norm;
                }
            }
        }

        Ok(normalized)
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let dot_product: F = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a = (a.mapv(|x| x * x).sum()).sqrt();
        let norm_b = (b.mapv(|x| x * x).sum()).sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::zero());
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute squared Euclidean distance
    fn squared_euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq)
    }
}

/// Self-supervised learning metrics
pub struct SelfSupervisedMetrics<F: Float> {
    /// Number of linear probing epochs
    pub n_probe_epochs: usize,
    /// Learning rate for linear probing
    pub probe_learning_rate: F,
    /// Number of clustering attempts
    pub n_clustering_runs: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for SelfSupervisedMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> SelfSupervisedMetrics<F> {
    /// Create new self-supervised learning metrics
    pub fn new() -> Self {
        Self {
            n_probe_epochs: 100,
            probe_learning_rate: F::from(0.001).unwrap(),
            n_clustering_runs: 5,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set linear probing parameters
    pub fn with_linear_probe(mut self, epochs: usize, lr: F) -> Self {
        self.n_probe_epochs = epochs;
        self.probe_learning_rate = lr;
        self
    }

    /// Set clustering parameters
    pub fn with_clustering(mut self, runs: usize) -> Self {
        self.n_clustering_runs = runs;
        self
    }

    /// Compute linear probing accuracy
    pub fn linear_probing_accuracy(
        &self,
        representations: &Array2<F>,
        labels: &Array1<usize>,
        test_representations: &Array2<F>,
        test_labels: &Array1<usize>,
    ) -> Result<LinearProbingResult<F>> {
        if representations.nrows() != labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched representations and labels".to_string(),
            ));
        }

        if test_representations.nrows() != test_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched test representations and labels".to_string(),
            ));
        }

        // Get number of classes
        let n_classes = labels.iter().max().unwrap_or(&0) + 1;
        let n_features = representations.ncols();

        // Initialize linear classifier weights (simplified to centroid-based)
        let mut class_centroids = Array2::zeros((n_classes, n_features));
        let mut class_counts = vec![0; n_classes];

        // Compute class centroids
        for (i, &label) in labels.iter().enumerate() {
            for j in 0..n_features {
                class_centroids[[label, j]] = class_centroids[[label, j]] + representations[[i, j]];
            }
            class_counts[label] += 1;
        }

        // Normalize centroids
        for class in 0..n_classes {
            if class_counts[class] > 0 {
                let count = F::from(class_counts[class]).unwrap();
                for j in 0..n_features {
                    class_centroids[[class, j]] = class_centroids[[class, j]] / count;
                }
            }
        }

        // Evaluate on test set
        let mut correct_predictions = 0;
        let mut per_class_correct = vec![0; n_classes];
        let mut per_class_total = vec![0; n_classes];

        for (i, &true_label) in test_labels.iter().enumerate() {
            let test_sample = test_representations.row(i);

            // Find closest centroid
            let mut best_distance = F::infinity();
            let mut predicted_class = 0;

            for class in 0..n_classes {
                if class_counts[class] > 0 {
                    let centroid = class_centroids.row(class);
                    let distance =
                        self.euclidean_distance(&test_sample.to_owned(), &centroid.to_owned())?;

                    if distance < best_distance {
                        best_distance = distance;
                        predicted_class = class;
                    }
                }
            }

            per_class_total[true_label] += 1;
            if predicted_class == true_label {
                correct_predictions += 1;
                per_class_correct[true_label] += 1;
            }
        }

        let overall_accuracy =
            F::from(correct_predictions).unwrap() / F::from(test_labels.len()).unwrap();

        // Compute per-class accuracies
        let mut per_class_accuracies = Vec::with_capacity(n_classes);
        for class in 0..n_classes {
            if per_class_total[class] > 0 {
                let acc = F::from(per_class_correct[class]).unwrap()
                    / F::from(per_class_total[class]).unwrap();
                per_class_accuracies.push(acc);
            } else {
                per_class_accuracies.push(F::zero());
            }
        }

        let balanced_accuracy =
            per_class_accuracies.iter().copied().sum::<F>() / F::from(n_classes).unwrap();

        Ok(LinearProbingResult {
            overall_accuracy,
            balanced_accuracy,
            per_class_accuracies,
            n_classes,
            n_test_samples: test_labels.len(),
        })
    }

    /// Compute representation rank (effective dimensionality)
    pub fn representation_rank(
        &self,
        representations: &Array2<F>,
        threshold: F,
    ) -> Result<RepresentationRankResult<F>> {
        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Compute covariance matrix
        let cov = self.compute_covariance_matrix(representations)?;

        // Compute eigenvalues (simplified approximation using diagonal)
        let mut eigenvalues: Vec<F> = (0..cov.nrows()).map(|i| cov[[i, i]]).collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Compute total variance
        let total_variance: F = eigenvalues.iter().copied().sum();

        // Find effective rank
        let mut cumulative_variance = F::zero();
        let mut effective_rank = 0;

        for &eigenval in &eigenvalues {
            cumulative_variance = cumulative_variance + eigenval;
            effective_rank += 1;

            if cumulative_variance / total_variance >= threshold {
                break;
            }
        }

        // Compute participation ratio
        let sum_eigenvals: F = eigenvalues.iter().copied().sum();
        let sum_eigenvals_sq: F = eigenvalues.iter().map(|&x| x * x).sum();
        let participation_ratio = if sum_eigenvals_sq > F::zero() {
            (sum_eigenvals * sum_eigenvals) / sum_eigenvals_sq
        } else {
            F::zero()
        };

        Ok(RepresentationRankResult {
            effective_rank,
            participation_ratio,
            eigenvalues,
            total_variance,
            explained_variance_ratio: cumulative_variance / total_variance,
        })
    }

    /// Compute clustering-based evaluation (simplified NMI)
    pub fn clustering_normalized_mutual_information(
        &self,
        representations: &Array2<F>,
        true_labels: &Array1<usize>,
        n_clusters: usize,
    ) -> Result<ClusteringResult<F>> {
        if representations.nrows() != true_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched representations and _labels".to_string(),
            ));
        }

        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Perform simple k-means clustering (simplified centroid-based)
        let cluster_assignments = self.simple_kmeans(representations, n_clusters)?;

        // Compute normalized mutual information
        let nmi = self.compute_normalized_mutual_information(true_labels, &cluster_assignments)?;

        // Compute adjusted rand index (simplified)
        let ari = self.compute_adjusted_rand_index(true_labels, &cluster_assignments)?;

        // Compute silhouette score
        let silhouette = self.compute_silhouette_score(representations, &cluster_assignments)?;

        Ok(ClusteringResult {
            normalized_mutual_information: nmi,
            adjusted_rand_index: ari,
            silhouette_score: silhouette,
            cluster_assignments,
            n_clusters,
        })
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq.sqrt())
    }

    /// Compute covariance matrix
    fn compute_covariance_matrix(&self, data: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean.insert_axis(Axis(0));

        // Compute covariance
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = F::zero();
                for k in 0..n_samples {
                    sum = sum + centered[[k, i]] * centered[[k, j]];
                }
                let cov_val = sum / F::from(n_samples - 1).unwrap();
                cov[[i, j]] = cov_val;
                if i != j {
                    cov[[j, i]] = cov_val;
                }
            }
        }

        Ok(cov)
    }

    /// Simple k-means clustering
    fn simple_kmeans(&self, data: &Array2<F>, k: usize) -> Result<Vec<usize>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if k == 0 || k > n_samples {
            return Err(MetricsError::InvalidInput(
                "Invalid number of clusters".to_string(),
            ));
        }

        // Initialize centroids (use first k samples)
        let mut centroids = Array2::zeros((k, n_features));
        for i in 0..k {
            for j in 0..n_features {
                centroids[[i, j]] = data[[i % n_samples, j]];
            }
        }

        let mut assignments = vec![0; n_samples];
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut best_distance = F::infinity();
                let mut best_cluster = 0;

                for j in 0..k {
                    let distance = self.euclidean_distance(
                        &data.row(i).to_owned(),
                        &centroids.row(j).to_owned(),
                    )?;

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut cluster_counts = vec![0; k];
            centroids.fill(F::zero());

            for i in 0..n_samples {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;

                for j in 0..n_features {
                    centroids[[cluster, j]] = centroids[[cluster, j]] + data[[i, j]];
                }
            }

            // Normalize centroids
            for i in 0..k {
                if cluster_counts[i] > 0 {
                    let count = F::from(cluster_counts[i]).unwrap();
                    for j in 0..n_features {
                        centroids[[i, j]] = centroids[[i, j]] / count;
                    }
                }
            }
        }

        Ok(assignments)
    }

    /// Compute normalized mutual information
    fn compute_normalized_mutual_information(
        &self,
        true_labels: &Array1<usize>,
        pred_labels: &[usize],
    ) -> Result<F> {
        if true_labels.len() != pred_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Label length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n == 0 {
            return Ok(F::zero());
        }

        // Build contingency table
        let max_true = *true_labels.iter().max().unwrap_or(&0) + 1;
        let max_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;

        let mut contingency = vec![vec![0; max_pred]; max_true];
        for i in 0..n {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }

        // Compute marginals
        let mut true_marginal = vec![0; max_true];
        let mut pred_marginal = vec![0; max_pred];

        for i in 0..max_true {
            for j in 0..max_pred {
                true_marginal[i] += contingency[i][j];
                pred_marginal[j] += contingency[i][j];
            }
        }

        // Compute mutual information
        let mut mi = F::zero();
        for i in 0..max_true {
            for j in 0..max_pred {
                if contingency[i][j] > 0 {
                    let p_ij = F::from(contingency[i][j]).unwrap() / F::from(n).unwrap();
                    let p_i = F::from(true_marginal[i]).unwrap() / F::from(n).unwrap();
                    let p_j = F::from(pred_marginal[j]).unwrap() / F::from(n).unwrap();

                    if p_i > F::zero() && p_j > F::zero() {
                        mi = mi + p_ij * (p_ij / (p_i * p_j)).ln();
                    }
                }
            }
        }

        // Compute entropies for normalization
        let mut h_true = F::zero();
        let mut h_pred = F::zero();

        for i in 0..max_true {
            if true_marginal[i] > 0 {
                let p_i = F::from(true_marginal[i]).unwrap() / F::from(n).unwrap();
                h_true = h_true - p_i * p_i.ln();
            }
        }

        for j in 0..max_pred {
            if pred_marginal[j] > 0 {
                let p_j = F::from(pred_marginal[j]).unwrap() / F::from(n).unwrap();
                h_pred = h_pred - p_j * p_j.ln();
            }
        }

        // Normalize
        let denominator = (h_true + h_pred) / F::from(2.0).unwrap();
        if denominator > F::zero() {
            Ok(mi / denominator)
        } else {
            Ok(F::zero())
        }
    }

    /// Compute adjusted rand index (simplified)
    fn compute_adjusted_rand_index(
        &self,
        true_labels: &Array1<usize>,
        pred_labels: &[usize],
    ) -> Result<F> {
        if true_labels.len() != pred_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Label length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n == 0 {
            return Ok(F::zero());
        }

        // Count agreements
        let mut agreements = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let same_true = true_labels[i] == true_labels[j];
                let same_pred = pred_labels[i] == pred_labels[j];

                if same_true == same_pred {
                    agreements += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            return Ok(F::zero());
        }

        // Simplified ARI (just agreement ratio)
        Ok(F::from(agreements).unwrap() / F::from(total_pairs).unwrap())
    }

    /// Compute silhouette score
    fn compute_silhouette_score(
        &self,
        data: &Array2<F>,
        cluster_assignments: &[usize],
    ) -> Result<F> {
        let n_samples = data.nrows();
        if n_samples != cluster_assignments.len() {
            return Err(MetricsError::InvalidInput(
                "Data and _assignments length mismatch".to_string(),
            ));
        }

        if n_samples < 2 {
            return Ok(F::zero());
        }

        let mut total_silhouette = F::zero();
        let mut valid_samples = 0;

        for i in 0..n_samples {
            let cluster_i = cluster_assignments[i];

            // Compute average intra-cluster distance
            let mut intra_distance = F::zero();
            let mut intra_count = 0;

            for j in 0..n_samples {
                if i != j && cluster_assignments[j] == cluster_i {
                    let distance =
                        self.euclidean_distance(&data.row(i).to_owned(), &data.row(j).to_owned())?;
                    intra_distance = intra_distance + distance;
                    intra_count += 1;
                }
            }

            if intra_count > 0 {
                intra_distance = intra_distance / F::from(intra_count).unwrap();
            }

            // Compute minimum average inter-cluster distance
            let mut min_inter_distance = F::infinity();
            let max_cluster = *cluster_assignments.iter().max().unwrap_or(&0);

            for other_cluster in 0..=max_cluster {
                if other_cluster != cluster_i {
                    let mut inter_distance = F::zero();
                    let mut inter_count = 0;

                    for j in 0..n_samples {
                        if cluster_assignments[j] == other_cluster {
                            let distance = self.euclidean_distance(
                                &data.row(i).to_owned(),
                                &data.row(j).to_owned(),
                            )?;
                            inter_distance = inter_distance + distance;
                            inter_count += 1;
                        }
                    }

                    if inter_count > 0 {
                        inter_distance = inter_distance / F::from(inter_count).unwrap();
                        min_inter_distance = min_inter_distance.min(inter_distance);
                    }
                }
            }

            // Compute silhouette for this sample
            if min_inter_distance != F::infinity() {
                let max_distance = intra_distance.max(min_inter_distance);
                if max_distance > F::zero() {
                    let silhouette = (min_inter_distance - intra_distance) / max_distance;
                    total_silhouette = total_silhouette + silhouette;
                    valid_samples += 1;
                }
            }
        }

        if valid_samples > 0 {
            Ok(total_silhouette / F::from(valid_samples).unwrap())
        } else {
            Ok(F::zero())
        }
    }
}

/// Foundation model metrics
pub struct FoundationModelMetrics<F: Float> {
    /// Number of shots for few-shot evaluation
    pub n_shots: Vec<usize>,
    /// Number of tasks for evaluation
    pub n_tasks: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for FoundationModelMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand>
    FoundationModelMetrics<F>
{
    /// Create new foundation model metrics
    pub fn new() -> Self {
        Self {
            n_shots: vec![1, 5, 10],
            n_tasks: 10,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set few-shot evaluation parameters
    pub fn with_few_shot(mut self, shots: Vec<usize>) -> Self {
        self.n_shots = shots;
        self
    }

    /// Set number of evaluation tasks
    pub fn with_tasks(mut self, n: usize) -> Self {
        self.n_tasks = n;
        self
    }

    /// Compute zero-shot performance
    pub fn zero_shot_accuracy(
        &self,
        predictions: &Array1<F>,
        targets: &Array1<usize>,
    ) -> Result<F> {
        if predictions.len() != targets.len() {
            return Err(MetricsError::InvalidInput(
                "Prediction and target length mismatch".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Ok(F::zero());
        }

        let mut correct = 0;
        for (i, &target) in targets.iter().enumerate() {
            // Convert prediction to class (assuming binary classification for simplicity)
            let predicted_class = if predictions[i] > F::from(0.5).unwrap() {
                1
            } else {
                0
            };
            if predicted_class == target {
                correct += 1;
            }
        }

        Ok(F::from(correct).unwrap() / F::from(predictions.len()).unwrap())
    }

    /// Compute few-shot learning performance
    pub fn few_shot_performance(
        &self,
        support_representations: &Array2<F>,
        support_labels: &Array1<usize>,
        query_representations: &Array2<F>,
        query_labels: &Array1<usize>,
        n_shot: usize,
    ) -> Result<FewShotResult<F>> {
        if support_representations.nrows() != support_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Support data length mismatch".to_string(),
            ));
        }

        if query_representations.nrows() != query_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Query data length mismatch".to_string(),
            ));
        }

        let n_classes = support_labels.iter().max().unwrap_or(&0) + 1;

        // Select n_shot examples per class
        let mut selected_support = Vec::new();
        let mut selected_labels = Vec::new();

        for class in 0..n_classes {
            let class_indices: Vec<usize> = support_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .take(n_shot)
                .collect();

            for &idx in &class_indices {
                selected_support.push(support_representations.row(idx).to_owned());
                selected_labels.push(class);
            }
        }

        if selected_support.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No support examples selected".to_string(),
            ));
        }

        // Perform nearest neighbor classification
        let mut correct = 0;
        let mut per_class_correct = vec![0; n_classes];
        let mut per_class_total = vec![0; n_classes];

        for (i, &true_label) in query_labels.iter().enumerate() {
            let query_sample = query_representations.row(i);

            let mut best_distance = F::infinity();
            let mut predicted_class = 0;

            for (j, support_sample) in selected_support.iter().enumerate() {
                let distance = self.euclidean_distance(&query_sample.to_owned(), support_sample)?;

                if distance < best_distance {
                    best_distance = distance;
                    predicted_class = selected_labels[j];
                }
            }

            per_class_total[true_label] += 1;
            if predicted_class == true_label {
                correct += 1;
                per_class_correct[true_label] += 1;
            }
        }

        let overall_accuracy = F::from(correct).unwrap() / F::from(query_labels.len()).unwrap();

        let mut per_class_accuracies = Vec::with_capacity(n_classes);
        for class in 0..n_classes {
            if per_class_total[class] > 0 {
                let acc = F::from(per_class_correct[class]).unwrap()
                    / F::from(per_class_total[class]).unwrap();
                per_class_accuracies.push(acc);
            } else {
                per_class_accuracies.push(F::zero());
            }
        }

        let balanced_accuracy =
            per_class_accuracies.iter().copied().sum::<F>() / F::from(n_classes).unwrap();

        Ok(FewShotResult {
            overall_accuracy,
            balanced_accuracy,
            per_class_accuracies,
            n_shot,
            n_classes,
            n_query_samples: query_labels.len(),
        })
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq.sqrt())
    }
}

/// Multimodal metrics
pub struct MultimodalMetrics<F: Float> {
    /// Number of retrieval candidates
    pub n_retrieval_candidates: usize,
    /// Top-k values for retrieval evaluation
    pub retrieval_k_values: Vec<usize>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for MultimodalMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> MultimodalMetrics<F> {
    /// Create new multimodal metrics
    pub fn new() -> Self {
        Self {
            n_retrieval_candidates: 1000,
            retrieval_k_values: vec![1, 5, 10],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set retrieval parameters
    pub fn with_retrieval(mut self, candidates: usize, kvalues: Vec<usize>) -> Self {
        self.n_retrieval_candidates = candidates;
        self.retrieval_k_values = kvalues;
        self
    }

    /// Compute cross-modal retrieval performance
    pub fn cross_modal_retrieval(
        &self,
        query_embeddings: &Array2<F>,
        candidate_embeddings: &Array2<F>,
        ground_truth_pairs: &[(usize, usize)], // (query_idx, candidate_idx) pairs
    ) -> Result<CrossModalRetrievalResult<F>> {
        if query_embeddings.is_empty() || candidate_embeddings.is_empty() {
            return Err(MetricsError::InvalidInput("Empty _embeddings".to_string()));
        }

        if query_embeddings.ncols() != candidate_embeddings.ncols() {
            return Err(MetricsError::InvalidInput(
                "Embedding dimension mismatch".to_string(),
            ));
        }

        let n_queries = query_embeddings.nrows();
        let n_candidates = candidate_embeddings.nrows();

        // Compute similarity matrix
        let mut similarities = Array2::zeros((n_queries, n_candidates));

        for i in 0..n_queries {
            for j in 0..n_candidates {
                let sim = self.cosine_similarity(
                    &query_embeddings.row(i).to_owned(),
                    &candidate_embeddings.row(j).to_owned(),
                )?;
                similarities[[i, j]] = sim;
            }
        }

        // Create ground truth lookup
        let mut gt_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(query_idx, candidate_idx) in ground_truth_pairs {
            gt_map.entry(query_idx).or_default().push(candidate_idx);
        }

        // Compute recall at k for each k value
        let mut recall_at_k = HashMap::new();
        let mut precision_at_k = HashMap::new();

        for &k in &self.retrieval_k_values {
            let mut total_recall = F::zero();
            let mut total_precision = F::zero();
            let mut valid_queries = 0;

            for query_idx in 0..n_queries {
                if let Some(gt_candidates) = gt_map.get(&query_idx) {
                    // Get top-k candidates for this query
                    let mut query_similarities: Vec<(F, usize)> = (0..n_candidates)
                        .map(|j| (similarities[[query_idx, j]], j))
                        .collect();

                    query_similarities
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                    let top_k_candidates: Vec<usize> = query_similarities
                        .iter()
                        .take(k)
                        .map(|(_, idx)| *idx)
                        .collect();

                    // Count hits
                    let hits = top_k_candidates
                        .iter()
                        .filter(|&&candidate| gt_candidates.contains(&candidate))
                        .count();

                    // Compute recall and precision for this query
                    let query_recall =
                        F::from(hits).unwrap() / F::from(gt_candidates.len()).unwrap();
                    let query_precision = F::from(hits).unwrap() / F::from(k).unwrap();

                    total_recall = total_recall + query_recall;
                    total_precision = total_precision + query_precision;
                    valid_queries += 1;
                }
            }

            if valid_queries > 0 {
                recall_at_k.insert(k, total_recall / F::from(valid_queries).unwrap());
                precision_at_k.insert(k, total_precision / F::from(valid_queries).unwrap());
            } else {
                recall_at_k.insert(k, F::zero());
                precision_at_k.insert(k, F::zero());
            }
        }

        // Compute mean reciprocal rank
        let mut mrr = F::zero();
        let mut valid_queries = 0;

        for query_idx in 0..n_queries {
            if let Some(gt_candidates) = gt_map.get(&query_idx) {
                let mut query_similarities: Vec<(F, usize)> = (0..n_candidates)
                    .map(|j| (similarities[[query_idx, j]], j))
                    .collect();

                query_similarities
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Find rank of first relevant item
                for (rank, (_, candidate_idx)) in query_similarities.iter().enumerate() {
                    if gt_candidates.contains(candidate_idx) {
                        mrr = mrr + F::one() / F::from(rank + 1).unwrap();
                        break;
                    }
                }
                valid_queries += 1;
            }
        }

        if valid_queries > 0 {
            mrr = mrr / F::from(valid_queries).unwrap();
        }

        Ok(CrossModalRetrievalResult {
            recall_at_k,
            precision_at_k,
            mean_reciprocal_rank: mrr,
            n_queries,
            n_candidates,
        })
    }

    /// Compute multimodal alignment score
    pub fn multimodal_alignment(
        &self,
        modality1_embeddings: &Array2<F>,
        modality2_embeddings: &Array2<F>,
        paired_indices: &[(usize, usize)],
    ) -> Result<MultimodalAlignmentResult<F>> {
        if modality1_embeddings.ncols() != modality2_embeddings.ncols() {
            return Err(MetricsError::InvalidInput(
                "Embedding dimension mismatch".to_string(),
            ));
        }

        if paired_indices.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No paired _indices provided".to_string(),
            ));
        }

        let mut alignment_scores = Vec::with_capacity(paired_indices.len());
        let mut positive_similarities = Vec::with_capacity(paired_indices.len());

        for &(idx1, idx2) in paired_indices {
            if idx1 >= modality1_embeddings.nrows() || idx2 >= modality2_embeddings.nrows() {
                return Err(MetricsError::InvalidInput(
                    "Index out of bounds".to_string(),
                ));
            }

            let similarity = self.cosine_similarity(
                &modality1_embeddings.row(idx1).to_owned(),
                &modality2_embeddings.row(idx2).to_owned(),
            )?;

            positive_similarities.push(similarity);
            alignment_scores.push(similarity);
        }

        // Compute negative similarities (random pairs)
        let mut negative_similarities = Vec::new();
        let n_negatives = paired_indices.len() * 5; // 5x negative sampling

        for i in 0..n_negatives {
            let idx1 = i % modality1_embeddings.nrows();
            let idx2 = (i * 7) % modality2_embeddings.nrows(); // Use prime for better randomness

            // Skip if this is actually a positive pair
            if !paired_indices.contains(&(idx1, idx2)) {
                let similarity = self.cosine_similarity(
                    &modality1_embeddings.row(idx1).to_owned(),
                    &modality2_embeddings.row(idx2).to_owned(),
                )?;
                negative_similarities.push(similarity);
            }
        }

        // Compute alignment metrics
        let mean_positive_similarity = positive_similarities.iter().copied().sum::<F>()
            / F::from(positive_similarities.len()).unwrap();

        let mean_negative_similarity = if !negative_similarities.is_empty() {
            negative_similarities.iter().copied().sum::<F>()
                / F::from(negative_similarities.len()).unwrap()
        } else {
            F::zero()
        };

        let alignment_gap = mean_positive_similarity - mean_negative_similarity;

        // Compute standard deviations
        let pos_variance = positive_similarities
            .iter()
            .map(|&x| {
                let diff = x - mean_positive_similarity;
                diff * diff
            })
            .sum::<F>()
            / F::from(positive_similarities.len()).unwrap();
        let pos_std = pos_variance.sqrt();

        let neg_variance = if !negative_similarities.is_empty() {
            negative_similarities
                .iter()
                .map(|&x| {
                    let diff = x - mean_negative_similarity;
                    diff * diff
                })
                .sum::<F>()
                / F::from(negative_similarities.len()).unwrap()
        } else {
            F::zero()
        };
        let neg_std = neg_variance.sqrt();

        Ok(MultimodalAlignmentResult {
            mean_positive_similarity,
            mean_negative_similarity,
            alignment_gap,
            positive_std: pos_std,
            negative_std: neg_std,
            n_positive_pairs: positive_similarities.len(),
            n_negative_pairs: negative_similarities.len(),
        })
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let dot_product: F = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a = (a.mapv(|x| x * x).sum()).sqrt();
        let norm_b = (b.mapv(|x| x * x).sum()).sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::zero());
        }

        Ok(dot_product / (norm_a * norm_b))
    }
}

// Result structures

/// Result of Inception Score computation
#[derive(Debug, Clone)]
pub struct InceptionScoreResult<F: Float> {
    pub mean_score: F,
    pub std_score: F,
    pub split_scores: Vec<F>,
}

/// Result of KID computation
#[derive(Debug, Clone)]
pub struct KIDResult<F: Float> {
    pub kid_estimate: F,
    pub kid_corrected: F,
    pub bias_correction: F,
    pub n_samples_real: usize,
    pub n_samples_fake: usize,
}

/// Result of InfoNCE computation
#[derive(Debug, Clone)]
pub struct InfoNCEResult<F: Float> {
    pub loss: F,
    pub accuracy: F,
    pub n_pairs: usize,
    pub temperature: F,
}

/// Result of linear probing evaluation
#[derive(Debug, Clone)]
pub struct LinearProbingResult<F: Float> {
    pub overall_accuracy: F,
    pub balanced_accuracy: F,
    pub per_class_accuracies: Vec<F>,
    pub n_classes: usize,
    pub n_test_samples: usize,
}

/// Result of representation rank analysis
#[derive(Debug, Clone)]
pub struct RepresentationRankResult<F: Float> {
    pub effective_rank: usize,
    pub participation_ratio: F,
    pub eigenvalues: Vec<F>,
    pub total_variance: F,
    pub explained_variance_ratio: F,
}

/// Result of clustering evaluation
#[derive(Debug, Clone)]
pub struct ClusteringResult<F: Float> {
    pub normalized_mutual_information: F,
    pub adjusted_rand_index: F,
    pub silhouette_score: F,
    pub cluster_assignments: Vec<usize>,
    pub n_clusters: usize,
}

/// Result of few-shot learning evaluation
#[derive(Debug, Clone)]
pub struct FewShotResult<F: Float> {
    pub overall_accuracy: F,
    pub balanced_accuracy: F,
    pub per_class_accuracies: Vec<F>,
    pub n_shot: usize,
    pub n_classes: usize,
    pub n_query_samples: usize,
}

/// Result of cross-modal retrieval evaluation
#[derive(Debug, Clone)]
pub struct CrossModalRetrievalResult<F: Float> {
    pub recall_at_k: HashMap<usize, F>,
    pub precision_at_k: HashMap<usize, F>,
    pub mean_reciprocal_rank: F,
    pub n_queries: usize,
    pub n_candidates: usize,
}

/// Result of multimodal alignment evaluation
#[derive(Debug, Clone)]
pub struct MultimodalAlignmentResult<F: Float> {
    pub mean_positive_similarity: F,
    pub mean_negative_similarity: F,
    pub alignment_gap: F,
    pub positive_std: F,
    pub negative_std: F,
    pub n_positive_pairs: usize,
    pub n_negative_pairs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn mock_inception_features() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    }

    #[test]
    fn test_inception_score() {
        let gan_metrics = GANEvaluationMetrics::<f64>::new();
        let features = mock_inception_features();

        let result = gan_metrics.inception_score(&features, 2).unwrap();

        assert!(result.mean_score > 0.0);
        assert!(result.std_score >= 0.0);
        assert_eq!(result.split_scores.len(), 2);
    }

    #[test]
    fn test_fid_score() {
        let gan_metrics = GANEvaluationMetrics::<f64>::new();
        let real_features = mock_inception_features();
        let fake_features = array![
            [1.1, 2.1, 3.1, 4.1],
            [2.1, 3.1, 4.1, 5.1],
            [3.1, 4.1, 5.1, 6.1],
            [4.1, 5.1, 6.1, 7.1],
        ];

        let fid = gan_metrics
            .frechet_inception_distance(&real_features, &fake_features)
            .unwrap();

        assert!(fid >= 0.0);
    }

    #[test]
    fn test_uniformity() {
        let contrastive_metrics = ContrastiveLearningMetrics::<f64>::new();
        let representations = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        let uniformity = contrastive_metrics
            .uniformity(&representations, 2.0)
            .unwrap();

        assert!(uniformity.is_finite());
    }

    #[test]
    fn test_alignment() {
        let contrastive_metrics = ContrastiveLearningMetrics::<f64>::new();
        let anchors = array![[1.0, 0.0], [0.0, 1.0],];
        let positives = array![[0.9, 0.1], [0.1, 0.9],];

        let alignment = contrastive_metrics
            .alignment(&anchors, &positives, 2.0)
            .unwrap();

        assert!(alignment >= 0.0);
    }

    #[test]
    fn test_linear_probing() {
        let ssl_metrics = SelfSupervisedMetrics::<f64>::new();

        let train_repr = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0],];
        let train_labels = array![0, 1, 0, 1];

        let test_repr = array![[1.1, 0.1], [0.1, 1.1],];
        let test_labels = array![0, 1];

        let result = ssl_metrics
            .linear_probing_accuracy(&train_repr, &train_labels, &test_repr, &test_labels)
            .unwrap();

        assert!(result.overall_accuracy >= 0.0);
        assert!(result.overall_accuracy <= 1.0);
        assert_eq!(result.n_classes, 2);
    }

    #[test]
    fn test_zero_shot_accuracy() {
        let foundation_metrics = FoundationModelMetrics::<f64>::new();

        let predictions = array![0.8, 0.3, 0.9, 0.1];
        let targets = array![1, 0, 1, 0];

        let accuracy = foundation_metrics
            .zero_shot_accuracy(&predictions, &targets)
            .unwrap();

        assert!(accuracy >= 0.0);
        assert!(accuracy <= 1.0);
    }

    #[test]
    fn test_cross_modal_retrieval() {
        let multimodal_metrics = MultimodalMetrics::<f64>::new();

        let query_emb = array![[1.0, 0.0], [0.0, 1.0],];
        let candidate_emb = array![
            [0.9, 0.1], // Similar to query 0
            [0.1, 0.9], // Similar to query 1
            [0.5, 0.5], // Neutral
        ];
        let gt_pairs = vec![(0, 0), (1, 1)];

        let result = multimodal_metrics
            .cross_modal_retrieval(&query_emb, &candidate_emb, &gt_pairs)
            .unwrap();

        assert!(result.mean_reciprocal_rank >= 0.0);
        assert!(result.mean_reciprocal_rank <= 1.0);
        assert_eq!(result.n_queries, 2);
        assert_eq!(result.n_candidates, 3);
    }

    #[test]
    fn test_generative_ai_suite() {
        let suite = GenerativeAISuite::<f64>::new();

        assert_eq!(suite.domain_name(), "Generative AI & Deep Learning");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
