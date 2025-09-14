//! Real-time streaming time series analysis
//!
//! This module provides capabilities for analyzing time series data in real-time,
//! including online learning algorithms, streaming forecasting, and incremental statistics.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// Configuration for streaming analysis
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum window size for online calculations
    pub window_size: usize,
    /// Minimum number of observations before starting analysis
    pub min_observations: usize,
    /// Update frequency for model parameters
    pub update_frequency: usize,
    /// Memory threshold for automatic cleanup
    pub memory_threshold: usize,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Detection threshold for change points
    pub change_detection_threshold: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            min_observations: 10,
            update_frequency: 10,
            memory_threshold: 10000,
            adaptive_windowing: false,
            change_detection_threshold: 3.0,
        }
    }
}

/// Real-time change point detection result
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index of the change point
    pub index: usize,
    /// Timestamp of the change point
    pub timestamp: Option<Instant>,
    /// Confidence score (higher = more confident)
    pub confidence: f64,
    /// Type of change detected
    pub change_type: ChangeType,
}

/// Types of changes that can be detected
#[derive(Debug, Clone)]
pub enum ChangeType {
    /// Change in mean
    MeanShift,
    /// Change in variance
    VarianceShift,
    /// Change in trend
    TrendChange,
    /// Change in seasonality
    SeasonalityChange,
    /// General structural break
    StructuralBreak,
}

/// Online statistics tracker
#[derive(Debug, Clone)]
pub struct OnlineStats<F: Float> {
    count: usize,
    mean: F,
    m2: F, // For variance calculation
    min_val: F,
    max_val: F,
    sum: F,
    sum_squares: F,
}

impl<F: Float + Debug> OnlineStats<F> {
    /// Create new online statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: F::zero(),
            m2: F::zero(),
            min_val: F::infinity(),
            max_val: F::neg_infinity(),
            sum: F::zero(),
            sum_squares: F::zero(),
        }
    }

    /// Update statistics with new observation
    pub fn update(&mut self, value: F) {
        self.count += 1;
        self.sum = self.sum + value;
        self.sum_squares = self.sum_squares + value * value;

        if value < self.min_val {
            self.min_val = value;
        }
        if value > self.max_val {
            self.max_val = value;
        }

        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean = self.mean + delta / F::from(self.count).unwrap();
        let delta2 = value - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    /// Get current mean
    pub fn mean(&self) -> F {
        self.mean
    }

    /// Get current variance
    pub fn variance(&self) -> F {
        if self.count < 2 {
            F::zero()
        } else {
            self.m2 / F::from(self.count - 1).unwrap()
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> F {
        self.variance().sqrt()
    }

    /// Get current minimum
    pub fn min(&self) -> F {
        self.min_val
    }

    /// Get current maximum
    pub fn max(&self) -> F {
        self.max_val
    }

    /// Get current count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get current sum
    pub fn sum(&self) -> F {
        self.sum
    }
}

impl<F: Float + Debug> Default for OnlineStats<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponentially Weighted Moving Average (EWMA) tracker
#[derive(Debug, Clone)]
pub struct EWMA<F: Float> {
    alpha: F,
    current_value: Option<F>,
    variance: Option<F>,
}

impl<F: Float + Debug> EWMA<F> {
    /// Create new EWMA tracker
    pub fn new(alpha: F) -> Result<Self> {
        if alpha <= F::zero() || alpha > F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "_alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        Ok(Self {
            alpha,
            current_value: None,
            variance: None,
        })
    }

    /// Update EWMA with new observation
    pub fn update(&mut self, value: F) {
        match self.current_value {
            None => {
                self.current_value = Some(value);
                self.variance = Some(F::zero());
            }
            Some(prev) => {
                let new_value = self.alpha * value + (F::one() - self.alpha) * prev;
                self.current_value = Some(new_value);

                // Update variance estimate
                let error = value - new_value;
                let new_variance = self.alpha * error * error
                    + (F::one() - self.alpha) * self.variance.unwrap_or(F::zero());
                self.variance = Some(new_variance);
            }
        }
    }

    /// Get current EWMA value
    pub fn value(&self) -> Option<F> {
        self.current_value
    }

    /// Get current variance estimate
    pub fn variance(&self) -> Option<F> {
        self.variance
    }

    /// Check if value is an outlier based on EWMA
    pub fn is_outlier(&self, value: F, threshold: F) -> bool {
        if let (Some(ewma), Some(var)) = (self.current_value, self.variance) {
            let std_dev = var.sqrt();
            let z_score = (value - ewma).abs() / std_dev;
            z_score > threshold
        } else {
            false
        }
    }
}

/// Cumulative Sum (CUSUM) change point detector
#[derive(Debug, Clone)]
pub struct CusumDetector<F: Float> {
    mean_estimate: F,
    threshold: F,
    cusum_pos: F,
    cusum_neg: F,
    count: usize,
    drift: F,
}

impl<F: Float + Debug> CusumDetector<F> {
    /// Create new CUSUM detector
    pub fn new(threshold: F, drift: F) -> Self {
        Self {
            mean_estimate: F::zero(),
            threshold,
            cusum_pos: F::zero(),
            cusum_neg: F::zero(),
            count: 0,
            drift,
        }
    }

    /// Update CUSUM with new observation
    pub fn update(&mut self, value: F) -> Option<ChangePoint> {
        self.count += 1;

        // Update mean estimate
        let delta = value - self.mean_estimate;
        self.mean_estimate = self.mean_estimate + delta / F::from(self.count).unwrap();

        // Update CUSUM statistics
        let diff = value - self.mean_estimate;
        self.cusum_pos = F::max(F::zero(), self.cusum_pos + diff - self.drift);
        self.cusum_neg = F::max(F::zero(), self.cusum_neg - diff - self.drift);

        // Check for change point
        if self.cusum_pos > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_pos.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else if self.cusum_neg > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_neg.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.cusum_pos = F::zero();
        self.cusum_neg = F::zero();
    }
}

/// Streaming time series analyzer
#[derive(Debug)]
pub struct StreamingAnalyzer<F: Float + Debug> {
    config: StreamConfig,
    buffer: VecDeque<F>,
    timestamps: VecDeque<Instant>,
    stats: OnlineStats<F>,
    ewma: EWMA<F>,
    cusum: CusumDetector<F>,
    change_points: Vec<ChangePoint>,
    forecasts: VecDeque<F>,
    last_update: Instant,
    observation_count: usize,
}

impl<F: Float + Debug> StreamingAnalyzer<F> {
    /// Create new streaming analyzer
    pub fn new(config: StreamConfig) -> Result<Self> {
        let ewma = EWMA::new(F::from(0.1).unwrap())?;
        let cusum = CusumDetector::new(
            F::from(config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );

        Ok(Self {
            config,
            buffer: VecDeque::new(),
            timestamps: VecDeque::new(),
            stats: OnlineStats::new(),
            ewma,
            cusum,
            change_points: Vec::new(),
            forecasts: VecDeque::new(),
            last_update: Instant::now(),
            observation_count: 0,
        })
    }

    /// Add new observation to the stream
    pub fn add_observation(&mut self, value: F) -> Result<()> {
        let now = Instant::now();

        // Add to buffer with window management
        self.buffer.push_back(value);
        self.timestamps.push_back(now);
        self.observation_count += 1;

        // Maintain window size
        if self.buffer.len() > self.config.window_size {
            self.buffer.pop_front();
            self.timestamps.pop_front();
        }

        // Update statistics
        self.stats.update(value);
        self.ewma.update(value);

        // Check for change points
        if let Some(change_point) = self.cusum.update(value) {
            self.change_points.push(change_point);
        }

        // Periodic model updates
        if self.observation_count % self.config.update_frequency == 0 {
            self.update_models()?;
        }

        // Memory management
        if self.change_points.len() > self.config.memory_threshold {
            self.cleanup_old_data();
        }

        self.last_update = now;
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &OnlineStats<F> {
        &self.stats
    }

    /// Get detected change points
    pub fn get_change_points(&self) -> &[ChangePoint] {
        &self.change_points
    }

    /// Get current EWMA value
    pub fn get_ewma(&self) -> Option<F> {
        self.ewma.value()
    }

    /// Check if a value is an outlier
    pub fn is_outlier(&self, value: F) -> bool {
        self.ewma.is_outlier(
            value,
            F::from(self.config.change_detection_threshold).unwrap(),
        )
    }

    /// Get streaming forecast for next n steps
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if self.buffer.len() < self.config.min_observations {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough observations for forecasting".to_string(),
                required: self.config.min_observations,
                actual: self.buffer.len(),
            });
        }

        let mut forecasts = Array1::zeros(steps);

        // Simple forecasting using EWMA and linear trend
        let current_value = self.ewma.value().unwrap_or(F::zero());
        let trend = self.estimate_trend();

        for i in 0..steps {
            let step_f = F::from(i + 1).unwrap();
            forecasts[i] = current_value + trend * step_f;
        }

        Ok(forecasts)
    }

    /// Estimate current trend from recent observations
    fn estimate_trend(&self) -> F {
        if self.buffer.len() < 2 {
            return F::zero();
        }

        let n = std::cmp::min(20, self.buffer.len()); // Use last 20 observations
        let recent: Vec<F> = self.buffer.iter().rev().take(n).cloned().collect();

        if recent.len() < 2 {
            return F::zero();
        }

        // Simple linear regression for trend
        let n_f = F::from(recent.len()).unwrap();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();

        for (i, &y) in recent.iter().enumerate() {
            let x = F::from(i).unwrap();
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_x2 = sum_x2 + x * x;
        }

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < F::epsilon() {
            F::zero()
        } else {
            (n_f * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Update internal models
    fn update_models(&mut self) -> Result<()> {
        // Here we could update more sophisticated models
        // For now, just update forecasts buffer
        if self.buffer.len() >= self.config.min_observations {
            let next_forecast = self.forecast(1)?;
            self.forecasts.push_back(next_forecast[0]);

            // Keep forecasts buffer reasonable size
            if self.forecasts.len() > 100 {
                self.forecasts.pop_front();
            }
        }
        Ok(())
    }

    /// Clean up old data to manage memory
    fn cleanup_old_data(&mut self) {
        // Keep only recent change points
        let keep_count = self.config.memory_threshold / 2;
        if self.change_points.len() > keep_count {
            self.change_points
                .drain(0..self.change_points.len() - keep_count);
        }

        // Clean up forecasts buffer
        if self.forecasts.len() > 50 {
            self.forecasts.drain(0..self.forecasts.len() - 50);
        }
    }

    /// Get time since last update
    pub fn time_since_last_update(&self) -> Duration {
        Instant::now().duration_since(self.last_update)
    }

    /// Get total observation count
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get current buffer as array view
    pub fn get_buffer(&self) -> Vec<F> {
        self.buffer.iter().cloned().collect()
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.timestamps.clear();
        self.stats = OnlineStats::new();
        self.ewma = EWMA::new(F::from(0.1).unwrap()).unwrap();
        self.cusum = CusumDetector::new(
            F::from(self.config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );
        self.change_points.clear();
        self.forecasts.clear();
        self.observation_count = 0;
        self.last_update = Instant::now();
    }
}

/// Multi-series streaming analyzer for handling multiple time series simultaneously
#[derive(Debug)]
pub struct MultiSeriesAnalyzer<F: Float + Debug> {
    analyzers: HashMap<String, StreamingAnalyzer<F>>,
    config: StreamConfig,
}

impl<F: Float + Debug> MultiSeriesAnalyzer<F> {
    /// Create new multi-series analyzer
    pub fn new(config: StreamConfig) -> Self {
        Self {
            analyzers: HashMap::new(),
            config,
        }
    }

    /// Add new time series to track
    pub fn add_series(&mut self, seriesid: String) -> Result<()> {
        let analyzer = StreamingAnalyzer::new(self.config.clone())?;
        self.analyzers.insert(seriesid, analyzer);
        Ok(())
    }

    /// Add observation to specific series
    pub fn add_observation(&mut self, seriesid: &str, value: F) -> Result<()> {
        if let Some(analyzer) = self.analyzers.get_mut(seriesid) {
            analyzer.add_observation(value)
        } else {
            Err(TimeSeriesError::InvalidInput(format!(
                "Series '{seriesid}' not found"
            )))
        }
    }

    /// Get analyzer for specific series
    pub fn get_analyzer(&self, seriesid: &str) -> Option<&StreamingAnalyzer<F>> {
        self.analyzers.get(seriesid)
    }

    /// Get mutable analyzer for specific series
    pub fn get_analyzer_mut(&mut self, seriesid: &str) -> Option<&mut StreamingAnalyzer<F>> {
        self.analyzers.get_mut(seriesid)
    }

    /// Get all series IDs
    pub fn get_series_ids(&self) -> Vec<String> {
        self.analyzers.keys().cloned().collect()
    }

    /// Remove series
    pub fn remove_series(&mut self, seriesid: &str) -> bool {
        self.analyzers.remove(seriesid).is_some()
    }

    /// Get cross-series correlation (simplified)
    pub fn get_correlation(&self, series1: &str, series2: &str) -> Result<F> {
        let analyzer1 = self.analyzers.get(series1).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{series1}' not found"))
        })?;

        let analyzer2 = self.analyzers.get(series2).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{series2}' not found"))
        })?;

        let buffer1 = analyzer1.get_buffer();
        let buffer2 = analyzer2.get_buffer();

        let min_len = std::cmp::min(buffer1.len(), buffer2.len());
        if min_len < 2 {
            return Ok(F::zero());
        }

        // Calculate Pearson correlation
        let mean1 = buffer1
            .iter()
            .take(min_len)
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(min_len).unwrap();
        let mean2 = buffer2
            .iter()
            .take(min_len)
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(min_len).unwrap();

        let mut numerator = F::zero();
        let mut sum1_sq = F::zero();
        let mut sum2_sq = F::zero();

        for i in 0..min_len {
            let diff1 = buffer1[i] - mean1;
            let diff2 = buffer2[i] - mean2;
            numerator = numerator + diff1 * diff2;
            sum1_sq = sum1_sq + diff1 * diff1;
            sum2_sq = sum2_sq + diff2 * diff2;
        }

        let denominator = (sum1_sq * sum2_sq).sqrt();
        if denominator > F::epsilon() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::<f64>::new();

        // Add some data points
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for &val in &data {
            stats.update(val);
        }

        assert_eq!(stats.count(), 5);
        assert_abs_diff_eq!(stats.mean(), 3.0);
        assert_abs_diff_eq!(stats.min(), 1.0);
        assert_abs_diff_eq!(stats.max(), 5.0);
        assert!(stats.variance() > 0.0);
    }

    #[test]
    fn test_ewma() {
        let mut ewma = EWMA::<f64>::new(0.3).unwrap();

        ewma.update(10.0);
        assert_abs_diff_eq!(ewma.value().unwrap(), 10.0);

        ewma.update(20.0);
        let expected = 0.3 * 20.0 + 0.7 * 10.0;
        assert_abs_diff_eq!(ewma.value().unwrap(), expected);
    }

    #[test]
    fn test_cusum_detector() {
        let mut cusum = CusumDetector::<f64>::new(10.0, 2.0); // Higher threshold and drift

        // Add normal data around mean 5
        for i in 0..10 {
            let value = 5.0 + (i as f64 % 2.0) - 0.5; // Values oscillate around 5
            let change = cusum.update(value);
            assert!(change.is_none());
        }

        // Add data with mean shift
        for i in 0..10 {
            let change = cusum.update(10.0 + i as f64);
            if change.is_some() {
                assert!(matches!(change.unwrap().change_type, ChangeType::MeanShift));
                break;
            }
        }
    }

    #[test]
    fn test_streaming_analyzer() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        // Add some observations
        for i in 0..50 {
            let value = (i as f64).sin();
            analyzer.add_observation(value).unwrap();
        }

        assert!(analyzer.observation_count() > 0);
        assert!(analyzer.get_stats().count() > 0);

        // Test forecasting
        let forecast = analyzer.forecast(5);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 5);
    }

    #[test]
    fn test_multi_series_analyzer() {
        let config = StreamConfig::default();
        let mut multi_analyzer = MultiSeriesAnalyzer::<f64>::new(config);

        // Add two series
        multi_analyzer.add_series("series1".to_string()).unwrap();
        multi_analyzer.add_series("series2".to_string()).unwrap();

        // Add data to both series
        for i in 0..20 {
            multi_analyzer.add_observation("series1", i as f64).unwrap();
            multi_analyzer
                .add_observation("series2", (i as f64) * 2.0)
                .unwrap();
        }

        // Check correlation
        let correlation = multi_analyzer
            .get_correlation("series1", "series2")
            .unwrap();
        assert!(correlation > 0.5); // Should be highly correlated
    }

    #[test]
    fn test_outlier_detection() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();

        // Add normal data
        for i in 0..20 {
            analyzer.add_observation(i as f64).unwrap();
        }

        // Check if a clear outlier is detected
        assert!(analyzer.is_outlier(1000.0));
        assert!(!analyzer.is_outlier(20.0)); // Should be normal
    }
}

/// Real-time feature engineering for streaming data
pub mod feature_engineering {
    use super::*;
    use ndarray::Array1;
    use std::collections::{HashMap, VecDeque};

    /// Real-time feature extractor with sliding window
    #[derive(Debug)]
    pub struct StreamingFeatureExtractor<F: Float + Debug> {
        /// Window buffer for feature computation
        window_buffer: VecDeque<F>,
        /// Maximum window size
        max_window_size: usize,
        /// Feature configurations
        feature_configs: Vec<FeatureConfig>,
        /// Cached statistics for efficiency
        cached_stats: HashMap<String, F>,
        /// Feature history for temporal features
        feature_history: VecDeque<Array1<F>>,
        /// Update frequency for expensive features
        #[allow(dead_code)]
        update_frequency: usize,
        /// Current update counter
        update_counter: usize,
    }

    #[derive(Debug, Clone)]
    /// Configuration for feature extraction
    pub struct FeatureConfig {
        /// Feature name
        pub name: String,
        /// Type of feature
        pub feature_type: FeatureType,
        /// Window size for feature calculation
        pub window_size: usize,
        /// How often to update the feature (in number of samples)
        pub update_frequency: usize,
        /// Whether this feature is enabled for extraction
        pub enabled: bool,
    }

    #[derive(Debug, Clone)]
    /// Types of features that can be extracted from time series data
    pub enum FeatureType {
        /// Basic statistical features
        Mean,
        /// Variance of the time series
        Variance,
        /// Skewness of the time series
        Skewness,
        /// Kurtosis of the time series
        Kurtosis,
        /// Minimum value in the time series
        Min,
        /// Maximum value in the time series
        Max,
        /// Range (max - min) of the time series
        Range,
        /// Advanced statistical features
        Quantile(f64),
        /// Interquartile range (Q3 - Q1)
        InterquartileRange,
        /// Coefficient of variation (std/mean)
        CoefficientOfVariation,
        /// Temporal features
        LinearTrend,
        /// Strength of seasonal component
        SeasonalStrength,
        /// Strength of trend component
        TrendStrength,
        /// Frequency domain features
        SpectralCentroid,
        /// Spectral rolloff point
        SpectralRolloff,
        /// Spectral flatness measure
        SpectralFlatness,
        /// Complexity features
        SampleEntropy,
        /// Lempel-Ziv complexity measure
        LempelZivComplexity,
        /// Fractal dimension of the time series
        FractalDimension,
        /// Technical indicators
        RelativeStrengthIndex,
        /// Moving Average Convergence Divergence (MACD)
        MovingAverageConvergenceDivergence,
        /// Bollinger Bands indicator
        BollingerBands,
        /// Stochastic oscillator indicator
        StochasticOscillator,
        /// Pattern-based features
        PatternRecognition,
        /// Frequency of recurring motifs
        MotifFrequency,
        /// Distance to learned shapelet patterns
        ShapeletDistance,
    }

    impl<F: Float + Debug + Clone + FromPrimitive> StreamingFeatureExtractor<F> {
        /// Create new streaming feature extractor
        pub fn new(_max_window_size: usize, featureconfigs: Vec<FeatureConfig>) -> Self {
            Self {
                window_buffer: VecDeque::with_capacity(_max_window_size),
                max_window_size: _max_window_size,
                feature_configs: featureconfigs,
                cached_stats: HashMap::new(),
                feature_history: VecDeque::with_capacity(100),
                update_frequency: 10,
                update_counter: 0,
            }
        }

        /// Add new observation and extract features
        pub fn extract_features(&mut self, value: F) -> Result<Array1<F>> {
            // Add to window buffer
            if self.window_buffer.len() >= self.max_window_size {
                self.window_buffer.pop_front();
            }
            self.window_buffer.push_back(value);

            self.update_counter += 1;

            // Extract features based on configurations
            let mut features = Vec::new();

            for config in &self.feature_configs {
                if !config.enabled {
                    continue;
                }

                if self.window_buffer.len() < config.window_size {
                    features.push(F::zero());
                    continue;
                }

                // Only update expensive features at specified frequency
                if config.update_frequency > 1
                    && self.update_counter % config.update_frequency != 0
                    && self.cached_stats.contains_key(&config.name)
                {
                    features.push(self.cached_stats[&config.name]);
                    continue;
                }

                let window_data: Vec<F> = self
                    .window_buffer
                    .iter()
                    .rev()
                    .take(config.window_size)
                    .rev()
                    .cloned()
                    .collect();

                let feature_value = self.compute_feature(&config.feature_type, &window_data)?;
                self.cached_stats.insert(config.name.clone(), feature_value);
                features.push(feature_value);
            }

            let feature_array = Array1::from_vec(features);

            // Store in feature history
            if self.feature_history.len() >= 100 {
                self.feature_history.pop_front();
            }
            self.feature_history.push_back(feature_array.clone());

            Ok(feature_array)
        }

        /// Compute specific feature type
        fn compute_feature(&self, featuretype: &FeatureType, data: &[F]) -> Result<F> {
            if data.is_empty() {
                return Ok(F::zero());
            }

            match featuretype {
                FeatureType::Mean => {
                    let sum = data.iter().fold(F::zero(), |acc, &x| acc + x);
                    Ok(sum / F::from(data.len()).unwrap())
                }
                FeatureType::Variance => {
                    let mean = data.iter().fold(F::zero(), |acc, &x| acc + x)
                        / F::from(data.len()).unwrap();
                    let var = data
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / F::from(data.len()).unwrap();
                    Ok(var)
                }
                FeatureType::Skewness => self.compute_skewness(data),
                FeatureType::Kurtosis => self.compute_kurtosis(data),
                FeatureType::Min => Ok(data.iter().fold(F::infinity(), |acc, &x| acc.min(x))),
                FeatureType::Max => Ok(data.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))),
                FeatureType::Range => {
                    let min_val = data.iter().fold(F::infinity(), |acc, &x| acc.min(x));
                    let max_val = data.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
                    Ok(max_val - min_val)
                }
                FeatureType::Quantile(q) => self.compute_quantile(data, *q),
                FeatureType::InterquartileRange => {
                    let q25 = self.compute_quantile(data, 0.25)?;
                    let q75 = self.compute_quantile(data, 0.75)?;
                    Ok(q75 - q25)
                }
                FeatureType::CoefficientOfVariation => {
                    let mean = data.iter().fold(F::zero(), |acc, &x| acc + x)
                        / F::from(data.len()).unwrap();
                    let var = data
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / F::from(data.len()).unwrap();
                    let std_dev = var.sqrt();
                    if mean.abs() > F::zero() {
                        Ok(std_dev / mean.abs())
                    } else {
                        Ok(F::zero())
                    }
                }
                FeatureType::LinearTrend => self.compute_linear_trend(data),
                FeatureType::SeasonalStrength => self.compute_seasonal_strength(data),
                FeatureType::TrendStrength => self.compute_trend_strength(data),
                FeatureType::SpectralCentroid => self.compute_spectral_centroid(data),
                FeatureType::SpectralRolloff => self.compute_spectral_rolloff(data, 0.85),
                FeatureType::SpectralFlatness => self.compute_spectral_flatness(data),
                FeatureType::SampleEntropy => self.compute_sample_entropy(data, 2, 0.2),
                FeatureType::LempelZivComplexity => self.compute_lempel_ziv_complexity(data),
                FeatureType::FractalDimension => self.compute_fractal_dimension(data),
                FeatureType::RelativeStrengthIndex => self.compute_rsi(data, 14),
                FeatureType::MovingAverageConvergenceDivergence => self.compute_macd(data),
                FeatureType::BollingerBands => self.compute_bollinger_bands(data, 20, 2.0),
                FeatureType::StochasticOscillator => self.compute_stochastic_oscillator(data, 14),
                FeatureType::PatternRecognition => self.compute_pattern_recognition(data),
                FeatureType::MotifFrequency => self.compute_motif_frequency(data),
                FeatureType::ShapeletDistance => self.computeshapelet_distance(data),
            }
        }

        /// Compute skewness
        fn compute_skewness(&self, data: &[F]) -> Result<F> {
            if data.len() < 3 {
                return Ok(F::zero());
            }

            let n = F::from(data.len()).unwrap();
            let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;

            let m2 = data
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / n;

            let m3 = data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / n;

            if m2 > F::zero() {
                Ok(m3 / m2.powf(F::from(1.5).unwrap()))
            } else {
                Ok(F::zero())
            }
        }

        /// Compute kurtosis
        fn compute_kurtosis(&self, data: &[F]) -> Result<F> {
            if data.len() < 4 {
                return Ok(F::zero());
            }

            let n = F::from(data.len()).unwrap();
            let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;

            let m2 = data
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / n;

            let m4 = data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    let sq = diff * diff;
                    sq * sq
                })
                .fold(F::zero(), |acc, x| acc + x)
                / n;

            if m2 > F::zero() {
                Ok(m4 / (m2 * m2) - F::from(3.0).unwrap())
            } else {
                Ok(F::zero())
            }
        }

        /// Compute quantile
        fn compute_quantile(&self, data: &[F], q: f64) -> Result<F> {
            if data.is_empty() {
                return Ok(F::zero());
            }

            let mut sorted_data = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let index = (q * (sorted_data.len() - 1) as f64) as usize;
            let index = index.min(sorted_data.len() - 1);

            Ok(sorted_data[index])
        }

        /// Compute linear trend
        fn compute_linear_trend(&self, data: &[F]) -> Result<F> {
            if data.len() < 2 {
                return Ok(F::zero());
            }

            let n = F::from(data.len()).unwrap();
            let x_mean = (n - F::one()) / F::from(2).unwrap();
            let y_mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;

            let mut numerator = F::zero();
            let mut denominator = F::zero();

            for (i, &y) in data.iter().enumerate() {
                let x = F::from(i).unwrap();
                let x_diff = x - x_mean;
                numerator = numerator + x_diff * (y - y_mean);
                denominator = denominator + x_diff * x_diff;
            }

            if denominator > F::zero() {
                Ok(numerator / denominator)
            } else {
                Ok(F::zero())
            }
        }

        /// Compute seasonal strength (simplified)
        fn compute_seasonal_strength(&self, data: &[F]) -> Result<F> {
            if data.len() < 12 {
                return Ok(F::zero());
            }

            // Simple seasonal decomposition using 12-period seasonality
            let period = 12.min(data.len() / 2);
            let mut seasonal_component = vec![F::zero(); period];
            let mut counts = vec![0; period];

            for (i, &value) in data.iter().enumerate() {
                let season_idx = i % period;
                seasonal_component[season_idx] = seasonal_component[season_idx] + value;
                counts[season_idx] += 1;
            }

            // Average seasonal components
            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    seasonal_component[i] = seasonal_component[i] / F::from(*count).unwrap();
                }
            }

            // Compute seasonal variance
            let seasonal_mean = seasonal_component.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(period).unwrap();
            let seasonal_var = seasonal_component
                .iter()
                .map(|&x| (x - seasonal_mean) * (x - seasonal_mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(period).unwrap();

            // Total variance
            let total_mean =
                data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
            let total_var = data
                .iter()
                .map(|&x| (x - total_mean) * (x - total_mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(data.len()).unwrap();

            if total_var > F::zero() {
                Ok(seasonal_var / total_var)
            } else {
                Ok(F::zero())
            }
        }

        /// Compute trend strength
        fn compute_trend_strength(&self, data: &[F]) -> Result<F> {
            if data.len() < 3 {
                return Ok(F::zero());
            }

            // Simple trend using moving average
            let window_size = 3.min(data.len());
            let mut trend = Vec::new();

            for i in 0..=data.len() - window_size {
                let avg = data[i..i + window_size]
                    .iter()
                    .fold(F::zero(), |acc, &x| acc + x)
                    / F::from(window_size).unwrap();
                trend.push(avg);
            }

            if trend.len() < 2 {
                return Ok(F::zero());
            }

            // Compute trend variance
            let trend_mean =
                trend.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(trend.len()).unwrap();
            let trend_var = trend
                .iter()
                .map(|&x| (x - trend_mean) * (x - trend_mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(trend.len()).unwrap();

            // Total variance
            let total_mean =
                data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
            let total_var = data
                .iter()
                .map(|&x| (x - total_mean) * (x - total_mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(data.len()).unwrap();

            if total_var > F::zero() {
                Ok(trend_var / total_var)
            } else {
                Ok(F::zero())
            }
        }

        /// Compute spectral centroid (simplified)
        fn compute_spectral_centroid(&self, data: &[F]) -> Result<F> {
            if data.len() < 4 {
                return Ok(F::zero());
            }

            // Simple magnitude spectrum approximation
            let mut magnitude_sum = F::zero();
            let mut weighted_sum = F::zero();

            for (i, &value) in data.iter().enumerate() {
                let magnitude = value.abs();
                magnitude_sum = magnitude_sum + magnitude;
                weighted_sum = weighted_sum + F::from(i).unwrap() * magnitude;
            }

            if magnitude_sum > F::zero() {
                Ok(weighted_sum / magnitude_sum)
            } else {
                Ok(F::zero())
            }
        }

        /// Compute spectral rolloff
        fn compute_spectral_rolloff(&self, data: &[F], rolloffthreshold: f64) -> Result<F> {
            if data.is_empty() {
                return Ok(F::zero());
            }

            let magnitudes: Vec<F> = data.iter().map(|&x| x.abs()).collect();
            let total_energy: F = magnitudes.iter().fold(F::zero(), |acc, &x| acc + x * x);
            let _threshold = F::from(rolloffthreshold).unwrap() * total_energy;

            let mut cumulative_energy = F::zero();
            for (i, &magnitude) in magnitudes.iter().enumerate() {
                cumulative_energy = cumulative_energy + magnitude * magnitude;
                if cumulative_energy >= _threshold {
                    return Ok(F::from(i).unwrap() / F::from(data.len()).unwrap());
                }
            }

            Ok(F::one())
        }

        /// Compute spectral flatness
        fn compute_spectral_flatness(&self, data: &[F]) -> Result<F> {
            if data.is_empty() {
                return Ok(F::zero());
            }

            let magnitudes: Vec<F> = data.iter().map(|&x| x.abs()).collect();

            // Geometric mean
            let log_sum = magnitudes
                .iter()
                .filter(|&&x| x > F::zero())
                .map(|&x| x.ln())
                .fold(F::zero(), |acc, x| acc + x);
            let geometric_mean = (log_sum / F::from(magnitudes.len()).unwrap()).exp();

            // Arithmetic mean
            let arithmetic_mean = magnitudes.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(magnitudes.len()).unwrap();

            if arithmetic_mean > F::zero() {
                Ok(geometric_mean / arithmetic_mean)
            } else {
                Ok(F::zero())
            }
        }

        /// Compute sample entropy (simplified)
        fn compute_sample_entropy(&self, data: &[F], m: usize, r: f64) -> Result<F> {
            if data.len() < m + 1 {
                return Ok(F::zero());
            }

            let tolerance = F::from(r).unwrap() * self.compute_std(data);
            let mut matches_m = 0;
            let mut matches_m_plus_1 = 0;

            for i in 0..data.len() - m {
                for j in i + 1..data.len() - m {
                    let mut match_m = true;
                    let mut match_m_plus_1 = true;

                    // Check m-length patterns
                    for k in 0..m {
                        if (data[i + k] - data[j + k]).abs() > tolerance {
                            match_m = false;
                            break;
                        }
                    }

                    if match_m {
                        matches_m += 1;

                        // Check (m+1)-length patterns
                        if i + m < data.len() && j + m < data.len() {
                            if (data[i + m] - data[j + m]).abs() > tolerance {
                                match_m_plus_1 = false;
                            }
                            if match_m_plus_1 {
                                matches_m_plus_1 += 1;
                            }
                        }
                    }
                }
            }

            if matches_m > 0 && matches_m_plus_1 > 0 {
                let phi_m = F::from(matches_m).unwrap();
                let phi_m_plus_1 = F::from(matches_m_plus_1).unwrap();
                Ok(-(phi_m_plus_1 / phi_m).ln())
            } else {
                Ok(F::zero())
            }
        }

        /// Compute standard deviation helper
        fn compute_std(&self, data: &[F]) -> F {
            if data.is_empty() {
                return F::zero();
            }

            let mean =
                data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
            let variance = data
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(data.len()).unwrap();
            variance.sqrt()
        }

        /// Compute Lempel-Ziv complexity (simplified)
        fn compute_lempel_ziv_complexity(&self, data: &[F]) -> Result<F> {
            if data.is_empty() {
                return Ok(F::zero());
            }

            // Convert to binary string (simplified)
            let median = self.compute_quantile(data, 0.5)?;
            let binary_string: Vec<u8> = data
                .iter()
                .map(|&x| if x >= median { 1 } else { 0 })
                .collect();

            let mut complexity = 0;
            let mut i = 0;

            while i < binary_string.len() {
                let mut j = 1;
                while i + j <= binary_string.len() {
                    let pattern = &binary_string[i..i + j];
                    let mut found = false;

                    // Check if pattern exists in previous part
                    for k in 0..i {
                        if k + j <= i && binary_string[k..k + j] == *pattern {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        complexity += 1;
                        i += j;
                        break;
                    }
                    j += 1;
                }

                if i + j > binary_string.len() {
                    complexity += 1;
                    break;
                }
            }

            Ok(F::from(complexity).unwrap())
        }

        /// Compute fractal dimension (simplified box-counting)
        fn compute_fractal_dimension(&self, data: &[F]) -> Result<F> {
            if data.len() < 4 {
                return Ok(F::zero());
            }

            // Simple fractal dimension using variation method
            let mut variations = Vec::new();
            let mut scale = 1;

            while scale < data.len() / 2 {
                let mut variation = F::zero();
                let mut count = 0;

                for i in 0..data.len() - scale {
                    variation = variation + (data[i + scale] - data[i]).abs();
                    count += 1;
                }

                if count > 0 {
                    let avg_variation = variation / F::from(count).unwrap();
                    variations.push((F::from(scale).unwrap(), avg_variation));
                }

                scale *= 2;
            }

            if variations.len() < 2 {
                return Ok(F::one());
            }

            // Simple linear regression on log-log plot
            let _log_scales: Vec<F> = variations.iter().map(|(s_, _)| s_.ln()).collect();
            let log_variations: Vec<F> = variations.iter().map(|(_, v)| v.ln()).collect();

            let slope = self.compute_linear_trend(&log_variations)?;
            Ok(F::one() + slope) // Fractal dimension approximation
        }

        /// Compute RSI (Relative Strength Index)
        fn compute_rsi(&self, data: &[F], period: usize) -> Result<F> {
            if data.len() < period + 1 {
                return Ok(F::from(50).unwrap()); // Neutral RSI
            }

            let mut gains = Vec::new();
            let mut losses = Vec::new();

            for i in 1..data.len() {
                let change = data[i] - data[i - 1];
                if change > F::zero() {
                    gains.push(change);
                    losses.push(F::zero());
                } else {
                    gains.push(F::zero());
                    losses.push(-change);
                }
            }

            if gains.len() < period {
                return Ok(F::from(50).unwrap());
            }

            // Average gain/loss over period
            let avg_gain = gains
                .iter()
                .rev()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from(period).unwrap();
            let avg_loss = losses
                .iter()
                .rev()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from(period).unwrap();

            if avg_loss == F::zero() {
                return Ok(F::from(100).unwrap());
            }

            let rs = avg_gain / avg_loss;
            let rsi = F::from(100).unwrap() - (F::from(100).unwrap() / (F::one() + rs));

            Ok(rsi)
        }

        /// Compute MACD (simplified)
        fn compute_macd(&self, data: &[F]) -> Result<F> {
            if data.len() < 26 {
                return Ok(F::zero());
            }

            // Simple moving averages (simplified MACD)
            let ema_12 = data
                .iter()
                .rev()
                .take(12)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from(12).unwrap();
            let ema_26 = data
                .iter()
                .rev()
                .take(26)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from(26).unwrap();

            Ok(ema_12 - ema_26)
        }

        /// Compute Bollinger Bands position
        fn compute_bollinger_bands(&self, data: &[F], period: usize, numstd: f64) -> Result<F> {
            if data.len() < period {
                return Ok(F::from(0.5).unwrap()); // Middle position
            }

            let window = &data[data.len() - period..];
            let mean = window.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(period).unwrap();
            let std_dev = self.compute_std(window);

            let upper_band = mean + F::from(numstd).unwrap() * std_dev;
            let lower_band = mean - F::from(numstd).unwrap() * std_dev;

            let current_price = data[data.len() - 1];

            if upper_band == lower_band {
                Ok(F::from(0.5).unwrap())
            } else {
                let position = (current_price - lower_band) / (upper_band - lower_band);
                Ok(position.max(F::zero()).min(F::one()))
            }
        }

        /// Compute Stochastic Oscillator
        fn compute_stochastic_oscillator(&self, data: &[F], period: usize) -> Result<F> {
            if data.len() < period {
                return Ok(F::from(50).unwrap());
            }

            let window = &data[data.len() - period..];
            let highest = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            let lowest = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let current = data[data.len() - 1];

            if highest == lowest {
                Ok(F::from(50).unwrap())
            } else {
                let k_percent = F::from(100).unwrap() * (current - lowest) / (highest - lowest);
                Ok(k_percent)
            }
        }

        /// Compute pattern recognition score (simplified)
        fn compute_pattern_recognition(&self, data: &[F]) -> Result<F> {
            if data.len() < 5 {
                return Ok(F::zero());
            }

            // Simple pattern: detect if current trend matches historical patterns
            let recent_trend = self.compute_linear_trend(&data[data.len().saturating_sub(5)..])?;

            // Compare with historical trends in feature history
            if self.feature_history.len() < 2 {
                return Ok(F::zero());
            }

            let mut pattern_score = F::zero();
            let mut count = 0;

            for historical_features in &self.feature_history {
                if !historical_features.is_empty() {
                    // Simple correlation-based pattern matching
                    let historical_trend = historical_features[0]; // Assuming first feature is trend
                    let similarity = F::one() - (recent_trend - historical_trend).abs();
                    pattern_score = pattern_score + similarity.max(F::zero());
                    count += 1;
                }
            }

            if count > 0 {
                Ok(pattern_score / F::from(count).unwrap())
            } else {
                Ok(F::zero())
            }
        }

        /// Compute motif frequency (simplified)
        fn compute_motif_frequency(&self, data: &[F]) -> Result<F> {
            if data.len() < 6 {
                return Ok(F::zero());
            }

            let motif_length = 3;
            let mut motif_counts = std::collections::HashMap::new();

            // Extract overlapping motifs
            for i in 0..=data.len() - motif_length {
                let motif = &data[i..i + motif_length];

                // Discretize motif (simple bucketing)
                let discretized: Vec<u8> = motif
                    .iter()
                    .map(|&x| ((x.to_f64().unwrap_or(0.0) * 10.0) as i32).clamp(0, 9) as u8)
                    .collect();

                *motif_counts.entry(discretized).or_insert(0) += 1;
            }

            // Find most frequent motif
            let max_frequency = motif_counts.values().max().unwrap_or(&0);
            Ok(F::from(*max_frequency).unwrap()
                / F::from(data.len().saturating_sub(motif_length - 1)).unwrap())
        }

        /// Compute shapelet distance (simplified)
        fn computeshapelet_distance(&self, data: &[F]) -> Result<F> {
            if data.len() < 4 {
                return Ok(F::zero());
            }

            // Simple shapelet: use a canonical pattern (e.g., increasing trend)
            let shapelet_length = 4.min(data.len());
            let canonicalshapelet: Vec<F> =
                (0..shapelet_length).map(|i| F::from(i).unwrap()).collect();

            let mut min_distance = F::infinity();

            // Find minimum distance to canonical shapelet
            for i in 0..=data.len() - shapelet_length {
                let subsequence = &data[i..i + shapelet_length];

                // Normalize both sequences
                let subseq_mean = subsequence.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(shapelet_length).unwrap();
                let subseq_std = {
                    let var = subsequence
                        .iter()
                        .map(|&x| (x - subseq_mean) * (x - subseq_mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / F::from(shapelet_length).unwrap();
                    var.sqrt()
                };

                let canonical_mean = canonicalshapelet.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(shapelet_length).unwrap();
                let canonical_std = {
                    let var = canonicalshapelet
                        .iter()
                        .map(|&x| (x - canonical_mean) * (x - canonical_mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / F::from(shapelet_length).unwrap();
                    var.sqrt()
                };

                // Compute normalized Euclidean distance
                let mut distance = F::zero();
                for j in 0..shapelet_length {
                    let norm_subseq = if subseq_std > F::zero() {
                        (subsequence[j] - subseq_mean) / subseq_std
                    } else {
                        F::zero()
                    };
                    let norm_canonical = if canonical_std > F::zero() {
                        (canonicalshapelet[j] - canonical_mean) / canonical_std
                    } else {
                        F::zero()
                    };
                    distance =
                        distance + (norm_subseq - norm_canonical) * (norm_subseq - norm_canonical);
                }

                min_distance = min_distance.min(distance.sqrt());
            }

            Ok(min_distance)
        }

        /// Get available feature names
        pub fn get_feature_names(&self) -> Vec<String> {
            self.feature_configs
                .iter()
                .filter(|config| config.enabled)
                .map(|config| config.name.clone())
                .collect()
        }

        /// Get current feature history
        pub fn get_feature_history(&self) -> &VecDeque<Array1<F>> {
            &self.feature_history
        }
    }
}

/// Adaptive models for streaming time series
pub mod adaptive {
    use super::*;
    use ndarray::Array1;

    /// Adaptive linear regression with forgetting factor
    #[derive(Debug)]
    pub struct AdaptiveLinearRegression<F: Float + Debug> {
        /// Regression coefficients
        coefficients: Array1<F>,
        /// Covariance matrix
        covariance: Array2<F>,
        /// Forgetting factor (0 < lambda <= 1)
        forgetting_factor: F,
        /// Regularization parameter
        regularization: F,
        /// Number of features
        num_features: usize,
        /// Update counter
        update_count: usize,
    }

    impl<F: Float + Debug + Clone + FromPrimitive> AdaptiveLinearRegression<F> {
        /// Create new adaptive linear regression
        pub fn new(_num_features: usize, forgettingfactor: F, regularization: F) -> Result<Self> {
            if forgettingfactor <= F::zero() || forgettingfactor > F::one() {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "forgetting_factor".to_string(),
                    message: "Forgetting _factor must be in (0, 1]".to_string(),
                });
            }

            let mut covariance = Array2::zeros((_num_features, _num_features));
            let identity_scale = F::from(1000.0).unwrap(); // Large initial uncertainty
            for i in 0.._num_features {
                covariance[[i, i]] = identity_scale;
            }

            Ok(Self {
                coefficients: Array1::zeros(_num_features),
                covariance,
                forgetting_factor: forgettingfactor,
                regularization,
                num_features: _num_features,
                update_count: 0,
            })
        }

        /// Update model with new observation using Recursive Least Squares
        pub fn update(&mut self, features: &Array1<F>, target: F) -> Result<()> {
            if features.len() != self.num_features {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: self.num_features,
                    actual: features.len(),
                });
            }

            self.update_count += 1;

            // Compute prediction error
            let prediction = self.predict(features)?;
            let error = target - prediction;

            // RLS update
            let mut temp_vector = Array1::zeros(self.num_features);
            for i in 0..self.num_features {
                let mut sum = F::zero();
                for j in 0..self.num_features {
                    sum = sum + self.covariance[[i, j]] * features[j];
                }
                temp_vector[i] = sum;
            }

            let mut denominator = self.forgetting_factor;
            for i in 0..self.num_features {
                denominator = denominator + features[i] * temp_vector[i];
            }

            // Kalman gain
            let mut gain = Array1::zeros(self.num_features);
            for i in 0..self.num_features {
                gain[i] = temp_vector[i] / denominator;
            }

            // Update coefficients
            for i in 0..self.num_features {
                self.coefficients[i] = self.coefficients[i] + gain[i] * error;
            }

            // Update covariance matrix
            let mut new_covariance = Array2::zeros((self.num_features, self.num_features));
            for i in 0..self.num_features {
                for j in 0..self.num_features {
                    let _update_term = gain[i] * features[j];
                    new_covariance[[i, j]] = (self.covariance[[i, j]]
                        - temp_vector[i] * features[j])
                        / self.forgetting_factor;

                    // Add regularization
                    if i == j {
                        new_covariance[[i, j]] = new_covariance[[i, j]] + self.regularization;
                    }
                }
            }
            self.covariance = new_covariance;

            Ok(())
        }

        /// Make prediction
        pub fn predict(&self, features: &Array1<F>) -> Result<F> {
            if features.len() != self.num_features {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: self.num_features,
                    actual: features.len(),
                });
            }

            let mut prediction = F::zero();
            for i in 0..self.num_features {
                prediction = prediction + self.coefficients[i] * features[i];
            }

            Ok(prediction)
        }

        /// Get prediction with confidence interval
        pub fn predict_with_confidence(
            &self,
            features: &Array1<F>,
            _confidence_level: F,
        ) -> Result<(F, F, F)> {
            let prediction = self.predict(features)?;

            // Compute prediction variance
            let mut variance = F::zero();
            for i in 0..self.num_features {
                for j in 0..self.num_features {
                    variance = variance + features[i] * self.covariance[[i, j]] * features[j];
                }
            }

            let std_dev = variance.sqrt();
            let z_score = F::from(1.96).unwrap(); // 95% confidence interval

            let lower_bound = prediction - z_score * std_dev;
            let upper_bound = prediction + z_score * std_dev;

            Ok((prediction, lower_bound, upper_bound))
        }

        /// Get current coefficients
        pub fn get_coefficients(&self) -> &Array1<F> {
            &self.coefficients
        }

        /// Get model confidence (trace of covariance matrix)
        pub fn get_confidence(&self) -> F {
            let mut trace = F::zero();
            for i in 0..self.num_features {
                trace = trace + self.covariance[[i, i]];
            }
            trace / F::from(self.num_features).unwrap()
        }
    }

    /// Adaptive ARIMA model with online parameter estimation
    #[derive(Debug)]
    pub struct AdaptiveARIMA<F: Float + Debug> {
        /// AR order
        p: usize,
        /// Differencing order
        d: usize,
        /// MA order
        q: usize,
        /// AR coefficients
        ar_coeffs: Array1<F>,
        /// MA coefficients
        ma_coeffs: Array1<F>,
        /// Recent observations
        observations: VecDeque<F>,
        /// Recent residuals
        residuals: VecDeque<F>,
        /// Learning rate
        learning_rate: F,
        /// Model has been initialized
        initialized: bool,
    }

    impl<F: Float + Debug + Clone + FromPrimitive> AdaptiveARIMA<F> {
        /// Create new adaptive ARIMA model
        pub fn new(p: usize, d: usize, q: usize, learningrate: F) -> Self {
            Self {
                p,
                d,
                q,
                ar_coeffs: Array1::zeros(p),
                ma_coeffs: Array1::zeros(q),
                observations: VecDeque::with_capacity(100),
                residuals: VecDeque::with_capacity(100),
                learning_rate: learningrate,
                initialized: false,
            }
        }

        /// Update model with new observation
        pub fn update(&mut self, observation: F) -> Result<()> {
            // Add observation to buffer
            if self.observations.len() >= 100 {
                self.observations.pop_front();
            }
            self.observations.push_back(observation);

            // Apply differencing if needed
            let processed_obs = if self.d > 0 && self.observations.len() > self.d {
                self.apply_differencing(observation)
            } else {
                observation
            };

            // Initialize model if we have enough data
            if !self.initialized && self.observations.len() >= self.p.max(self.q) + self.d + 10 {
                self.initialize_parameters()?;
                self.initialized = true;
            }

            if self.initialized {
                // Make prediction and compute residual
                let prediction = self.predict_next()?;
                let residual = processed_obs - prediction;

                // Update residuals buffer
                if self.residuals.len() >= 100 {
                    self.residuals.pop_front();
                }
                self.residuals.push_back(residual);

                // Update parameters using gradient descent
                self.update_parameters(processed_obs, residual)?;
            }

            Ok(())
        }

        /// Apply differencing to observation
        fn apply_differencing(&self, observation: F) -> F {
            let len = self.observations.len();
            if len <= self.d {
                return observation;
            }

            let mut diff_obs = observation;
            for _ in 0..self.d {
                diff_obs = diff_obs - self.observations[len - 1];
            }
            diff_obs
        }

        /// Initialize parameters using method of moments
        fn initialize_parameters(&mut self) -> Result<()> {
            let processed_data: Vec<F> = if self.d > 0 {
                self.apply_differencing_to_series()
            } else {
                self.observations.iter().cloned().collect()
            };

            if processed_data.len() < self.p.max(self.q) + 5 {
                return Ok(());
            }

            // Simple initialization using autocorrelations
            self.initialize_ar_parameters(&processed_data)?;
            self.initialize_ma_parameters(&processed_data)?;

            Ok(())
        }

        /// Apply differencing to entire series
        fn apply_differencing_to_series(&self) -> Vec<F> {
            let mut series: Vec<F> = self.observations.iter().cloned().collect();

            for _ in 0..self.d {
                let mut diff_series = Vec::new();
                for i in 1..series.len() {
                    diff_series.push(series[i] - series[i - 1]);
                }
                series = diff_series;
            }

            series
        }

        /// Initialize AR parameters using Yule-Walker equations
        fn initialize_ar_parameters(&mut self, data: &[F]) -> Result<()> {
            if self.p == 0 || data.len() < self.p + 1 {
                return Ok(());
            }

            // Compute autocorrelations
            let mut autocorrs = vec![F::zero(); self.p + 1];
            for lag in 0..=self.p {
                let mut sum = F::zero();
                let mut count = 0;

                for i in lag..data.len() {
                    sum = sum + data[i] * data[i - lag];
                    count += 1;
                }

                if count > 0 {
                    autocorrs[lag] = sum / F::from(count).unwrap();
                }
            }

            // Solve Yule-Walker equations (simplified)
            for i in 0..self.p {
                let mut coeff = F::zero();
                if autocorrs[0] > F::zero() {
                    coeff = autocorrs[i + 1] / autocorrs[0];
                }
                self.ar_coeffs[i] = coeff
                    .max(F::from(-0.99).unwrap())
                    .min(F::from(0.99).unwrap());
            }

            Ok(())
        }

        /// Initialize MA parameters (simplified)
        fn initialize_ma_parameters(&mut self, data: &[F]) -> Result<()> {
            // Simple initialization: small random values
            for i in 0..self.q {
                self.ma_coeffs[i] = F::from(0.1).unwrap() * F::from((i + 1) as f64 * 0.1).unwrap();
            }
            Ok(())
        }

        /// Predict next value
        fn predict_next(&self) -> Result<F> {
            if !self.initialized {
                return Ok(F::zero());
            }

            let processed_data: Vec<F> = if self.d > 0 {
                self.apply_differencing_to_series()
            } else {
                self.observations.iter().cloned().collect()
            };

            let mut prediction = F::zero();

            // AR component
            for i in 0..self.p {
                if i < processed_data.len() {
                    let lag_index = processed_data.len() - 1 - i;
                    prediction = prediction + self.ar_coeffs[i] * processed_data[lag_index];
                }
            }

            // MA component
            for i in 0..self.q {
                if i < self.residuals.len() {
                    let lag_index = self.residuals.len() - 1 - i;
                    prediction = prediction + self.ma_coeffs[i] * self.residuals[lag_index];
                }
            }

            Ok(prediction)
        }

        /// Update parameters using gradient descent
        fn update_parameters(&mut self, observation: F, residual: F) -> Result<()> {
            let processed_data: Vec<F> = if self.d > 0 {
                self.apply_differencing_to_series()
            } else {
                self.observations.iter().cloned().collect()
            };

            // Update AR coefficients
            for i in 0..self.p {
                if i < processed_data.len() {
                    let lag_index = processed_data.len() - 1 - i;
                    let gradient = residual * processed_data[lag_index];
                    self.ar_coeffs[i] = self.ar_coeffs[i] + self.learning_rate * gradient;

                    // Keep coefficients stable
                    self.ar_coeffs[i] = self.ar_coeffs[i]
                        .max(F::from(-0.99).unwrap())
                        .min(F::from(0.99).unwrap());
                }
            }

            // Update MA coefficients
            for i in 0..self.q {
                if i < self.residuals.len() {
                    let lag_index = self.residuals.len() - 1 - i;
                    let gradient = residual * self.residuals[lag_index];
                    self.ma_coeffs[i] = self.ma_coeffs[i] + self.learning_rate * gradient;

                    // Keep coefficients stable
                    self.ma_coeffs[i] = self.ma_coeffs[i]
                        .max(F::from(-0.99).unwrap())
                        .min(F::from(0.99).unwrap());
                }
            }

            Ok(())
        }

        /// Generate forecast
        pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
            if !self.initialized {
                return Ok(Array1::zeros(steps));
            }

            let mut forecasts = Array1::zeros(steps);
            let mut extended_data = self.observations.clone();
            let mut extended_residuals = self.residuals.clone();

            for step in 0..steps {
                // Apply differencing to get stationary series
                let processed_data: Vec<F> = if self.d > 0 {
                    self.apply_differencing_to_extended(&extended_data)
                } else {
                    extended_data.iter().cloned().collect()
                };

                let mut prediction = F::zero();

                // AR component
                for i in 0..self.p {
                    if i < processed_data.len() {
                        let lag_index = processed_data.len() - 1 - i;
                        prediction = prediction + self.ar_coeffs[i] * processed_data[lag_index];
                    }
                }

                // MA component
                for i in 0..self.q {
                    if i < extended_residuals.len() {
                        let lag_index = extended_residuals.len() - 1 - i;
                        prediction = prediction + self.ma_coeffs[i] * extended_residuals[lag_index];
                    }
                }

                // Convert back from differenced space if needed
                let forecast = if self.d > 0 && !extended_data.is_empty() {
                    prediction + extended_data[extended_data.len() - 1]
                } else {
                    prediction
                };

                forecasts[step] = forecast;

                // Extend data for next iteration
                extended_data.push_back(forecast);
                extended_residuals.push_back(F::zero()); // Assume zero residual for future

                // Maintain buffer size
                if extended_data.len() > 100 {
                    extended_data.pop_front();
                }
                if extended_residuals.len() > 100 {
                    extended_residuals.pop_front();
                }
            }

            Ok(forecasts)
        }

        /// Apply differencing to extended data
        fn apply_differencing_to_extended(&self, data: &VecDeque<F>) -> Vec<F> {
            let mut series: Vec<F> = data.iter().cloned().collect();

            for _ in 0..self.d {
                let mut diff_series = Vec::new();
                for i in 1..series.len() {
                    diff_series.push(series[i] - series[i - 1]);
                }
                series = diff_series;
            }

            series
        }

        /// Get current model parameters
        pub fn get_parameters(&self) -> (Array1<F>, Array1<F>) {
            (self.ar_coeffs.clone(), self.ma_coeffs.clone())
        }

        /// Check if model is initialized
        pub fn is_initialized(&self) -> bool {
            self.initialized
        }
    }
}

/// Advanced streaming time series capabilities
pub mod advanced {
    use super::*;
    use ndarray::Array1;
    use std::collections::VecDeque;

    /// Real-time forecasting with online model updates
    #[derive(Debug)]
    pub struct StreamingForecaster<F: Float + Debug> {
        /// Exponential smoothing parameter
        alpha: F,
        /// Trend parameter
        beta: Option<F>,
        /// Seasonal parameter
        gamma: Option<F>,
        /// Seasonal period
        seasonal_period: Option<usize>,
        /// Current level
        level: Option<F>,
        /// Current trend
        trend: Option<F>,
        /// Seasonal components
        seasonal: VecDeque<F>,
        /// Recent observations buffer
        buffer: VecDeque<F>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Number of observations processed
        observation_count: usize,
    }

    impl<F: Float + Debug + Clone> StreamingForecaster<F> {
        /// Create new streaming forecaster
        pub fn new(
            alpha: F,
            beta: Option<F>,
            gamma: Option<F>,
            seasonal_period: Option<usize>,
            max_buffer_size: usize,
        ) -> Result<Self> {
            if alpha <= F::zero() || alpha > F::one() {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "alpha".to_string(),
                    message: "Alpha must be between 0 and 1".to_string(),
                });
            }

            let seasonal = if let Some(_period) = seasonal_period {
                VecDeque::with_capacity(_period)
            } else {
                VecDeque::new()
            };

            Ok(Self {
                alpha,
                beta,
                gamma,
                seasonal_period,
                level: None,
                trend: None,
                seasonal,
                buffer: VecDeque::with_capacity(max_buffer_size),
                max_buffer_size,
                observation_count: 0,
            })
        }

        /// Add new observation and update model
        pub fn update(&mut self, value: F) -> Result<()> {
            self.observation_count += 1;

            // Add to buffer
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(value);

            // Initialize components
            if self.level.is_none() {
                self.level = Some(value);
                if self.beta.is_some() {
                    self.trend = Some(F::zero());
                }
                if let Some(period) = self.seasonal_period {
                    for _ in 0..period {
                        self.seasonal.push_back(F::zero());
                    }
                }
                return Ok(());
            }

            let current_level = self.level.unwrap();
            let mut new_level = value;

            // Handle seasonality
            let _seasonal_component = if let Some(period) = self.seasonal_period {
                if self.seasonal.len() >= period {
                    let seasonal_idx = (self.observation_count - 1) % period;
                    let seasonal_val = self.seasonal[seasonal_idx];
                    new_level = new_level - seasonal_val;
                    seasonal_val
                } else {
                    F::zero()
                }
            } else {
                F::zero()
            };

            // Update level
            self.level = Some(self.alpha * new_level + (F::one() - self.alpha) * current_level);

            // Update trend if enabled
            if let Some(beta) = self.beta {
                if let Some(current_trend) = self.trend {
                    let new_trend = beta * (self.level.unwrap() - current_level)
                        + (F::one() - beta) * current_trend;
                    self.trend = Some(new_trend);
                }
            }

            // Update seasonal component if enabled
            if let (Some(gamma), Some(period)) = (self.gamma, self.seasonal_period) {
                if self.seasonal.len() >= period {
                    let seasonal_idx = (self.observation_count - 1) % period;
                    let current_seasonal = self.seasonal[seasonal_idx];
                    let new_seasonal = gamma * (value - self.level.unwrap())
                        + (F::one() - gamma) * current_seasonal;
                    self.seasonal[seasonal_idx] = new_seasonal;
                }
            }

            Ok(())
        }

        /// Generate forecast for next h steps
        pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
            if self.level.is_none() {
                return Err(TimeSeriesError::InvalidModel(
                    "Model not initialized with any data".to_string(),
                ));
            }

            let mut forecasts = Array1::zeros(steps);
            let level = self.level.unwrap();
            let trend = self.trend.unwrap_or(F::zero());

            for h in 0..steps {
                let h_f = F::from(h + 1).unwrap();
                let mut forecast = level + trend * h_f;

                // Add seasonal component if available
                if let Some(period) = self.seasonal_period {
                    if !self.seasonal.is_empty() {
                        let seasonal_idx = (self.observation_count + h) % period;
                        if seasonal_idx < self.seasonal.len() {
                            forecast = forecast + self.seasonal[seasonal_idx];
                        }
                    }
                }

                forecasts[h] = forecast;
            }

            Ok(forecasts)
        }

        /// Get current model state summary
        pub fn get_state(&self) -> ModelState<F> {
            ModelState {
                level: self.level,
                trend: self.trend,
                seasonal_components: self.seasonal.iter().cloned().collect(),
                observation_count: self.observation_count,
                buffer_size: self.buffer.len(),
            }
        }
    }

    /// Model state summary
    #[derive(Debug, Clone)]
    pub struct ModelState<F: Float> {
        /// Current level component
        pub level: Option<F>,
        /// Current trend component
        pub trend: Option<F>,
        /// Seasonal components vector
        pub seasonal_components: Vec<F>,
        /// Number of observations processed
        pub observation_count: usize,
        /// Size of the internal buffer
        pub buffer_size: usize,
    }

    /// Online anomaly detection using Isolation Forest-like approach
    #[derive(Debug)]
    pub struct StreamingAnomalyDetector<F: Float + Debug> {
        /// Recent feature vectors for comparison
        feature_buffer: VecDeque<Vec<F>>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Anomaly threshold
        threshold: F,
        /// Feature extractors
        window_size: usize,
        /// Number of features to extract
        num_features: usize,
    }

    impl<F: Float + Debug + Clone> StreamingAnomalyDetector<F> {
        /// Create new anomaly detector
        pub fn new(
            max_buffer_size: usize,
            threshold: F,
            window_size: usize,
            num_features: usize,
        ) -> Self {
            Self {
                feature_buffer: VecDeque::with_capacity(max_buffer_size),
                max_buffer_size,
                threshold,
                window_size,
                num_features,
            }
        }

        /// Extract features from a time series window
        fn extract_features(&self, window: &[F]) -> Vec<F> {
            if window.is_empty() {
                return vec![F::zero(); self.num_features];
            }

            let mut features = Vec::with_capacity(self.num_features);
            let n = F::from(window.len()).unwrap();

            // Feature 1: Mean
            let mean = window.iter().fold(F::zero(), |acc, &x| acc + x) / n;
            features.push(mean);

            // Feature 2: Standard deviation
            let variance = window
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / n;
            features.push(variance.sqrt());

            // Feature 3: Skewness (simplified)
            let skewness = window
                .iter()
                .map(|&x| {
                    let normalized = (x - mean) / variance.sqrt();
                    normalized * normalized * normalized
                })
                .fold(F::zero(), |acc, x| acc + x)
                / n;
            features.push(skewness);

            // Feature 4: Range
            let min_val = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            features.push(max_val - min_val);

            // Feature 5: Trend (slope of linear regression)
            if window.len() > 1 {
                let x_mean = F::from(window.len() - 1).unwrap() / F::from(2).unwrap();
                let mut num = F::zero();
                let mut den = F::zero();

                for (i, &y) in window.iter().enumerate() {
                    let x = F::from(i).unwrap();
                    num = num + (x - x_mean) * (y - mean);
                    den = den + (x - x_mean) * (x - x_mean);
                }

                let slope = if den > F::zero() {
                    num / den
                } else {
                    F::zero()
                };
                features.push(slope);
            } else {
                features.push(F::zero());
            }

            features
        }

        /// Update detector with new window and check for anomalies
        pub fn update(&mut self, window: &[F]) -> Result<bool> {
            if window.len() < self.window_size {
                return Ok(false); // Not enough data
            }

            let features = self.extract_features(&window[window.len() - self.window_size..]);

            if self.feature_buffer.is_empty() {
                // First observation - just store
                if self.feature_buffer.len() >= self.max_buffer_size {
                    self.feature_buffer.pop_front();
                }
                self.feature_buffer.push_back(features);
                return Ok(false);
            }

            // Calculate isolation score (simplified)
            let mut min_distance = F::infinity();
            for stored_features in &self.feature_buffer {
                let distance = features
                    .iter()
                    .zip(stored_features.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();
                min_distance = min_distance.min(distance);
            }

            // Add current features to buffer
            if self.feature_buffer.len() >= self.max_buffer_size {
                self.feature_buffer.pop_front();
            }
            self.feature_buffer.push_back(features);

            // Check if anomaly (isolated point)
            Ok(min_distance > self.threshold)
        }

        /// Update threshold based on recent observations
        pub fn adapt_threshold(&mut self, factor: F) {
            if self.feature_buffer.len() > 2 {
                // Calculate average distance between recent features
                let mut total_distance = F::zero();
                let mut count = 0;

                for i in 0..self.feature_buffer.len() {
                    for j in i + 1..self.feature_buffer.len() {
                        let distance = self.feature_buffer[i]
                            .iter()
                            .zip(self.feature_buffer[j].iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .fold(F::zero(), |acc, x| acc + x)
                            .sqrt();
                        total_distance = total_distance + distance;
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_distance = total_distance / F::from(count).unwrap();
                    self.threshold = avg_distance * factor;
                }
            }
        }
    }

    /// Online pattern matching for streaming time series
    #[derive(Debug)]
    pub struct StreamingPatternMatcher<F: Float + Debug> {
        /// Template patterns to match against
        patterns: Vec<Vec<F>>,
        /// Pattern names
        pattern_names: Vec<String>,
        /// Recent data buffer for pattern matching
        buffer: VecDeque<F>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Matching threshold (normalized correlation)
        threshold: F,
    }

    impl<F: Float + Debug + Clone> StreamingPatternMatcher<F> {
        /// Create new pattern matcher
        pub fn new(_max_buffersize: usize, threshold: F) -> Self {
            Self {
                patterns: Vec::new(),
                pattern_names: Vec::new(),
                buffer: VecDeque::with_capacity(_max_buffersize),
                max_buffer_size: _max_buffersize,
                threshold,
            }
        }

        /// Add a pattern to match against
        pub fn add_pattern(&mut self, pattern: Vec<F>, name: String) -> Result<()> {
            if pattern.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Pattern cannot be empty".to_string(),
                ));
            }
            self.patterns.push(pattern);
            self.pattern_names.push(name);
            Ok(())
        }

        /// Update buffer and check for pattern matches
        pub fn update(&mut self, value: F) -> Vec<PatternMatch> {
            // Add to buffer
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(value);

            let mut matches = Vec::new();

            // Check each pattern
            for (i, pattern) in self.patterns.iter().enumerate() {
                if self.buffer.len() >= pattern.len() {
                    let recent_data: Vec<F> = self
                        .buffer
                        .iter()
                        .rev()
                        .take(pattern.len())
                        .rev()
                        .cloned()
                        .collect();

                    if let Ok(correlation) = self.normalized_correlation(&recent_data, pattern) {
                        if correlation >= self.threshold {
                            matches.push(PatternMatch {
                                pattern_name: self.pattern_names[i].clone(),
                                correlation: correlation.to_f64().unwrap(),
                                start_index: self.buffer.len() - pattern.len(),
                                pattern_length: pattern.len(),
                            });
                        }
                    }
                }
            }

            matches
        }

        /// Calculate normalized correlation between two sequences
        fn normalized_correlation(&self, a: &[F], b: &[F]) -> Result<F> {
            if a.len() != b.len() || a.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Sequences must have the same non-zero length".to_string(),
                ));
            }

            let n = F::from(a.len()).unwrap();

            // Calculate means
            let mean_a = a.iter().fold(F::zero(), |acc, &x| acc + x) / n;
            let mean_b = b.iter().fold(F::zero(), |acc, &x| acc + x) / n;

            // Calculate correlation components
            let mut num = F::zero();
            let mut den_a = F::zero();
            let mut den_b = F::zero();

            for (&val_a, &val_b) in a.iter().zip(b.iter()) {
                let diff_a = val_a - mean_a;
                let diff_b = val_b - mean_b;

                num = num + diff_a * diff_b;
                den_a = den_a + diff_a * diff_a;
                den_b = den_b + diff_b * diff_b;
            }

            let denominator = (den_a * den_b).sqrt();
            if denominator > F::zero() {
                Ok(num / denominator)
            } else {
                Ok(F::zero())
            }
        }
    }

    /// Pattern match result
    #[derive(Debug, Clone)]
    pub struct PatternMatch {
        /// Name of the matched pattern
        pub pattern_name: String,
        /// Correlation coefficient with the pattern
        pub correlation: f64,
        /// Starting index in the time series
        pub start_index: usize,
        /// Length of the matched pattern
        pub pattern_length: usize,
    }

    /// Memory-efficient circular buffer for streaming data
    #[derive(Debug)]
    pub struct CircularBuffer<F: Float> {
        /// Internal buffer
        buffer: Vec<F>,
        /// Current write position
        position: usize,
        /// Maximum capacity
        capacity: usize,
        /// Whether buffer is full
        is_full: bool,
    }

    impl<F: Float + Debug + Clone + Default> CircularBuffer<F> {
        /// Create new circular buffer
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: vec![F::default(); capacity],
                position: 0,
                capacity,
                is_full: false,
            }
        }

        /// Add new value to buffer
        pub fn push(&mut self, value: F) {
            self.buffer[self.position] = value;
            self.position = (self.position + 1) % self.capacity;

            if self.position == 0 {
                self.is_full = true;
            }
        }

        /// Get current size of buffer
        pub fn len(&self) -> usize {
            if self.is_full {
                self.capacity
            } else {
                self.position
            }
        }

        /// Check if buffer is empty
        pub fn is_empty(&self) -> bool {
            !self.is_full && self.position == 0
        }

        /// Get slice of recent n values
        pub fn recent(&self, n: usize) -> Vec<F> {
            let available = self.len();
            let take = n.min(available);
            let mut result = Vec::with_capacity(take);

            if self.is_full {
                // Buffer is full, need to handle wrap-around
                let start_pos = (self.position + self.capacity - take) % self.capacity;

                if start_pos + take <= self.capacity {
                    // No wrap-around needed
                    result.extend_from_slice(&self.buffer[start_pos..start_pos + take]);
                } else {
                    // Need to handle wrap-around
                    let first_part = self.capacity - start_pos;
                    result.extend_from_slice(&self.buffer[start_pos..]);
                    result.extend_from_slice(&self.buffer[..take - first_part]);
                }
            } else {
                // Buffer not full, simple case
                let start = self.position.saturating_sub(take);
                result.extend_from_slice(&self.buffer[start..self.position]);
            }

            result
        }

        /// Get all values in chronological order
        pub fn to_vec(&self) -> Vec<F> {
            self.recent(self.len())
        }

        /// Calculate statistics over recent window
        pub fn window_stats(&self, windowsize: usize) -> OnlineStats<F> {
            let recent_data = self.recent(windowsize);
            let mut stats = OnlineStats::new();

            for value in recent_data {
                stats.update(value);
            }

            stats
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_streaming_forecaster() {
            let mut forecaster = StreamingForecaster::new(0.3, Some(0.1), None, None, 100).unwrap();

            // Add trend data
            for i in 1..=20 {
                forecaster.update(i as f64).unwrap();
            }

            let forecast = forecaster.forecast(5).unwrap();
            assert_eq!(forecast.len(), 5);

            // Should forecast increasing trend
            assert!(forecast[1] > forecast[0]);
            assert!(forecast[2] > forecast[1]);
        }

        #[test]
        fn test_anomaly_detector() {
            let mut detector = StreamingAnomalyDetector::new(100, 2.0, 10, 5);

            // Add normal data
            let normal_data: Vec<f64> = (0..20).map(|x| x as f64).collect();

            for window in normal_data.windows(10) {
                let is_anomaly = detector.update(window).unwrap();
                assert!(!is_anomaly, "Normal data should not be anomalous");
            }

            // Add anomalous data
            let mut anomalous_data = normal_data.clone();
            anomalous_data.extend(vec![1000.0; 10]); // Clear anomaly

            let result = detector
                .update(&anomalous_data[anomalous_data.len() - 10..])
                .unwrap();
            assert!(result, "Clear anomaly should be detected");
        }

        #[test]
        fn test_pattern_matcher() {
            let mut matcher = StreamingPatternMatcher::new(100, 0.8);

            // Add a simple pattern
            let pattern = vec![1.0, 2.0, 3.0, 2.0, 1.0];
            matcher
                .add_pattern(pattern.clone(), "triangle".to_string())
                .unwrap();

            // Add matching data
            for &value in &pattern {
                let matches = matcher.update(value);
                if !matches.is_empty() {
                    assert_eq!(matches[0].pattern_name, "triangle");
                    assert!(matches[0].correlation >= 0.8);
                }
            }
        }

        #[test]
        fn test_circular_buffer() {
            let mut buffer = CircularBuffer::new(5);

            // Add data
            for i in 1..=3 {
                buffer.push(i as f64);
            }

            assert_eq!(buffer.len(), 3);
            assert_eq!(buffer.recent(2), vec![2.0, 3.0]);

            // Fill buffer completely
            for i in 4..=7 {
                buffer.push(i as f64);
            }

            assert_eq!(buffer.len(), 5);
            assert_eq!(buffer.to_vec(), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        }
    }
}
