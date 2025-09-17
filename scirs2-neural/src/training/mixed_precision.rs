//! Mixed precision training utilities

/// Configuration for mixed precision training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Whether to enable mixed precision training
    pub enabled: bool,
    /// Loss scaling factor
    pub loss_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 1.0,
        }
    }
}
