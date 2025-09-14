//! Quantization-aware training utilities

/// Configuration for quantization-aware training
#[derive(Debug, Clone)]
pub struct QuantizationAwareConfig {
    /// Whether to enable quantization-aware training
    pub enabled: bool,
    /// Bit width for quantization
    pub bit_width: u8,
}

impl Default for QuantizationAwareConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bit_width: 8,
        }
    }
}
