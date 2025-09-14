//! Configuration validation utilities
//!
//! This module provides functions for validating model configurations
//! against schema and parameter constraints.

use super::ModelConfig;
use crate::error::{Error, Result};
use crate::models::architectures::*;
/// Validate a model configuration
#[allow(dead_code)]
pub fn validate_model_config(config: &ModelConfig) -> Result<()> {
    match _config {
        ModelConfig::ResNet(_config) => validate_resnet_config(_config),
        ModelConfig::ViT(_config) => validate_vit_config(_config),
        ModelConfig::Bert(_config) => validate_bert_config(_config),
        ModelConfig::GPT(_config) => validate_gpt_config(_config),
        ModelConfig::EfficientNet(_config) => validate_efficientnet_config(_config),
        ModelConfig::MobileNet(_config) => validate_mobilenet_config(_config),
        ModelConfig::ConvNeXt(_config) => validate_convnext_config(_config),
        ModelConfig::CLIP(_config) => validate_clip_config(_config),
        ModelConfig::FeatureFusion(_config) => validate_feature_fusion_config(_config),
        ModelConfig::Seq2Seq(_config) => validate_seq2seq_config(_config),
    }
}
/// Validate a ResNet configuration
#[allow(dead_code)]
fn validate_resnet_config(config: &ResNetConfig) -> Result<()> {
    // Validate number of layers
    if !vec![18, 34, 50, 101, 152].contains(&_config.num_layers) {
        return Err(Error::ValidationError(format!(
            "Invalid ResNet depth: {}. Expected one of: 18, 34, 50, 101, 152",
            config.num_layers
        )));
    // Validate input channels
    if config.in_channels == 0 {
        return Err(Error::ValidationError(
            "Invalid input channels: must be greater than 0".to_string(),
        ));
    // Validate number of classes
    if config.num_classes == 0 {
            "Invalid number of classes: must be greater than 0".to_string(),
    Ok(())
/// Validate a Vision Transformer configuration
#[allow(dead_code)]
fn validate_vit_config(config: &ViTConfig) -> Result<()> {
    // Validate image size
    if config.image_size % config.patch_size != 0 {
            "Image size ({}) must be divisible by patch size ({})",
            config.image_size, config.patch_size
    // Validate patch size
    if config.patch_size == 0 {
            "Invalid patch size: must be greater than 0".to_string(),
    // Validate hidden size
    if config.hidden_size % config.num_heads != 0 {
            "Hidden size ({}) must be divisible by number of heads ({})",
            config.hidden_size, config.num_heads
    // Validate dropout rates
    if config.dropout_rate < 0.0 || config.dropout_rate > 1.0 {
            "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
            config.dropout_rate
    if config.attention_dropout_rate < 0.0 || config.attention_dropout_rate > 1.0 {
            "Invalid attention dropout rate: {}. Must be between 0.0 and 1.0",
            config.attention_dropout_rate
    // Validate classifier type
    if config.classifier != "token" && config.classifier != "gap" {
            "Invalid classifier type: {}. Expected 'token' or 'gap'",
            config.classifier
/// Validate a BERT configuration
#[allow(dead_code)]
fn validate_bert_config(config: &BertConfig) -> Result<()> {
    // Validate vocab size
    if config.vocab_size == 0 {
            "Invalid vocabulary size: must be greater than 0".to_string(),
    if config.hidden_size % config.num_attention_heads != 0 {
            "Hidden size ({}) must be divisible by number of attention heads ({})",
            config.hidden_size, config.num_attention_heads
    // Validate intermediate size
    if config.intermediate_size < config.hidden_size {
            "Intermediate size ({}) should not be smaller than hidden size ({})",
            config.intermediate_size, config.hidden_size
    // Validate max position embeddings
    if config.max_position_embeddings < 64 {
            "Max position embeddings ({}) is too small, should be at least 64",
            config.max_position_embeddings
    // Validate dropout
    if config.hidden_dropout_prob < 0.0 || config.hidden_dropout_prob > 1.0 {
            "Invalid hidden dropout probability: {}. Must be between 0.0 and 1.0",
            config.hidden_dropout_prob
    if config.attention_probs_dropout_prob < 0.0 || config.attention_probs_dropout_prob > 1.0 {
            "Invalid attention dropout probability: {}. Must be between 0.0 and 1.0",
            config.attention_probs_dropout_prob
/// Validate a GPT configuration
#[allow(dead_code)]
fn validate_gpt_config(config: &GPTConfig) -> Result<()> {
    if config.n_embd % config.n_head != 0 {
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            config.n_embd, config.n_head
    // Validate context length
    if config.n_ctx < 64 {
            "Context length ({}) is too small, should be at least 64",
            config.n_ctx
    if config.embd_pdrop < 0.0 || config.embd_pdrop > 1.0 {
            "Invalid embedding dropout probability: {}. Must be between 0.0 and 1.0",
            config.embd_pdrop
    if config.attn_pdrop < 0.0 || config.attn_pdrop > 1.0 {
            config.attn_pdrop
    if config.resid_pdrop < 0.0 || config.resid_pdrop > 1.0 {
            "Invalid residual dropout probability: {}. Must be between 0.0 and 1.0",
            config.resid_pdrop
/// Validate an EfficientNet configuration
#[allow(dead_code)]
fn validate_efficientnet_config(config: &EfficientNetConfig) -> Result<()> {
    // Validate width multiplier
    if config.width_multiplier <= 0.0 {
            "Invalid width multiplier: {}. Must be greater than 0.0",
            config.width_multiplier
    // Validate depth multiplier
    if config.depth_multiplier <= 0.0 {
            "Invalid depth multiplier: {}. Must be greater than 0.0",
            config.depth_multiplier
    if config.input_channels == 0 {
    if config.num_classes == 0 && config.include_top {
            "Invalid number of classes: must be greater than 0 when include_top is true"
                .to_string(),
    // Validate dropout rate
    if let Some(rate) = config.dropout_rate {
        if rate < 0.0 || rate > 1.0 {
            return Err(Error::ValidationError(format!(
                "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
                rate
            )));
        }
/// Validate a MobileNet configuration
#[allow(dead_code)]
fn validate_mobilenet_config(config: &MobileNetConfig) -> Result<()> {
/// Validate a ConvNeXt configuration
#[allow(dead_code)]
fn validate_convnext_config(config: &ConvNeXtConfig) -> Result<()> {
    // Validate depths
    if config.depths.is_empty() {
            "Invalid depths: must have at least one stage".to_string(),
    // Validate dims
    if config.dims.is_empty() {
            "Invalid dims: must have at least one stage".to_string(),
    // Validate depths and dims have same length
    if config.depths.len() != config.dims.len() {
            "Depths array length ({}) must match dims array length ({})",
            config.depths.len(),
            config.dims.len()
    // Validate layer scale initialization value
    if config.layer_scale_init_value < 0.0 {
            "Invalid layer scale initialization value: {}. Must be non-negative",
            config.layer_scale_init_value
/// Validate a CLIP configuration
#[allow(dead_code)]
fn validate_clip_config(config: &CLIPConfig) -> Result<()> {
    // Validate vision _config
    validate_vit_config(&_config.vision_config)?;
    // Validate text _config
    if config.text_config.vocab_size == 0 {
    if config.text_config.hidden_size % config.text_config.num_heads != 0 {
            "Text hidden size ({}) must be divisible by number of heads ({})",
            config.text_config.hidden_size, config.text_config.num_heads
    if config.text_config.dropout_rate < 0.0 || config.text_config.dropout_rate > 1.0 {
            "Invalid text dropout rate: {}. Must be between 0.0 and 1.0",
            config.text_config.dropout_rate
    // Validate projection dimension
    if config.projection_dim == 0 {
            "Invalid projection dimension: must be greater than 0".to_string(),
    if config.include_head && config.num_classes == 0 {
            "Invalid number of classes: must be greater than 0 when include_head is true"
/// Validate a Feature Fusion configuration
#[allow(dead_code)]
fn validate_feature_fusion_config(config: &FeatureFusionConfig) -> Result<()> {
    // Validate input dimensions
    if config.input_dims.is_empty() {
            "Invalid input dimensions: must have at least one modality".to_string(),
    for (i, &dim) in config.input_dims.iter().enumerate() {
        if dim == 0 {
                "Invalid input dimension for modality {}: must be greater than 0",
                i
    // Validate hidden dimension
    if config.hidden_dim == 0 {
            "Invalid hidden dimension: must be greater than 0".to_string(),
    // Validate fusion method specific constraints
    match config.fusion_method {
        FusionMethod::Attention | FusionMethod::FiLM | FusionMethod::Bilinear => {
            if config.input_dims.len() != 2 {
                return Err(Error::ValidationError(format!(
                    "{:?} fusion requires exactly 2 modalities, got {}",
                    config.fusion_method,
                    config.input_dims.len()
                )));
            }
        _ => {}
/// Validate a Seq2Seq configuration
#[allow(dead_code)]
fn validate_seq2seq_config(config: &Seq2SeqConfig) -> Result<()> {
    // Validate vocabulary sizes
    if config.input_vocab_size == 0 {
            "Invalid input vocabulary size: must be greater than 0".to_string(),
    if config.output_vocab_size == 0 {
            "Invalid output vocabulary size: must be greater than 0".to_string(),
    // Validate embedding dimension
    if config.embedding_dim == 0 {
            "Invalid embedding dimension: must be greater than 0".to_string(),
    if config.num_layers == 0 {
            "Invalid number of layers: must be greater than 0".to_string(),
    // Validate maximum sequence length
    if config.max_seq_len == 0 {
            "Invalid maximum sequence length: must be greater than 0".to_string(),
