//! GPT implementation
//!
//! GPT (Generative Pre-trained Transformer) is a transformer-based language model
//! designed for autoregressive language modeling. Unlike BERT which is bidirectional,
//! GPT uses a unidirectional (left-to-right) transformer architecture.
//! Reference: "Improving Language Understanding by Generative Pre-Training", Radford et al. (2018)
//! https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Embedding, EmbeddingConfig, Layer, LayerNorm};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::SeedableRng;
use std::fmt::Debug;
/// Configuration for a GPT model
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size in feed-forward networks
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f64,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}
impl GPTConfig {
    /// Create a GPT-2 Small configuration
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
    /// Create a GPT-2 Medium configuration
    pub fn gpt2_medium() -> Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
    /// Create a GPT-2 Large configuration
    pub fn gpt2_large() -> Self {
            hidden_size: 1280,
            num_hidden_layers: 36,
            num_attention_heads: 20,
            intermediate_size: 5120,
    /// Create a custom GPT configuration
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
    ) -> Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size: hidden_size * 4,
/// GPT embedding combining token and position embeddings
struct GPTEmbeddings<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Token embeddings
    token_embeddings: Embedding<F>,
    /// Position embeddings
    position_embeddings: Embedding<F>,
    /// Dropout
    dropout: Dropout<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> GPTEmbeddings<F> {
    /// Create GPT embeddings
    pub fn new(config: &GPTConfig) -> Result<Self> {
        // Token embeddings
        let token_embedding_config = EmbeddingConfig {
            num_embeddings: config.vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let token_embeddings = Embedding::new(token_embedding_config)?;
        // Position embeddings
        let position_embedding_config = EmbeddingConfig {
            num_embeddings: config.max_position_embeddings,
        let position_embeddings = Embedding::new(position_embedding_config)?;
        // Dropout
        let dropout_prob = config.hidden_dropout_prob;
        let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
        let dropout = Dropout::new(dropout_prob, &mut rng)?;
        Ok(Self {
            token_embeddings,
            position_embeddings,
            dropout,
        })
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for GPTEmbeddings<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input should be of shape [batch_size, seq_len] and contain token IDs
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, seq_len], got {:?}",
                shape
            )));
        let batch_size = shape[0];
        let seq_len = shape[1];
        // Get token embeddings
        let inputs_embeds = self.token_embeddings.forward(input)?;
        // Create position IDs
        let mut position_ids = Array::zeros(IxDyn(&[batch_size, seq_len]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                position_ids[[b, s]] = F::from(s).unwrap();
            }
        // Get position embeddings
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        // Combine embeddings
        let mut embeddings = inputs_embeds.clone();
        // Add position embeddings
        for i in 0..embeddings.len() {
            embeddings[i] = embeddings[i] + position_embeddings[i];
        // Apply dropout
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    fn backward(
        &mut self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    fn update(&mut self, learningrate: F) -> Result<()> {
        self.token_embeddings.update(learning_rate)?;
        self.position_embeddings.update(learning_rate)?;
        Ok(())
    fn as_any(&self) -> &dyn std::any::Any {
        self
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
/// GPT attention layer (masked multi-head attention)
struct GPTAttention<F: Float + Debug + ScalarOperand + Send + Sync> {
    num_attention_heads: usize,
    /// Size of each attention head
    attention_head_size: usize,
    /// Query projection
    query: Dense<F>,
    /// Key projection
    key: Dense<F>,
    /// Value projection
    value: Dense<F>,
    /// Output projection
    output: Dense<F>,
    /// Attention dropout
    attn_dropout: Dropout<F>,
    /// Output dropout
    resid_dropout: Dropout<F>,
    /// Scale factor for attention scores
    scale: F,
impl<F: Float + Debug + ScalarOperand + Send + Sync> GPTAttention<F> {
    /// Create GPT attention layer
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / num_attention_heads;
        // Projections
        let query = Dense::new(hidden_size, hidden_size, None, &mut rng)?;
        let key = Dense::new(hidden_size, hidden_size, None, &mut rng)?;
        let value = Dense::new(hidden_size, hidden_size, None, &mut rng)?;
        let output = Dense::new(hidden_size, hidden_size, None, &mut rng)?;
        // Dropouts
        let attn_dropout_prob = config.attention_probs_dropout_prob;
        let resid_dropout_prob = config.hidden_dropout_prob;
        let attn_dropout = Dropout::new(attn_dropout_prob, &mut rng)?;
        let resid_dropout = Dropout::new(resid_dropout_prob, &mut rng)?;
        // Scale factor
        let scale = F::from(1.0 / (attention_head_size as f64).sqrt()).unwrap();
            attention_head_size,
            query,
            key,
            value,
            output,
            attn_dropout,
            resid_dropout,
            scale,
    /// Transpose for attention computation
    fn transpose_for_scores(
        x: &Array<F, IxDyn>,
        batch_size: usize,
        seq_len: usize,
        // Reshape from [batch_size, seq_len, hidden_size] to
        // [batch_size, seq_len, num_heads, head_size]
        let mut x_reshaped = Array::zeros(IxDyn(&[
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ]));
                for h in 0..self.num_attention_heads {
                    for d in 0..self.attention_head_size {
                        let hidden_idx = h * self.attention_head_size + d;
                        x_reshaped[[b, s, h, d]] = x[[b, s, hidden_idx]];
                    }
                }
        // Transpose to [batch_size, num_heads, seq_len, head_size]
        let mut x_transposed = Array::zeros(IxDyn(&[
            for h in 0..self.num_attention_heads {
                for s in 0..seq_len {
                        x_transposed[[b, h, s, d]] = x_reshaped[[b, s, h, d]];
        Ok(x_transposed)
    /// Create attention mask for autoregressive (left-to-right) attention
    fn create_causal_mask(&self, seqlen: usize) -> Array<F, IxDyn> {
        let mut mask = Array::zeros(IxDyn(&[seq_len, seq_len]));
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    mask[[i, j]] = F::one();
                } else {
                    mask[[i, j]] = F::from(-10000.0).unwrap();
        mask
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for GPTAttention<F> {
        // Input should be of shape [batch_size, seq_len, hidden_size]
        if shape.len() != 3 {
                "Expected input shape [batch_size, seq_len, hidden_size], got {:?}",
        let hidden_size = shape[2];
        // Project query, key, value
        let query = self.query.forward(input)?;
        let key = self.key.forward(input)?;
        let value = self.value.forward(input)?;
        // Reshape and transpose for attention computation
        let query_layer = self.transpose_for_scores(&query, batch_size, seq_len)?;
        let key_layer = self.transpose_for_scores(&key, batch_size, seq_len)?;
        let value_layer = self.transpose_for_scores(&value, batch_size, seq_len)?;
        // Compute attention scores
        // [batch_size, num_heads, seq_len, seq_len]
        let mut attention_scores = Array::zeros(IxDyn(&[
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score = F::zero();
                        for k in 0..self.attention_head_size {
                            score = score + query_layer[[b, h, i, k]] * key_layer[[b, h, j, k]];
                        }
                        attention_scores[[b, h, i, j]] = score * self.scale;
        // Apply causal mask
        let causal_mask = self.create_causal_mask(seq_len);
                        attention_scores[[b, h, i, j]] =
                            attention_scores[[b, h, i, j]] + causal_mask[[i, j]];
        // Apply softmax
        let mut attention_probs = Array::zeros(attention_scores.dim());
                    // Find max for numerical stability
                    let mut max_val = F::from(f64::NEG_INFINITY).unwrap();
                        max_val = max_val.max(attention_scores[[b, h, i, j]]);
                    // Compute softmax
                    let mut sum_exp = F::zero();
                        sum_exp = sum_exp + (attention_scores[[b, h, i, j]] - max_val).exp();
                        attention_probs[[b, h, i, j]] =
                            (attention_scores[[b, h, i, j]] - max_val).exp() / sum_exp;
        attention_probs = self.attn_dropout.forward(&attention_probs)?;
        // Compute context layer
        // [batch_size, num_heads, seq_len, head_size]
        let mut context_layer = Array::zeros(IxDyn(&[
                    for k in 0..self.attention_head_size {
                        let mut weighted_sum = F::zero();
                        for j in 0..seq_len {
                            weighted_sum = weighted_sum
                                + attention_probs[[b, h, i, j]] * value_layer[[b, h, j, k]];
                        context_layer[[b, h, i, k]] = weighted_sum;
        // Transpose back to [batch_size, seq_len, hidden_size]
        let mut context_layer_transposed = Array::zeros(IxDyn(&[batch_size, seq_len, hidden_size]));
            for i in 0..seq_len {
                        let hidden_idx = h * self.attention_head_size + k;
                        context_layer_transposed[[b, i, hidden_idx]] = context_layer[[b, h, i, k]];
        // Apply output projection
        let output = self.output.forward(&context_layer_transposed)?;
        let output = self.resid_dropout.forward(&output)?;
        Ok(output)
        self.query.update(learning_rate)?;
        self.key.update(learning_rate)?;
        self.value.update(learning_rate)?;
        self.output.update(learning_rate)?;
/// GPT MLP (feed-forward network)
struct GptMlp<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// First dense layer
    fc1: Dense<F>,
    /// Second dense layer
    fc2: Dense<F>,
    /// Activation function
    activation_fn: Box<dyn Fn(F) -> F + Send + Sync>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> GptMlp<F> {
    /// Create GPT MLP
        // Dense layers
        let fc1 = Dense::new(config.hidden_size, config.intermediate_size, None, &mut rng)?;
        let fc2 = Dense::new(config.intermediate_size, config.hidden_size, None, &mut rng)?;
        // Activation function
        let activation_fn: Box<dyn Fn(F) -> F + Send + Sync> = match config.hidden_act.as_str() {
            "gelu" => Box::new(|x: F| {
                // Approximation of GELU
                let x3 = x * x * x;
                x * F::from(0.5).unwrap()
                    * (F::one() + (x + F::from(0.044715).unwrap() * x3).tanh())
            }),
            "relu" => Box::new(|x: F| x.max(F::zero()), _ => {
                return Err(NeuralError::InferenceError(format!(
                    "Unsupported activation function: {}",
                    config.hidden_act
                )))
            fc1,
            fc2,
            activation_fn,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for GptMlp<F> {
        // Apply first dense layer
        let hidden_states = self.fc1.forward(input)?;
        // Apply activation function
        let hidden_states = hidden_states.mapv(|x| (self.activation_fn)(x));
        // Apply second dense layer
        let hidden_states = self.fc2.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
        self.fc1.update(learning_rate)?;
        self.fc2.update(learning_rate)?;
/// GPT block (attention + MLP)
struct GPTBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Layer normalization for attention
    ln_1: LayerNorm<F>,
    /// Attention layer
    attn: GPTAttention<F>,
    /// Layer normalization for MLP
    ln_2: LayerNorm<F>,
    /// MLP
    mlp: GptMlp<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> GPTBlock<F> {
    /// Create GPT block
        // Layer normalizations
        let layer_norm_eps = config.layer_norm_eps;
        let ln_1 = LayerNorm::new(config.hidden_size, layer_norm_eps, &mut rng)?;
        let ln_2 = LayerNorm::new(config.hidden_size, layer_norm_eps, &mut rng)?;
        // Attention and MLP
        let attn = GPTAttention::new(config)?;
        let mlp = GptMlp::new(config)?;
            ln_1,
            attn,
            ln_2,
            mlp,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for GPTBlock<F> {
        // Attention with residual connection
        let ln1_output = self.ln_1.forward(input)?;
        let attn_output = self.attn.forward(&ln1_output)?;
        // Add residual connection
        let mut residual1 = input.clone();
        for i in 0..residual1.len() {
            residual1[i] = residual1[i] + attn_output[i];
        // MLP with residual connection
        let ln2_output = self.ln_2.forward(&residual1)?;
        let mlp_output = self.mlp.forward(&ln2_output)?;
        let mut residual2 = residual1.clone();
        for i in 0..residual2.len() {
            residual2[i] = residual2[i] + mlp_output[i];
        Ok(residual2)
        self.ln_1.update(learning_rate)?;
        self.attn.update(learning_rate)?;
        self.ln_2.update(learning_rate)?;
        self.mlp.update(learning_rate)?;
/// GPT model implementation
pub struct GPTModel<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Embeddings layer
    embeddings: GPTEmbeddings<F>,
    /// Transformer blocks
    blocks: Vec<GPTBlock<F>>,
    /// Final layer normalization
    ln_f: LayerNorm<F>,
    /// Model configuration
    config: GPTConfig,
impl<F: Float + Debug + ScalarOperand + Send + Sync> GPTModel<F> {
    /// Create a new GPT model
    pub fn new(config: GPTConfig) -> Result<Self> {
        let embeddings = GPTEmbeddings::new(&config)?;
        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(GPTBlock::new(&config)?);
        // Final layer normalization
        let ln_f = LayerNorm::new(config.hidden_size, layer_norm_eps, &mut rng)?;
            embeddings,
            blocks,
            ln_f,
            config,
    /// Create a GPT-2 Small model
    pub fn gpt2_small() -> Result<Self> {
        let config = GPTConfig::gpt2_small();
        Self::new(config)
    /// Create a GPT-2 Medium model
    pub fn gpt2_medium() -> Result<Self> {
        let config = GPTConfig::gpt2_medium();
    /// Create a GPT-2 Large model
    pub fn gpt2_large() -> Result<Self> {
        let config = GPTConfig::gpt2_large();
    /// Create a custom GPT model
    ) -> Result<Self> {
        let config = GPTConfig::custom(
        );
    /// Calculate logits (prediction scores) for next tokens
    pub fn logits(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let hidden_states = self.forward(input)?;
        // Create linear layer for token prediction (weight tied with token embeddings)
        let linear = Dense::new(
            self.config.hidden_size,
            self._config.vocab_size,
            None,
            &mut rng,
        )?;
        // Use the same weights as token embeddings (weight tying)
        // In a real implementation, we'd set the linear weights to be the same as token embeddings.weight
        // Compute logits
        let logits = linear.forward(&hidden_states)?;
        Ok(logits)
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for GPTModel<F> {
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input)?;
        // Apply transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        // Apply final layer normalization
        hidden_states = self.ln_f.forward(&hidden_states)?;
        self.embeddings.update(learning_rate)?;
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        self.ln_f.update(learning_rate)?;
