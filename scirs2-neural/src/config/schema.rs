//! JSON Schema definitions for model configurations
//!
//! This module provides JSON Schema definitions for validating model configurations.
//! These schemas can be used for validation and for generating documentation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// JSON Schema for model configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema title
    pub title: String,
    /// Schema description
    pub description: Option<String>,
    /// Schema type
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Schema properties
    pub properties: HashMap<String, JsonSchemaProperty>,
    /// Required properties
    pub required: Vec<String>,
    /// Additional properties allowed
    pub additional_properties: Option<bool>,
}
/// JSON Schema property
pub struct JsonSchemaProperty {
    /// Property type
    pub property_type: Option<String>,
    /// Property description
    /// Property enum values
    #[serde(rename = "enum")]
    pub enum_values: Option<Vec<String>>,
    /// Property format
    pub format: Option<String>,
    /// Minimum value
    pub minimum: Option<f64>,
    /// Maximum value
    pub maximum: Option<f64>,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Items schema (for arrays)
    pub items: Option<Box<JsonSchemaProperty>>,
    /// Reference to another schema
    #[serde(rename = "$ref")]
    pub reference: Option<String>,
    /// Properties for objects
    pub properties: Option<HashMap<String, JsonSchemaProperty>>,
    /// Required properties for objects
    pub required: Option<Vec<String>>,
    /// One of schemas
    pub one_of: Option<Vec<JsonSchemaProperty>>,
    /// All of schemas
    pub all_of: Option<Vec<JsonSchemaProperty>>,
/// Schema registry for all model configurations
pub struct SchemaRegistry;
impl SchemaRegistry {
    /// Get schema for ResNet configuration
    pub fn resnet_schema() -> JsonSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "num_layers".to_string(),
            JsonSchemaProperty {
                property_type: Some("integer".to_string()),
                description: Some("Number of layers in the ResNet model".to_string()),
                enum_values: Some(vec![
                    "18".to_string(),
                    "34".to_string(),
                    "50".to_string(),
                    "101".to_string(),
                    "152".to_string(),
                ]),
                format: None,
                minimum: Some(18.0),
                maximum: Some(152.0),
                default: Some(serde_json::json!(50)),
                items: None,
                reference: None,
                properties: None,
                required: None,
                one_of: None,
                all_of: None,
            },
        );
            "in_channels".to_string(),
                description: Some("Number of input channels".to_string()),
                enum_values: None,
                minimum: Some(1.0),
                maximum: None,
                default: Some(serde_json::json!(3)),
            "num_classes".to_string(),
                description: Some("Number of output classes".to_string()),
                default: Some(serde_json::json!(1000)),
            "zero_init_residual".to_string(),
                property_type: Some("boolean".to_string()),
                description: Some(
                    "Whether to initialize residual connections with zero".to_string(),
                ),
                minimum: None,
                default: Some(serde_json::json!(false)),
        JsonSchema {
            title: "ResNet Configuration".to_string(),
            description: Some("Configuration for ResNet models".to_string()),
            schema_type: "object".to_string(),
            properties,
            required: vec![
                "num_layers".to_string(),
                "in_channels".to_string(),
                "num_classes".to_string(),
            ],
            additional_properties: Some(false),
        }
    }
    /// Get schema for Vision Transformer configuration
    pub fn vit_schema() -> JsonSchema {
            "image_size".to_string(),
                description: Some("Size of the input image (square)".to_string()),
                minimum: Some(32.0),
                default: Some(serde_json::json!(224)),
            "patch_size".to_string(),
                description: Some("Size of the patches to divide the image into".to_string()),
                default: Some(serde_json::json!(16)),
            "hidden_size".to_string(),
                description: Some("Dimension of transformer hidden layers".to_string()),
                default: Some(serde_json::json!(768)),
                description: Some("Number of transformer layers".to_string()),
                default: Some(serde_json::json!(12)),
            "num_heads".to_string(),
                description: Some("Number of attention heads".to_string()),
            "mlp_dim".to_string(),
                description: Some("Dimension of the MLP layers".to_string()),
                default: Some(serde_json::json!(3072)),
            "dropout_rate".to_string(),
                property_type: Some("number".to_string()),
                description: Some("Dropout rate".to_string()),
                minimum: Some(0.0),
                maximum: Some(1.0),
                default: Some(serde_json::json!(0.1)),
            "attention_dropout_rate".to_string(),
                description: Some("Attention dropout rate".to_string()),
                default: Some(serde_json::json!(0.0)),
            "classifier".to_string(),
                property_type: Some("string".to_string()),
                description: Some("Type of classifier ('token' or 'gap')".to_string()),
                enum_values: Some(vec!["token".to_string(), "gap".to_string()]),
                default: Some(serde_json::json!("token")),
            "include_top".to_string(),
                description: Some("Whether to include the classification head".to_string()),
                default: Some(serde_json::json!(true)),
            title: "Vision Transformer Configuration".to_string(),
            description: Some("Configuration for Vision Transformer models".to_string()),
                "image_size".to_string(),
                "patch_size".to_string(),
                "hidden_size".to_string(),
                "num_heads".to_string(),
    /// Get all available schemas
    pub fn get_all_schemas() -> HashMap<String, JsonSchema> {
        let mut schemas = HashMap::new();
        schemas.insert("resnet".to_string(), Self::resnet_schema());
        schemas.insert("vit".to_string(), Self::vit_schema());
        // Add more schemas as needed
        schemas
