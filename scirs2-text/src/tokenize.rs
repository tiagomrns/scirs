//! Text tokenization utilities
//!
//! This module provides functionality for tokenizing text into
//! words, sentences, or characters.

use crate::error::{Result, TextError};
use lazy_static::lazy_static;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

lazy_static! {
    static ref WORD_PATTERN: Regex = Regex::new(r"\b\w+\b").unwrap();
    static ref SENTENCE_PATTERN: Regex = Regex::new(r"[^.!?]+[.!?]").unwrap();
}

/// Trait for tokenizing text
pub trait Tokenizer {
    /// Tokenize the input text into tokens
    fn tokenize(&self, text: &str) -> Result<Vec<String>>;

    /// Tokenize batch of text
    fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

/// Tokenizer for splitting text into words
#[derive(Debug, Clone)]
pub struct WordTokenizer {
    lowercase: bool,
    pattern: Option<Regex>,
}

impl WordTokenizer {
    /// Create a new word tokenizer
    pub fn new(lowercase: bool) -> Self {
        Self {
            lowercase,
            pattern: None,
        }
    }

    /// Create a new word tokenizer with a custom pattern
    pub fn with_pattern(lowercase: bool, pattern: &str) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                lowercase,
                pattern: Some(regex),
            }),
            Err(e) => Err(TextError::TokenizationError(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl Default for WordTokenizer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens = match &self.pattern {
            Some(pattern) => pattern
                .find_iter(&text)
                .map(|m| m.as_str().to_string())
                .collect(),
            None => WORD_PATTERN
                .find_iter(&text)
                .map(|m| m.as_str().to_string())
                .collect(),
        };

        Ok(tokens)
    }
}

/// Tokenizer for splitting text into sentences
#[derive(Debug, Clone)]
pub struct SentenceTokenizer {
    pattern: Option<Regex>,
}

impl SentenceTokenizer {
    /// Create a new sentence tokenizer
    pub fn new() -> Self {
        Self { pattern: None }
    }

    /// Create a new sentence tokenizer with a custom pattern
    pub fn with_pattern(pattern: &str) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                pattern: Some(regex),
            }),
            Err(e) => Err(TextError::TokenizationError(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl Default for SentenceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SentenceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = match &self.pattern {
            Some(pattern) => pattern
                .find_iter(text)
                .map(|m| m.as_str().trim().to_string())
                .collect(),
            None => SENTENCE_PATTERN
                .find_iter(text)
                .map(|m| m.as_str().trim().to_string())
                .collect(),
        };

        Ok(tokens)
    }
}

/// Tokenizer for splitting text into characters or grapheme clusters
#[derive(Debug, Clone)]
pub struct CharacterTokenizer {
    use_grapheme_clusters: bool,
}

impl CharacterTokenizer {
    /// Create a new character tokenizer
    pub fn new(use_grapheme_clusters: bool) -> Self {
        Self {
            use_grapheme_clusters,
        }
    }
}

impl Default for CharacterTokenizer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Tokenizer for CharacterTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = if self.use_grapheme_clusters {
            text.graphemes(true).map(|g| g.to_string()).collect()
        } else {
            text.chars().map(|c| c.to_string()).collect()
        };

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenizer() {
        let tokenizer = WordTokenizer::default();
        let text = "Hello, world! This is a test.";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_word_tokenizer_custom_pattern() {
        let tokenizer = WordTokenizer::with_pattern(false, r"\w+").unwrap();
        let text = "Hello, world! This is a test.";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["Hello", "world", "This", "is", "a", "test"]);
    }

    #[test]
    fn test_sentence_tokenizer() {
        let tokenizer = SentenceTokenizer::default();
        let text = "Hello, world! This is a test. How are you today?";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(
            tokens,
            vec!["Hello, world!", "This is a test.", "How are you today?"]
        );
    }

    #[test]
    fn test_character_tokenizer() {
        let tokenizer = CharacterTokenizer::new(false);
        let text = "Hello";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["H", "e", "l", "l", "o"]);
    }

    #[test]
    fn test_grapheme_tokenizer() {
        let tokenizer = CharacterTokenizer::default();
        let text = "café";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["c", "a", "f", "é"]);
    }
}
