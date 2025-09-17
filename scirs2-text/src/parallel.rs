//! Parallel processing utilities for text
//!
//! This module provides utilities for parallel text processing
//! using multiple threads.

use crate::error::Result;
use crate::tokenize::Tokenizer;
use crate::vectorize::Vectorizer;
use ndarray::Array2;
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

/// Parallel tokenizer
pub struct ParallelTokenizer<T: Tokenizer + Send + Sync> {
    /// The tokenizer to use
    tokenizer: T,
    /// Chunk size for parallel processing
    chunk_size: usize,
}

impl<T: Tokenizer + Send + Sync> ParallelTokenizer<T> {
    /// Create a new parallel tokenizer
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            chunk_size: 1000,
        }
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, chunksize: usize) -> Self {
        self.chunk_size = chunksize;
        self
    }

    /// Tokenize texts in parallel
    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        let results: Result<Vec<_>> = texts
            .par_chunks(self.chunk_size)
            .flat_map(|chunk| {
                let mut chunk_results = Vec::new();
                for &text in chunk {
                    match self.tokenizer.tokenize(text) {
                        Ok(tokens) => chunk_results.push(tokens),
                        Err(e) => return vec![Err(e)],
                    }
                }
                chunk_results.into_iter().map(Ok).collect::<Vec<_>>()
            })
            .collect();

        results
    }

    /// Tokenize texts in parallel and apply a mapper function
    pub fn tokenize_and_map<F, R>(&self, texts: &[&str], mapper: F) -> Result<Vec<R>>
    where
        F: Fn(Vec<String>) -> R + Send + Sync,
        R: Send,
    {
        let results: Result<Vec<_>> = texts
            .par_chunks(self.chunk_size)
            .flat_map(|chunk| {
                let mut chunk_results = Vec::new();
                for &text in chunk {
                    match self.tokenizer.tokenize(text) {
                        Ok(tokens) => chunk_results.push(Ok(mapper(tokens))),
                        Err(e) => return vec![Err(e)],
                    }
                }
                chunk_results
            })
            .collect();

        results
    }
}

/// Parallel vectorizer
pub struct ParallelVectorizer<T: Vectorizer + Send + Sync> {
    /// The vectorizer to use
    vectorizer: Arc<T>,
    /// Chunk size for parallel processing
    chunk_size: usize,
}

impl<T: Vectorizer + Send + Sync> ParallelVectorizer<T> {
    /// Create a new parallel vectorizer
    pub fn new(vectorizer: T) -> Self {
        Self {
            vectorizer: Arc::new(vectorizer),
            chunk_size: 100,
        }
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, chunksize: usize) -> Self {
        self.chunk_size = chunksize;
        self
    }

    /// Transform texts in parallel
    pub fn transform(&self, texts: &[&str]) -> Result<Array2<f64>> {
        // First estimate the dimensions by transforming the first text
        let first_features = self.vectorizer.transform_batch(&texts[0..1])?;
        let n_features = first_features.ncols();

        // Allocate the result matrix
        let n_samples = texts.len();
        let result = Arc::new(Mutex::new(Array2::zeros((n_samples, n_features))));

        // Process in parallel
        let chunk_size = self.chunk_size;
        let errors = Arc::new(Mutex::new(Vec::new()));

        texts
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;

                match self.vectorizer.transform_batch(chunk) {
                    Ok(chunk_vectors) => {
                        let mut result = result.lock().unwrap();

                        for (i, row) in chunk_vectors.rows().into_iter().enumerate() {
                            if start_idx + i < n_samples {
                                result.row_mut(start_idx + i).assign(&row);
                            }
                        }
                    }
                    Err(e) => {
                        let mut errors = errors.lock().unwrap();
                        errors.push(e);
                    }
                }
            });

        let errors = errors.lock().unwrap();
        if !errors.is_empty() {
            return Err(errors[0].clone());
        }

        let result = Arc::try_unwrap(result)
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap result Arc".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap result Mutex".to_string())
            })?;

        Ok(result)
    }
}

/// Parallel text processor that can run multiple operations in parallel
pub struct ParallelTextProcessor {
    /// Number of threads to use
    num_threads: usize,
}

impl Default for ParallelTextProcessor {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }
}

impl ParallelTextProcessor {
    /// Create a new parallel text processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads
    pub fn with_threads(mut self, numthreads: usize) -> Self {
        self.num_threads = numthreads;
        self
    }

    /// Process texts in parallel with a given function
    pub fn process<F, R>(&self, texts: &[&str], f: F) -> Vec<R>
    where
        F: Fn(&str) -> R + Send + Sync,
        R: Send,
    {
        texts.par_iter().map(|&text| f(text)).collect()
    }

    /// Process texts in parallel and flatten the results
    pub fn process_and_flatten<F, R>(&self, texts: &[&str], f: F) -> Vec<R>
    where
        F: Fn(&str) -> Vec<R> + Send + Sync,
        R: Send,
    {
        texts.par_iter().flat_map(|&text| f(text)).collect()
    }

    /// Process texts in parallel with progress tracking
    pub fn process_with_progress<F, R>(
        &self,
        texts: &[&str],
        f: F,
        update_interval: usize,
    ) -> Result<(Vec<R>, Vec<usize>)>
    where
        F: Fn(&str) -> R + Send + Sync,
        R: Send,
    {
        let progress = Arc::new(Mutex::new(Vec::new()));
        let total = texts.len();

        let results: Vec<R> = texts
            .par_iter()
            .enumerate()
            .map(|(i, &text)| {
                let result = f(text);

                // Update progress periodically
                if i % update_interval == 0 || i == total - 1 {
                    let mut progress = progress.lock().unwrap();
                    progress.push(i + 1);
                }

                result
            })
            .collect();

        let progress = Arc::try_unwrap(progress)
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap progress Arc".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap progress Mutex".to_string())
            })?;

        Ok((results, progress))
    }

    /// Batch process texts with custom chunking
    pub fn batch_process<F, R>(&self, texts: &[&str], chunksize: usize, f: F) -> Vec<Vec<R>>
    where
        F: Fn(&[&str]) -> Vec<R> + Send + Sync,
        R: Send,
    {
        texts.par_chunks(chunksize).map(f).collect()
    }
}

/// Parallel corpus processor for handling larger datasets
pub struct ParallelCorpusProcessor {
    /// Number of documents to process at once
    batch_size: usize,
    /// Number of threads to use
    num_threads: Option<usize>,
    /// Maximum memory usage (in bytes)
    max_memory: Option<usize>,
}

impl ParallelCorpusProcessor {
    /// Create a new parallel corpus processor
    pub fn new(_batchsize: usize) -> Self {
        Self {
            batch_size: _batchsize,
            num_threads: None,
            max_memory: None,
        }
    }

    /// Set the number of threads
    pub fn with_threads(mut self, numthreads: usize) -> Self {
        self.num_threads = Some(numthreads);
        self
    }

    /// Set the maximum memory usage
    pub fn with_max_memory(mut self, maxmemory: usize) -> Self {
        self.max_memory = Some(maxmemory);
        self
    }

    /// Process a corpus in parallel with a thread pool
    pub fn process<F, R>(&self, corpus: &[&str], processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[&str]) -> Result<Vec<R>> + Send + Sync,
        R: Send,
    {
        // Process in batches using scirs2-core parallel abstractions
        let results = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));

        // Use scirs2-core parallel processing instead of custom thread pool
        {
            // Collect results with indices to preserve order
            let indexed_results: Vec<_> = corpus
                .par_chunks(self.batch_size)
                .enumerate()
                .map(|(idx, batch)| match processor(batch) {
                    Ok(batch_results) => Ok((idx, batch_results)),
                    Err(e) => Err(e),
                })
                .collect();

            // Check for errors
            for result in &indexed_results {
                if let Err(e) = result {
                    let mut errors = errors.lock().unwrap();
                    errors.push(e.clone());
                    return Err(e.clone());
                }
            }

            // Sort by index and flatten results
            let mut sorted_results: Vec<_> =
                indexed_results.into_iter().filter_map(|r| r.ok()).collect();
            sorted_results.sort_by_key(|(idx_, _)| *idx_);

            let mut results_guard = results.lock().unwrap();
            for (_, batch_results) in sorted_results {
                results_guard.extend(batch_results);
            }
        }

        // Check for errors
        let errors = errors.lock().unwrap();
        if !errors.is_empty() {
            return Err(errors[0].clone());
        }

        // Return results
        let results = Arc::try_unwrap(results)
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap results Arc".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                crate::error::TextError::RuntimeError("Failed to unwrap results Mutex".to_string())
            })?;

        Ok(results)
    }

    /// Process a corpus in parallel with progress tracking
    pub fn process_with_progress<F, R>(
        &self,
        corpus: &[&str],
        processor: F,
        progress_callback: impl Fn(usize, usize) + Send + Sync,
    ) -> Result<Vec<R>>
    where
        F: Fn(&[&str]) -> Result<Vec<R>> + Send + Sync,
        R: Send,
    {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let processed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let total = corpus.len();

        let batches: Vec<_> = corpus.chunks(self.batch_size).collect();

        // Collect results with indices to preserve order
        let indexed_results: Vec<_> = batches
            .into_par_iter()
            .enumerate()
            .map(|(idx, batch)| {
                let result = match processor(batch) {
                    Ok(batch_results) => Ok((idx, batch_results)),
                    Err(e) => Err(e),
                };

                // Update progress
                let current = processed.fetch_add(batch.len(), std::sync::atomic::Ordering::SeqCst)
                    + batch.len();
                progress_callback(current, total);

                result
            })
            .collect();

        // Check for errors
        for result in &indexed_results {
            if let Err(e) = result {
                let mut errors = errors.lock().unwrap();
                errors.push(e.clone());
            }
        }

        // Check for errors
        let errors = errors.lock().unwrap();
        if !errors.is_empty() {
            return Err(errors[0].clone());
        }
        drop(errors);

        // Sort by index and flatten results
        let mut sorted_results: Vec<_> =
            indexed_results.into_iter().filter_map(|r| r.ok()).collect();
        sorted_results.sort_by_key(|(idx_, _)| *idx_);

        let mut final_results = Vec::new();
        for (_, batch_results) in sorted_results {
            final_results.extend(batch_results);
        }

        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WhitespaceTokenizer;
    use crate::vectorize::TfidfVectorizer;

    fn create_testtexts() -> Vec<&'static str> {
        vec![
            "This is a test document",
            "Another test document here",
            "Document with more words for testing",
            "Short text",
            "More documents for parallel processing testing",
        ]
    }

    #[test]
    fn test_parallel_tokenizer() {
        let tokenizer = ParallelTokenizer::new(WhitespaceTokenizer::new());
        let texts = create_testtexts();

        let tokens = tokenizer.tokenize(&texts);

        let tokens = tokens.expect("Tokenization should succeed");
        assert_eq!(tokens.len(), texts.len());
        assert_eq!(tokens[0], vec!["This", "is", "a", "test", "document"]);
    }

    #[test]
    fn test_parallel_tokenizer_with_mapper() {
        let tokenizer = ParallelTokenizer::new(WhitespaceTokenizer::new());
        let texts = create_testtexts();

        let token_counts = tokenizer.tokenize_and_map(&texts, |tokens| tokens.len());

        let token_counts = token_counts.expect("Tokenization and mapping should succeed");
        assert_eq!(token_counts, vec![5, 4, 6, 2, 6]);
    }

    #[test]
    fn test_parallel_vectorizer() {
        let mut vectorizer = TfidfVectorizer::default();
        let texts = create_testtexts();

        vectorizer.fit(&texts).unwrap();
        let parallel_vectorizer = ParallelVectorizer::new(vectorizer);

        let vectors = parallel_vectorizer.transform(&texts).unwrap();

        assert_eq!(vectors.nrows(), texts.len());
        assert!(vectors.ncols() > 0);
    }

    #[test]
    fn test_paralleltext_processor() {
        let processor = ParallelTextProcessor::new();
        let texts = create_testtexts();

        let word_counts = processor.process(&texts, |text| text.split_whitespace().count());

        assert_eq!(word_counts, vec![5, 4, 6, 2, 6]);
    }

    #[test]
    fn test_paralleltext_processor_with_progress() {
        let processor = ParallelTextProcessor::new();
        let texts = create_testtexts();

        let (word_counts, progress) = processor
            .process_with_progress(&texts, |text| text.split_whitespace().count(), 2)
            .unwrap();

        assert_eq!(word_counts, vec![5, 4, 6, 2, 6]);
        assert!(!progress.is_empty());
    }

    #[test]
    fn test_parallel_corpus_processor() {
        let processor = ParallelCorpusProcessor::new(2);
        let texts = create_testtexts();

        let result = processor
            .process(&texts, |batch| {
                let counts = batch
                    .iter()
                    .map(|text| text.split_whitespace().count())
                    .collect();
                Ok(counts)
            })
            .unwrap();

        assert_eq!(result, vec![5, 4, 6, 2, 6]);
    }
}
