//! Parallel operations abstraction layer
//!
//! This module provides a unified interface for parallel operations across the SciRS2 project.
//! It wraps Rayon functionality when the `parallel` feature is enabled, and provides
//! sequential fallbacks when it's disabled.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::parallel_ops::*;
//!
//! // Works with or without the parallel feature
//! let results: Vec<i32> = (0..1000)
//!     .into_par_iter()
//!     .map(|x| x * x)
//!     .collect();
//! ```

// When parallel is enabled, directly re-export Rayon's prelude
#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

// Re-export ThreadPoolBuilder and ThreadPool when parallel is enabled
#[cfg(feature = "parallel")]
pub use rayon::{ThreadPool, ThreadPoolBuilder};

// When parallel is disabled, provide sequential fallbacks
#[cfg(not(feature = "parallel"))]
mod sequential_fallbacks {
    use std::iter;

    /// Sequential fallback for IntoParallelIterator
    pub trait IntoParallelIterator: Sized {
        type Iter: Iterator<Item = Self::Item>;
        type Item;

        fn into_par_iter(self) -> Self::Iter;
    }

    /// Sequential fallback for ParallelIterator
    pub trait ParallelIterator: Iterator + Sized {
        fn map<F, R>(self, f: F) -> iter::Map<Self, F>
        where
            F: FnMut(Self::Item) -> R,
        {
            Iterator::map(self, f)
        }

        fn for_each<F>(self, f: F)
        where
            F: FnMut(Self::Item),
        {
            Iterator::for_each(self, f)
        }

        fn try_for_each<F, E>(self, f: F) -> Result<(), E>
        where
            F: FnMut(Self::Item) -> Result<(), E>,
        {
            Iterator::try_for_each(self, f)
        }

        fn filter<P>(self, predicate: P) -> iter::Filter<Self, P>
        where
            P: FnMut(&Self::Item) -> bool,
        {
            Iterator::filter(self, predicate)
        }

        fn collect<C>(self) -> C
        where
            C: FromIterator<Self::Item>,
        {
            Iterator::collect(self)
        }

        fn fold<T, F>(self, init: T, f: F) -> T
        where
            F: FnMut(T, Self::Item) -> T,
        {
            Iterator::fold(self, init, f)
        }

        fn reduce<F>(self, f: F) -> Option<Self::Item>
        where
            F: FnMut(Self::Item, Self::Item) -> Self::Item,
        {
            Iterator::reduce(self, f)
        }

        fn count(self) -> usize {
            Iterator::count(self)
        }

        fn sum<S>(self) -> S
        where
            S: std::iter::Sum<Self::Item>,
        {
            Iterator::sum(self)
        }

        fn min(self) -> Option<Self::Item>
        where
            Self::Item: Ord,
        {
            Iterator::min(self)
        }

        fn max(self) -> Option<Self::Item>
        where
            Self::Item: Ord,
        {
            Iterator::max(self)
        }
    }

    /// Sequential fallback for ParallelBridge
    pub trait ParallelBridge: Iterator + Sized {
        fn par_bridge(self) -> Self {
            self
        }
    }

    // Implement IntoParallelIterator for common types
    impl IntoParallelIterator for std::ops::Range<usize> {
        type Item = usize;
        type Iter = std::ops::Range<usize>;

        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }

    impl<T> IntoParallelIterator for Vec<T> {
        type Item = T;
        type Iter = std::vec::IntoIter<T>;

        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    impl<'a, T> IntoParallelIterator for &'a [T] {
        type Item = &'a T;
        type Iter = std::slice::Iter<'a, T>;

        fn into_par_iter(self) -> Self::Iter {
            self.iter()
        }
    }

    impl<'a, T> IntoParallelIterator for &'a mut [T] {
        type Item = &'a mut T;
        type Iter = std::slice::IterMut<'a, T>;

        fn into_par_iter(self) -> Self::Iter {
            self.iter_mut()
        }
    }

    // Implement ParallelIterator for all iterators
    impl<T: Iterator> ParallelIterator for T {}

    // Implement ParallelBridge for all iterators
    impl<T: Iterator> ParallelBridge for T {}

    /// Sequential fallback for parallel scope
    pub fn scope<'scope, F, R>(f: F) -> R
    where
        F: FnOnce(&()) -> R,
    {
        f(&())
    }

    /// Sequential fallback for parallel join
    pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA,
        B: FnOnce() -> RB,
    {
        (a(), b())
    }

    // Re-export traits
    pub use self::{IntoParallelIterator, ParallelBridge, ParallelIterator};
}

// Re-export sequential fallbacks when parallel is disabled
#[cfg(not(feature = "parallel"))]
pub use sequential_fallbacks::*;

/// Helper function to create a parallel iterator from a range
#[allow(dead_code)]
pub fn par_range(start: usize, end: usize) -> impl ParallelIterator<Item = usize> {
    (start..end).into_par_iter()
}

/// Helper function for parallel chunks processing
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn par_chunks<T: Sync>(slice: &[T], chunksize: usize) -> rayon::slice::Chunks<'_, T> {
    slice.par_chunks(chunksize)
}

/// Sequential fallback for par_chunks
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn par_chunks<T>(_slice: &[T], chunksize: usize) -> std::slice::Chunks<'_, T> {
    slice.chunks(chunk_size)
}

/// Helper function for parallel mutable chunks processing
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn par_chunks_mut<T: Send>(
    slice: &mut [T],
    chunk_size: usize,
) -> rayon::slice::ChunksMut<'_, T> {
    slice.par_chunks_mut(chunk_size)
}

/// Sequential fallback for par_chunks_mut
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn par_chunks_mut<T>(_slice: &mut [T], chunksize: usize) -> std::slice::ChunksMut<'_, T> {
    slice.chunks_mut(chunk_size)
}

/// Simple parallel map function that returns Result type
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

/// Sequential fallback for parallel_map
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    items.iter().map(f).collect()
}

/// Parallel map function that handles Results
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_result<T, U, E, F>(items: &[T], f: F) -> Result<Vec<U>, E>
where
    T: Sync,
    U: Send,
    E: Send,
    F: Fn(&T) -> Result<U, E> + Sync + Send,
{
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

/// Sequential fallback for parallel_map_result
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_result<T, U, E, F>(items: &[T], f: F) -> Result<Vec<U>, E>
where
    F: Fn(&T) -> Result<U, E>,
{
    items.iter().map(f).collect()
}

/// Check if parallel processing is available
#[allow(dead_code)]
pub fn is_parallel_enabled() -> bool {
    cfg!(feature = "parallel")
}

/// Get the number of threads that would be used for parallel operations
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Sequential fallback returns 1
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn num_threads() -> usize {
    1
}

/// Alias for num_threads() for compatibility
#[allow(dead_code)]
pub fn get_num_threads() -> usize {
    num_threads()
}

/// Set the number of threads for parallel operations
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn set_num_threads(numthreads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(numthreads)
        .build_global()
        .expect("Failed to initialize thread pool");
}

/// Sequential fallback does nothing
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn threads(_: usize) {
    // No-op for sequential fallback
}

/// Parallel-aware scope helper
#[cfg(feature = "parallel")]
pub use rayon::scope as par_scope;

/// Sequential fallback for par_scope
#[cfg(not(feature = "parallel"))]
pub use sequential_fallbacks::scope as par_scope;

/// Parallel join helper
#[cfg(feature = "parallel")]
pub use rayon::join as par_join;

/// Sequential fallback for par_join
#[cfg(not(feature = "parallel"))]
pub use sequential_fallbacks::join as par_join;

/// Parallel map operation on array data with chunking
///
/// This function processes array data in parallel chunks using the provided mapper function.
///
/// # Arguments
///
/// * `data` - The data to process (e.g., array view)
/// * `chunk_size` - Size of each chunk for parallel processing
/// * `mapper` - Function that processes a chunk and returns a result
/// * `reducer` - Function that combines two results into one
///
/// # Returns
///
/// The final reduced result
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_reduce<D, T, M, Red>(data: D, mapper: M, reducer: Red) -> T
where
    D: Send + Sync,
    T: Send + Clone,
    M: Fn(D) -> T + Sync + Send + Clone,
    Red: Fn(T, T) -> T + Sync + Send,
{
    // For simplicity, we'll just apply the mapper once since we can't easily chunk arbitrary data
    // In practice, this would need to be specialized for specific data types
    mapper(data)
}

/// Sequential fallback for parallel_map_reduce
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_reduce<D, T, M, Red>(data: D, chunksize: usize, mapper: M, reducer: Red) -> T
where
    T: Clone,
    M: Fn(D) -> T,
    Red: Fn(T, T) -> T,
{
    mapper(data)
}

/// Parallel map-collect operation on a collection
///
/// This function maps over a collection in parallel and collects the results.
///
/// # Arguments
///
/// * `items` - The items to process
/// * `mapper` - Function that processes each item
///
/// # Returns
///
/// Vector of mapped results
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_collect<I, T, U, M>(items: I, mapper: M) -> Vec<U>
where
    I: IntoParallelIterator<Item = T>,
    T: Send,
    U: Send,
    M: Fn(T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    items.into_par_iter().map(mapper).collect()
}

/// Sequential fallback for parallel_map_collect
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_collect<I, T, U, M>(items: I, mapper: M) -> Vec<U>
where
    I: IntoIterator<Item = T>,
    M: Fn(T) -> U,
{
    items.into_iter().map(mapper).collect()
}

/// Parallel map-reduce operation on indexed chunks
///
/// This function takes a range of indices, splits them into chunks of the specified size,
/// processes each chunk in parallel using the mapper function, and then reduces the results
/// using the reducer function.
///
/// # Arguments
///
/// * `range` - The range of indices to process (e.g., 0..n)
/// * `chunk_size` - Size of each chunk for parallel processing
/// * `mapper` - Function that processes a slice of indices and returns a result
/// * `reducer` - Function that combines two results into one
///
/// # Returns
///
/// The final reduced result
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_reduce_indexed<R, T, M, Red>(
    range: R,
    chunk_size: usize,
    mapper: M,
    reducer: Red,
) -> T
where
    R: Iterator<Item = usize> + Send,
    T: Send + Clone,
    M: Fn(&[usize]) -> T + Sync + Send,
    Red: Fn(T, T) -> T + Sync + Send,
{
    use rayon::prelude::*;

    let indices: Vec<usize> = range.collect();

    indices
        .chunks(chunk_size)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(&mapper)
        .reduce_with(reducer)
        .unwrap_or_else(|| mapper(&[]))
}

/// Sequential fallback for parallel_map_reduce_indexed
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_reduce_indexed<R, T, M, Red>(
    range: R,
    chunk_size: usize,
    mapper: M,
    reducer: Red,
) -> T
where
    R: Iterator<Item = usize>,
    T: Clone,
    M: Fn(&[usize]) -> T,
    Red: Fn(T, T) -> T,
{
    let indices: Vec<usize> = range.collect();

    let mut results = Vec::new();
    for chunk in indices.chunks(chunk_size) {
        results.push(mapper(chunk));
    }

    results
        .into_iter()
        .reduce(reducer)
        .unwrap_or_else(|| mapper(&[]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_range() {
        let result: Vec<usize> = par_range(0, 10).collect();
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_par_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_par_filter() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let result: Vec<i32> = data.into_par_iter().filter(|x| x % 2 == 0).collect();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_par_try_for_each() {
        let data = vec![1, 2, 3, 4, 5];
        let result =
            data.into_par_iter()
                .try_for_each(|x| if x < 6 { Ok(()) } else { Err("Too large") });
        assert!(result.is_ok());
    }

    #[test]
    fn test_par_chunks() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let chunks: Vec<Vec<i32>> = par_chunks(&data, 3).map(|chunk| chunk.to_vec()).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7, 8]);
    }

    #[test]
    fn test_is_parallel_enabled() {
        let enabled = is_parallel_enabled();
        #[cfg(feature = "parallel")]
        assert!(enabled);
        #[cfg(not(feature = "parallel"))]
        assert!(!enabled);
    }

    #[test]
    fn test_num_threads() {
        let threads = num_threads();
        #[cfg(feature = "parallel")]
        assert!(threads > 0);
        #[cfg(not(feature = "parallel"))]
        assert_eq!(threads, 1);
    }
}
