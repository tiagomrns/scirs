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
pub fn par_range(start: usize, end: usize) -> impl ParallelIterator<Item = usize> {
    (start..end).into_par_iter()
}

/// Helper function for parallel chunks processing
#[cfg(feature = "parallel")]
pub fn par_chunks<T: Sync>(slice: &[T], chunk_size: usize) -> rayon::slice::Chunks<'_, T> {
    slice.par_chunks(chunk_size)
}

/// Sequential fallback for par_chunks
#[cfg(not(feature = "parallel"))]
pub fn par_chunks<T>(slice: &[T], chunk_size: usize) -> std::slice::Chunks<'_, T> {
    slice.chunks(chunk_size)
}

/// Helper function for parallel mutable chunks processing
#[cfg(feature = "parallel")]
pub fn par_chunks_mut<T: Send>(
    slice: &mut [T],
    chunk_size: usize,
) -> rayon::slice::ChunksMut<'_, T> {
    slice.par_chunks_mut(chunk_size)
}

/// Sequential fallback for par_chunks_mut
#[cfg(not(feature = "parallel"))]
pub fn par_chunks_mut<T>(slice: &mut [T], chunk_size: usize) -> std::slice::ChunksMut<'_, T> {
    slice.chunks_mut(chunk_size)
}

/// Check if parallel processing is available
pub fn is_parallel_enabled() -> bool {
    cfg!(feature = "parallel")
}

/// Get the number of threads that would be used for parallel operations
#[cfg(feature = "parallel")]
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Sequential fallback returns 1
#[cfg(not(feature = "parallel"))]
pub fn num_threads() -> usize {
    1
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
