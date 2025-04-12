//! Distribution traits
//!
//! This module defines traits for different types of probability distributions.
//! These traits provide a unified interface for working with various distributions
//! in the library.

use crate::error::StatsResult;
use ndarray::Array1;
use num_traits::Float;

/// Common trait for all probability distributions
pub trait Distribution<F: Float> {
    /// Return the mean of the distribution
    fn mean(&self) -> F;

    /// Return the variance of the distribution
    fn var(&self) -> F;

    /// Return the standard deviation of the distribution
    fn std(&self) -> F;

    /// Generate random samples from this distribution
    fn rvs(&self, size: usize) -> StatsResult<Array1<F>>;

    /// Return the entropy of the distribution
    fn entropy(&self) -> F;
}

/// Trait for continuous probability distributions
pub trait ContinuousDistribution<F: Float>: Distribution<F> {
    /// Calculate the probability density function (PDF) at a given point
    fn pdf(&self, x: F) -> F;

    /// Calculate the cumulative distribution function (CDF) at a given point
    fn cdf(&self, x: F) -> F;

    /// Calculate the survival function (1 - CDF) at a given point
    fn sf(&self, x: F) -> F {
        F::one() - self.cdf(x)
    }

    /// Calculate the inverse cumulative distribution function (quantile function)
    /// for a given probability
    fn ppf(&self, p: F) -> StatsResult<F>;

    /// Calculate the inverse survival function for a given probability
    fn isf(&self, p: F) -> StatsResult<F> {
        self.ppf(F::one() - p)
    }

    /// Calculate the hazard function (PDF / SF) at a given point
    fn hazard(&self, x: F) -> F {
        let survival = self.sf(x);
        if survival <= F::epsilon() {
            F::infinity()
        } else {
            self.pdf(x) / survival
        }
    }

    /// Calculate the cumulative hazard function (-ln(SF)) at a given point
    fn cumhazard(&self, x: F) -> F {
        -self.sf(x).ln()
    }
}

/// Trait for discrete probability distributions
pub trait DiscreteDistribution<F: Float>: Distribution<F> {
    /// Calculate the probability mass function (PMF) at a given point
    fn pmf(&self, x: F) -> F;

    /// Calculate the cumulative distribution function (CDF) at a given point
    fn cdf(&self, x: F) -> F;

    /// Calculate the survival function (1 - CDF) at a given point
    fn sf(&self, x: F) -> F {
        F::one() - self.cdf(x)
    }

    /// Calculate the inverse cumulative distribution function (quantile function)
    /// for a given probability
    fn ppf(&self, p: F) -> StatsResult<F>;

    /// Calculate the inverse survival function for a given probability
    fn isf(&self, p: F) -> StatsResult<F> {
        self.ppf(F::one() - p)
    }

    /// Calculate the log of the probability mass function
    fn logpmf(&self, x: F) -> F {
        self.pmf(x).ln()
    }
}

/// Trait for multivariate probability distributions
pub trait MultivariateDistribution<F: Float>: Distribution<F> {
    /// Calculate the probability density function (PDF) at a given point
    fn pdf(&self, x: &[F]) -> F;

    /// Calculate the log of the probability density function
    fn logpdf(&self, x: &[F]) -> F {
        self.pdf(x).ln()
    }

    /// Generate a single random sample from this distribution
    fn rvs_single(&self) -> StatsResult<Vec<F>>;
}
