//! Machine Learning evaluation metrics module for SciRS2
//!
//! This module provides functions for evaluating machine learning models
//! including classification, regression, and clustering metrics, as well as
//! model evaluation utilities like cross-validation and train-test split.
//!
//! # Classification Metrics
//!
//! Classification metrics evaluate the performance of classification models:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::{accuracy_score, precision_score, f1_score};
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
//! let precision = precision_score(&y_true, &y_pred, 1).unwrap();
//! let f1 = f1_score(&y_true, &y_pred, 1).unwrap();
//! ```
//!
//! # Regression Metrics
//!
//! Regression metrics evaluate the performance of regression models:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::regression::{mean_squared_error, r2_score};
//!
//! let y_true = array![3.0, -0.5, 2.0, 7.0];
//! let y_pred = array![2.5, 0.0, 2.0, 8.0];
//!
//! let mse = mean_squared_error(&y_true, &y_pred).unwrap();
//! let r2 = r2_score(&y_true, &y_pred).unwrap();
//! ```
//!
//! # Clustering Metrics
//!
//! Clustering metrics evaluate the performance of clustering algorithms:
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_metrics::clustering::silhouette_score;
//!
//! // Create a small dataset with 2 clusters
//! let X = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.5, 1.8,
//!     1.2, 2.2,
//!     5.0, 6.0,
//!     5.2, 5.8,
//!     5.5, 6.2,
//! ]).unwrap();
//!
//! let labels = array![0, 0, 0, 1, 1, 1];
//!
//! let score = silhouette_score(&X, &labels, "euclidean").unwrap();
//! ```
//!
//! # Model Evaluation Utilities
//!
//! Utilities for model evaluation like cross-validation:
//!
//! ```
//! use ndarray::{Array, Ix1};
//! use scirs2_metrics::evaluation::train_test_split;
//!
//! let x = Array::<f64, _>::linspace(0., 9., 10).into_shape(Ix1(10)).unwrap();
//! let y = &x * 2.;
//!
//! let (train_arrays, test_arrays) = train_test_split(&[&x, &y], 0.3, Some(42)).unwrap();
//! ```

#![warn(missing_docs)]

pub mod classification;
pub mod clustering;
pub mod error;
pub mod evaluation;
pub mod regression;
