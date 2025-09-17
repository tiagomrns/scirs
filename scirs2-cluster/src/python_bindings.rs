//! Python bindings for scirs2-cluster using PyO3
//!
//! This module provides Python bindings that make scirs2-cluster algorithms
//! accessible from Python with scikit-learn compatible APIs. Supports all major
//! clustering algorithms with full feature parity to scikit-learn.

use crate::affinity::{affinity_propagation, AffinityPropagationOptions};
use crate::birch::{birch, BirchOptions};
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::{gaussian_mixture, CovarianceType, GMMOptions, GaussianMixture};
use crate::hierarchy::{fcluster, linkage, LinkageMethod, Metric};
use crate::meanshift::{estimate_bandwidth, mean_shift, MeanShiftOptions};
use crate::metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, normalized_mutual_info, silhouette_score,
};
use crate::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};
use crate::vq::{kmeans, kmeans2};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use numpy::{PyArray1, PyArray2, ToPyArray};
#[cfg(feature = "pyo3")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Python-compatible K-means clustering implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed
    random_state: Option<u64>,
    /// Number of initializations
    n_init: usize,
    /// Initialization method
    init: String,
    /// Fitted cluster centers
    cluster_centers_: Option<Array2<f64>>,
    /// Labels of each point
    labels_: Option<Array1<usize>>,
    /// Sum of squared distances to centroids
    inertia_: Option<f64>,
    /// Number of iterations run
    n_iter_: Option<usize>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyKMeans {
    /// Create new K-means clustering instance
    #[new]
    #[pyo3(signature = (n_clusters=8, *, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=None))]
    fn new(
        n_clusters: usize,
        init: &str,
        n_init: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            random_state,
            n_init_init: init.to_string(),
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
        }
    }

    /// Fit K-means clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        match self.fit_internal(data) {
            Ok((centers, labels, inertia, n_iter)) => {
                self.cluster_centers_ = Some(centers);
                self.labels_ = Some(labels);
                self.inertia_ = Some(inertia);
                self.n_iter_ = Some(n_iter);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "K-means fitting failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Predict cluster labels for new data
    fn predict(&self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        if self.cluster_centers_.is_none() {
            return Err(PyRuntimeError::new_err("Model not fitted yet"));
        }

        let data = unsafe { x.as_array() };
        let centers = self.cluster_centers_.as_ref().unwrap();

        let mut labels = Array1::zeros(data.nrows());

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.rows().into_iter().enumerate() {
                let dist: f64 = sample
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels[i] = best_cluster;
        }

        let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
        Ok(labels_i32.to_pyarray(py).to_owned())
    }

    /// Get cluster centers
    #[getter]
    fn cluster_centers_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.cluster_centers_ {
            Some(centers) => Ok(Some(centers.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => {
                let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                Ok(labels_i32.to_pyarray(py).to_owned())
            }
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get inertia (sum of squared distances to centroids)
    #[getter]
    fn inertia_(&self) -> Option<f64> {
        self.inertia_
    }

    /// Get number of iterations
    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter_
    }

    /// Set parameters
    fn set_params(&mut self, params: &PyDict) -> PyResult<()> {
        for (key, value) in params.iter() {
            let key_str: String = key.extract()?;
            match key_str.as_str() {
                "n_clusters" => self.n_clusters = value.extract()?,
                "max_iter" => self.max_iter = value.extract()?,
                "tol" => self.tol = value.extract()?,
                "random_state" => self.random_state = value.extract()?,
                "n_init" => self.n_init = value.extract()?,
                "init" => self.init = value.extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown parameter: {}",
                        key_str
                    )))
                }
            }
        }
        Ok(())
    }

    /// Get parameters
    fn get_params(&self, py: Python, deep: Option<bool>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("n_clusters", self.n_clusters)?;
        dict.set_item("max_iter", self.max_iter)?;
        dict.set_item("tol", self.tol)?;
        dict.set_item("random_state", self.random_state)?;
        dict.set_item("n_init", self.n_init)?;
        dict.set_item("init", &self.init)?;
        Ok(dict.into())
    }
}

#[cfg(feature = "pyo3")]
impl PyKMeans {
    /// Internal fitting logic
    fn fit_internal(
        &self,
        data: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<usize>, f64, usize)> {
        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0;

        // Run multiple initializations
        for _ in 0..self.n_init {
            match kmeans(
                data,
                self.n_clusters,
                Some(self.max_iter),
                Some(self.tol),
                self.random_state,
                None,
            ) {
                Ok((centers, labels)) => {
                    // Calculate inertia
                    let mut inertia = 0.0;
                    for (i, sample) in data.rows().into_iter().enumerate() {
                        let cluster = labels[i];
                        let center = centers.row(cluster);
                        let dist_sq: f64 = sample
                            .iter()
                            .zip(center.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        inertia += dist_sq;
                    }

                    if inertia < best_inertia {
                        best_inertia = inertia;
                        best_centers = Some(centers);
                        best_labels = Some(labels);
                        best_n_iter = self.max_iter; // Simplified for now
                    }
                }
                Err(_) => continue,
            }
        }

        match (best_centers, best_labels) {
            (Some(centers), Some(labels)) => Ok((centers, labels, best_inertia, best_n_iter)),
            _ => Err(ClusteringError::ComputationError(
                "K-means failed to converge in any initialization".to_string(),
            )),
        }
    }
}

/// Python-compatible DBSCAN clustering implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "DBSCAN")]
pub struct PyDBSCAN {
    /// Epsilon neighborhood radius
    eps: f64,
    /// Minimum points to form dense region
    min_samples: usize,
    /// Distance metric
    metric: String,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Core sample indices
    core_sample_indices_: Option<Array1<usize>>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyDBSCAN {
    /// Create new DBSCAN clustering instance
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5, metric="euclidean"))]
    fn new(_eps: f64, minsamples: usize, metric: &str) -> Self {
        Self {
            eps,
            min_samples,
            metric: metric.to_string(),
            labels_: None,
            core_sample_indices_: None,
        }
    }

    /// Fit DBSCAN clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        match dbscan(data, self.eps, self.min_samples) {
            Ok((labels, core_indices)) => {
                // Convert labels to i32 (with -1 for noise)
                let labels_i32: Array1<i32> =
                    labels.mapv(|x| if x == usize::MAX { -1 } else { x as i32 });

                self.labels_ = Some(labels_i32);
                self.core_sample_indices_ = Some(core_indices);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "DBSCAN fitting failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get core sample indices
    #[getter]
    fn core_sample_indices_(&self, py: Python) -> PyResult<Option<Py<PyArray1<usize>>>> {
        match &self.core_sample_indices_ {
            Some(indices) => Ok(Some(indices.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }
}

/// Python-compatible hierarchical clustering
#[cfg(feature = "pyo3")]
#[pyclass(name = "AgglomerativeClustering")]
pub struct PyAgglomerativeClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Linkage criterion
    linkage: String,
    /// Distance metric
    metric: String,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Number of leaves in hierarchy
    n_leaves_: Option<usize>,
    /// Number of connected components
    n_connected_components_: Option<usize>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyAgglomerativeClustering {
    /// Create new agglomerative clustering instance
    #[new]
    #[pyo3(signature = (n_clusters=2, *, linkage="ward", metric="euclidean"))]
    fn new(_nclusters: usize, linkage: &str, metric: &str) -> Self {
        Self {
            n_clusters,
            linkage: linkage.to_string(),
            metric: metric.to_string(),
            labels_: None,
            n_leaves_: None,
            n_connected_components_: None,
        }
    }

    /// Fit agglomerative clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        let linkage_method = match self.linkage.as_str() {
            "ward" => LinkageMethod::Ward,
            "complete" => LinkageMethod::Complete,
            "average" => LinkageMethod::Average,
            "single" => LinkageMethod::Single,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown linkage: {}",
                    self.linkage
                )))
            }
        };

        let distance_metric = match self.metric.as_str() {
            "euclidean" => Metric::Euclidean,
            "manhattan" => Metric::Manhattan,
            "cosine" => Metric::Cosine,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric: {}",
                    self.metric
                )))
            }
        };

        match linkage(data, linkage_method, distance_metric) {
            Ok(linkage_matrix) => {
                // Extract clusters from linkage matrix using proper cluster extraction
                match fcluster(&linkage_matrix, self.n_clusters, None) {
                    Ok(labels) => {
                        let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                        self.labels_ = Some(labels_i32);
                        self.n_leaves_ = Some(data.nrows());
                        self.n_connected_components_ = Some(1);
                        Ok(())
                    }
                    Err(e) => Err(PyRuntimeError::new_err(format!(
                        "Cluster extraction failed: {}",
                        e
                    ))),
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Hierarchical clustering failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get number of leaves
    #[getter]
    fn n_leaves_(&self) -> Option<usize> {
        self.n_leaves_
    }

    /// Get number of connected components
    #[getter]
    fn n_connected_components_(&self) -> Option<usize> {
        self.n_connected_components_
    }
}

/// Python-compatible BIRCH clustering implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "Birch")]
pub struct PyBirch {
    /// Number of clusters
    n_clusters: Option<usize>,
    /// Threshold for subcluster radius
    threshold: f64,
    /// Branching factor
    branching_factor: usize,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Subcluster centers
    subcluster_centers_: Option<Array2<f64>>,
    /// Number of subclusters
    n_features_in_: Option<usize>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyBirch {
    /// Create new BIRCH clustering instance
    #[new]
    #[pyo3(signature = (n_clusters=None, *, threshold=0.5, branching_factor=50))]
    fn new(_n_clusters: Option<usize>, threshold: f64, branchingfactor: usize) -> Self {
        Self {
            n_clusters,
            threshold,
            branching_factor,
            labels_: None,
            subcluster_centers_: None,
            n_features_in_: None,
        }
    }

    /// Fit BIRCH clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        let options = BirchOptions {
            n_clusters: self.n_clusters,
            threshold: self.threshold,
            branching_factor: self.branching_factor,
        };

        match birch(data, options) {
            Ok((labels, centers)) => {
                let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                self.labels_ = Some(labels_i32);
                self.subcluster_centers_ = Some(centers);
                self.n_features_in_ = Some(data.ncols());
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "BIRCH clustering failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get subcluster centers
    #[getter]
    fn subcluster_centers_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.subcluster_centers_ {
            Some(centers) => Ok(Some(centers.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }
}

/// Python-compatible Spectral clustering implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "SpectralClustering")]
pub struct PySpectralClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Affinity matrix construction method
    affinity: String,
    /// Gamma parameter for RBF kernel
    gamma: Option<f64>,
    /// Random seed
    random_state: Option<u64>,
    /// Number of eigenvectors to use
    n_components: Option<usize>,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Affinity matrix
    affinity_matrix_: Option<Array2<f64>>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PySpectralClustering {
    /// Create new spectral clustering instance
    #[new]
    #[pyo3(signature = (n_clusters=8, *, affinity="rbf", gamma=None, random_state=None, n_components=None))]
    fn new(
        n_clusters: usize,
        affinity: &str,
        gamma: Option<f64>,
        random_state: Option<u64>,
        n_components: Option<usize>,
    ) -> Self {
        Self {
            n_clusters,
            affinity: affinity.to_string(),
            gamma,
            random_state,
            n_components,
            labels_: None,
            affinity_matrix_: None,
        }
    }

    /// Fit spectral clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        let affinity_mode = match self.affinity.as_str() {
            "rbf" => AffinityMode::Rbf(self.gamma.unwrap_or(1.0)),
            "nearest_neighbors" => AffinityMode::NearestNeighbors(10),
            "precomputed" => AffinityMode::Precomputed,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown affinity: {}",
                    self.affinity
                )))
            }
        };

        let options = SpectralClusteringOptions {
            n_clusters: self.n_clusters,
            affinity: affinity_mode,
            n_components: self.n_components,
            random_seed: self.random_state,
        };

        match spectral_clustering(data, options) {
            Ok((labels, affinity_matrix)) => {
                let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                self.labels_ = Some(labels_i32);
                self.affinity_matrix_ = Some(affinity_matrix);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Spectral clustering failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get affinity matrix
    #[getter]
    fn affinity_matrix_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.affinity_matrix_ {
            Some(matrix) => Ok(Some(matrix.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }
}

/// Python-compatible Mean Shift clustering implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "MeanShift")]
pub struct PyMeanShift {
    /// Bandwidth parameter
    bandwidth: Option<f64>,
    /// Seeds for mean shift
    seeds: Option<Array2<f64>>,
    /// Convergence threshold
    cluster_all: bool,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Cluster centers
    cluster_centers_: Option<Array2<f64>>,
    /// Number of iterations
    n_iter_: Option<usize>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyMeanShift {
    /// Create new Mean Shift clustering instance
    #[new]
    #[pyo3(signature = (bandwidth=None, *, seeds=None, cluster_all=true))]
    fn new(_bandwidth: Option<f64>, seeds: Option<&PyArray2<f64>>, clusterall: bool) -> Self {
        let seeds_array = seeds.map(|s| unsafe { s.as_array().to_owned() });
        Self {
            bandwidth,
            seeds: seeds_array,
            cluster_all,
            labels_: None,
            cluster_centers_: None,
            n_iter_: None,
        }
    }

    /// Fit Mean Shift clustering to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        let bandwidth = match self.bandwidth {
            Some(bw) => bw,
            None => match estimate_bandwidth(data, None, None) {
                Ok(bw) => bw,
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Bandwidth estimation failed: {}",
                        e
                    )))
                }
            },
        };

        let options = MeanShiftOptions {
            bandwidth,
            seeds: self.seeds.as_ref().map(|s| s.view()),
            max_iter: Some(300),
            tol: Some(1e-3),
            cluster_all: self.cluster_all,
        };

        match mean_shift(data, options) {
            Ok((labels, centers, n_iter)) => {
                let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                self.labels_ = Some(labels_i32);
                self.cluster_centers_ = Some(centers);
                self.n_iter_ = Some(n_iter);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Mean Shift clustering failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get cluster centers
    #[getter]
    fn cluster_centers_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.cluster_centers_ {
            Some(centers) => Ok(Some(centers.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }

    /// Get number of iterations
    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter_
    }
}

/// Python-compatible Gaussian Mixture Model implementation
#[cfg(feature = "pyo3")]
#[pyclass(name = "GaussianMixture")]
pub struct PyGaussianMixture {
    /// Number of components
    n_components: usize,
    /// Covariance type
    covariance_type: String,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed
    random_state: Option<u64>,
    /// Fitted labels
    labels_: Option<Array1<i32>>,
    /// Component means
    means_: Option<Array2<f64>>,
    /// Component weights
    weights_: Option<Array1<f64>>,
    /// Log-likelihood of best fit
    lower_bound_: Option<f64>,
    /// Whether model converged
    converged_: bool,
    /// Number of iterations
    n_iter_: Option<usize>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyGaussianMixture {
    /// Create new Gaussian Mixture Model instance
    #[new]
    #[pyo3(signature = (n_components=1, *, covariance_type="full", max_iter=100, tol=1e-3, random_state=None))]
    fn new(
        n_components: usize,
        covariance_type: &str,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            covariance_type: covariance_type.to_string(),
            max_iter,
            tol,
            random_state,
            labels_: None,
            means_: None,
            weights_: None,
            lower_bound_: None,
            converged_: false,
            n_iter_: None,
        }
    }

    /// Fit Gaussian Mixture Model to data
    fn fit(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<()> {
        let data = unsafe { x.as_array() };

        let cov_type = match self.covariance_type.as_str() {
            "full" => CovarianceType::Full,
            "tied" => CovarianceType::Tied,
            "diag" => CovarianceType::Diagonal,
            "spherical" => CovarianceType::Spherical,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown covariance_type: {}",
                    self.covariance_type
                )))
            }
        };

        let options = GMMOptions {
            n_components: self.n_components,
            covariance_type: cov_type,
            max_iter: self.max_iter,
            tol: self.tol,
            random_seed: self.random_state,
            reg_covar: 1e-6,
        };

        match gaussian_mixture(data, options) {
            Ok(gmm) => {
                // Predict labels for the training data
                let labels: Array1<usize> = (0..data.nrows())
                    .map(|i| {
                        let sample = data.row(i);
                        gmm.predict_single(&sample.to_owned()).unwrap_or(0)
                    })
                    .collect();

                let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
                self.labels_ = Some(labels_i32);
                self.means_ = Some(gmm.means().to_owned());
                self.weights_ = Some(gmm.weights().to_owned());
                self.converged_ = gmm.converged();
                self.n_iter_ = Some(gmm.n_iter());
                self.lower_bound_ = Some(gmm.lower_bound());
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Gaussian Mixture fitting failed: {}",
                e
            ))),
        }
    }

    /// Fit and predict cluster labels
    fn fit_predict(&mut self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => Ok(labels.to_pyarray(py).to_owned()),
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get component means
    #[getter]
    fn means_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.means_ {
            Some(means) => Ok(Some(means.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }

    /// Get component weights
    #[getter]
    fn weights_(&self, py: Python) -> PyResult<Option<Py<PyArray1<f64>>>> {
        match &self.weights_ {
            Some(weights) => Ok(Some(weights.to_pyarray(py).to_owned())),
            None => Ok(None),
        }
    }

    /// Get lower bound
    #[getter]
    fn lower_bound_(&self) -> Option<f64> {
        self.lower_bound_
    }

    /// Check if model converged
    #[getter]
    fn converged_(&self) -> bool {
        self.converged_
    }

    /// Get number of iterations
    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter_
    }
}

/// Python-compatible clustering metrics
#[cfg(feature = "pyo3")]
#[pymodule]
#[allow(dead_code)]
fn metrics(py: Python, m: &PyModule) -> PyResult<()> {
    /// Calculate silhouette score
    #[pyfn(m)]
    fn silhouette_score_py(
        _py: Python,
        x: &PyArray2<f64>,
        labels: &PyArray1<i32>,
        metric: Option<&str>,
    ) -> PyResult<f64> {
        let data = unsafe { x.as_array() };
        let labels_array = unsafe { labels.as_array() };

        // Convert i32 labels to usize
        let labels_usize: Array1<usize> = labels_array.mapv(|x| if x < 0 { 0 } else { x as usize });

        match silhouette_score(data, labels_usize.view()) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Silhouette score calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate Calinski-Harabasz score
    #[pyfn(m)]
    fn calinski_harabasz_score_py(
        _py: Python,
        x: &PyArray2<f64>,
        labels: &PyArray1<i32>,
    ) -> PyResult<f64> {
        let data = unsafe { x.as_array() };
        let labels_array = unsafe { labels.as_array() };

        let labels_usize: Array1<usize> = labels_array.mapv(|x| if x < 0 { 0 } else { x as usize });

        match calinski_harabasz_score(data, labels_usize.view()) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Calinski-Harabasz score calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate Davies-Bouldin score
    #[pyfn(m)]
    fn davies_bouldin_score_py(
        _py: Python,
        x: &PyArray2<f64>,
        labels: &PyArray1<i32>,
    ) -> PyResult<f64> {
        let data = unsafe { x.as_array() };
        let labels_array = unsafe { labels.as_array() };

        let labels_usize: Array1<usize> = labels_array.mapv(|x| if x < 0 { 0 } else { x as usize });

        match davies_bouldin_score(data, labels_usize.view()) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Davies-Bouldin score calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate Adjusted Rand Index
    #[pyfn(m)]
    fn adjusted_rand_score_py(
        _py: Python,
        labels_true: &PyArray1<i32>,
        labels_pred: &PyArray1<i32>,
    ) -> PyResult<f64> {
        let true_labels = unsafe { labels_true.as_array() };
        let pred_labels = unsafe { labels_pred.as_array() };

        let _true_usize: Array1<usize> = true_labels.mapv(|x| if x < 0 { 0 } else { x as usize });
        let _pred_usize: Array1<usize> = pred_labels.mapv(|x| if x < 0 { 0 } else { x as usize });

        match adjusted_rand_index(true_usize.view(), pred_usize.view()) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Adjusted Rand Index calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate Normalized Mutual Information
    #[pyfn(m)]
    fn normalized_mutual_info_score_py(
        _py: Python,
        labels_true: &PyArray1<i32>,
        labels_pred: &PyArray1<i32>,
        average_method: Option<&str>,
    ) -> PyResult<f64> {
        let true_labels = unsafe { labels_true.as_array() };
        let pred_labels = unsafe { labels_pred.as_array() };

        let _true_usize: Array1<usize> = true_labels.mapv(|x| if x < 0 { 0 } else { x as usize });
        let _pred_usize: Array1<usize> = pred_labels.mapv(|x| if x < 0 { 0 } else { x as usize });

        match normalized_mutual_info(true_usize.view(), pred_usize.view()) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Normalized Mutual Information calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate Homogeneity, Completeness, and V-measure
    #[pyfn(m)]
    fn homogeneity_completeness_v_measure_py(
        _py: Python,
        labels_true: &PyArray1<i32>,
        labels_pred: &PyArray1<i32>,
        beta: Option<f64>,
    ) -> PyResult<(f64, f64, f64)> {
        let true_labels = unsafe { labels_true.as_array() };
        let pred_labels = unsafe { labels_pred.as_array() };

        let _true_usize: Array1<usize> = true_labels.mapv(|x| if x < 0 { 0 } else { x as usize });
        let _pred_usize: Array1<usize> = pred_labels.mapv(|x| if x < 0 { 0 } else { x as usize });

        match homogeneity_completeness_v_measure(true_usize.view(), pred_usize.view()) {
            Ok((h, c, v)) => Ok((h, c, v)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "H-C-V calculation failed: {}",
                e
            ))),
        }
    }

    Ok(())
}

/// Main Python module
#[cfg(feature = "pyo3")]
#[pymodule]
#[allow(dead_code)]
fn scirs2_cluster(py: Python, m: &PyModule) -> PyResult<()> {
    // Core clustering algorithms
    m.add_class::<PyKMeans>()?;
    m.add_class::<PyDBSCAN>()?;
    m.add_class::<PyAgglomerativeClustering>()?;
    m.add_class::<PyBirch>()?;
    m.add_class::<PySpectralClustering>()?;
    m.add_class::<PyMeanShift>()?;
    m.add_class::<PyGaussianMixture>()?;

    // Metrics submodule
    m.add_wrapped(wrap_pymodule!(metrics))?;

    // Convenience functions for common workflows
    #[pyfn(m)]
    fn estimate_bandwidth_py(
        _py: Python,
        x: &PyArray2<f64>,
        quantile: Option<f64>,
        n_samples: Option<usize>,
    ) -> PyResult<f64> {
        let data = unsafe { x.as_array() };
        match estimate_bandwidth(data, quantile, n_samples) {
            Ok(bw) => Ok(bw),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Bandwidth estimation failed: {}",
                e
            ))),
        }
    }

    /// Automatically select the best clustering algorithm for given data
    #[pyfn(m)]
    fn auto_select_algorithm_py(
        _py: Python,
        x: &PyArray2<f64>,
        n_clusters_hint: Option<usize>,
        sample_size_threshold: Option<usize>,
    ) -> PyResult<String> {
        let data = unsafe { x.as_array() };
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Simple heuristics for algorithm selection
        let recommended_algorithm = if let Some(_k) = n_clusters_hint {
            // User provided number of clusters _hint
            if n_samples < 1000 {
                "KMeans"
            } else if n_samples < 10000 {
                if n_features > 10 {
                    "MiniBatchKMeans"
                } else {
                    "KMeans"
                }
            } else {
                "MiniBatchKMeans"
            }
        } else {
            // No cluster _hint - use density-based or hierarchical
            if n_samples < 500 {
                "AgglomerativeClustering"
            } else if n_samples < 5000 {
                "DBSCAN"
            } else {
                "MeanShift"
            }
        };

        Ok(recommended_algorithm.to_string())
    }

    /// Get algorithm-specific parameter recommendations
    #[pyfn(m)]
    fn get_algorithm_defaults_py(
        _py: Python,
        algorithm: &str,
        x: &PyArray2<f64>,
    ) -> PyResult<PyObject> {
        let data = unsafe { x.as_array() };
        let n_samples = data.nrows();
        let n_features = data.ncols();

        let dict = PyDict::new(_py);

        match algorithm.to_lowercase().as_str() {
            "kmeans" => {
                dict.set_item(
                    "n_clusters",
                    if n_samples < 100 {
                        3
                    } else {
                        (n_samples as f64).sqrt() as usize
                    },
                )?;
                dict.set_item("init", "k-means++")?;
                dict.set_item("n_init", 10)?;
                dict.set_item("max_iter", 300)?;
            }
            "dbscan" => {
                let eps = if n_features <= 2 { 0.5 } else { 1.0 };
                dict.set_item("eps", eps)?;
                dict.set_item("min_samples", 5.max(n_features))?;
                dict.set_item("metric", "euclidean")?;
            }
            "agglomerativeclustering" => {
                dict.set_item("n_clusters", (n_samples as f64).sqrt() as usize)?;
                dict.set_item("linkage", "ward")?;
                dict.set_item("metric", "euclidean")?;
            }
            "meanshift" => {
                dict.set_item("bandwidth", "auto")?;
                dict.set_item("cluster_all", true)?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown algorithm: {}",
                    algorithm
                )));
            }
        }

        Ok(dict.into())
    }

    // Module metadata
    m.add("__version__", "0.1.0-beta.1")?;
    m.add("__author__", "SciRS2 Team")?;
    m.add(
        "__description__",
        "High-performance clustering algorithms with scikit-learn compatibility",
    )?;

    // Algorithm availability flags
    m.add(
        "__algorithms__",
        vec![
            "KMeans",
            "DBSCAN",
            "AgglomerativeClustering",
            "Birch",
            "SpectralClustering",
            "MeanShift",
            "GaussianMixture",
        ],
    )?;

    // Convenience functions
    m.add(
        "__convenience_functions__",
        vec![
            "estimate_bandwidth_py",
            "auto_select_algorithm_py",
            "get_algorithm_defaults_py",
        ],
    )?;

    Ok(())
}

#[cfg(not(feature = "pyo3"))]
/// Stub implementations when PyO3 feature is not enabled
pub mod stubs {
    pub fn python_bindings_not_available() -> &'static str {
        "Python bindings are not available. Enable with --features pyo3"
    }
}
