//\! Utility functions and data structures for datasets

use crate::error::{DatasetsError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Helper module for serializing ndarray types with serde
mod serde_array {
    use ndarray::{Array1, Array2};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::vec::Vec;

    pub fn serialize_array2<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let shape = array.shape();
        let mut vec = Vec::with_capacity(shape[0] * shape[1] + 2);

        // Store shape at the beginning
        vec.push(shape[0] as f64);
        vec.push(shape[1] as f64);

        // Store data
        vec.extend(array.iter().cloned());

        vec.serialize(serializer)
    }

    pub fn deserialize_array2<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        if vec.len() < 2 {
            return Err(serde::de::Error::custom("Invalid array2 serialization"));
        }

        let nrows = vec[0] as usize;
        let ncols = vec[1] as usize;

        if vec.len() != nrows * ncols + 2 {
            return Err(serde::de::Error::custom("Invalid array2 serialization"));
        }

        let data = vec[2..].to_vec();
        match Array2::from_shape_vec((nrows, ncols), data) {
            Ok(array) => Ok(array),
            Err(_) => Err(serde::de::Error::custom("Failed to reshape array2")),
        }
    }

    #[allow(dead_code)]
    pub fn serialize_array1<S>(array: &Array1<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vec = array.to_vec();
        vec.serialize(serializer)
    }

    pub fn deserialize_array1<'de, D>(deserializer: D) -> Result<Array1<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        Ok(Array1::from(vec))
    }
}

/// Represents a dataset with features, optional targets, and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Features/data matrix (n_samples, n_features)
    #[serde(
        serialize_with = "serde_array::serialize_array2",
        deserialize_with = "serde_array::deserialize_array2"
    )]
    pub data: Array2<f64>,

    /// Optional target values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Array1<f64>>,

    /// Optional target names for classification problems
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_names: Option<Vec<String>>,

    /// Optional feature names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<Vec<String>>,

    /// Optional descriptions for each feature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_descriptions: Option<Vec<String>>,

    /// Optional dataset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Optional dataset metadata
    pub metadata: HashMap<String, String>,
}

// Helper module for serializing Option<Array1<f64>>
mod optional_array1 {
    use super::serde_array;
    use ndarray::Array1;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    #[allow(dead_code)]
    pub fn serialize<S>(array_opt: &Option<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array_opt {
            Some(array) => {
                #[derive(Serialize)]
                struct Helper<'a>(&'a Array1<f64>);

                #[derive(Serialize)]
                struct Wrapper<'a> {
                    #[serde(
                        serialize_with = "serde_array::serialize_array1",
                        deserialize_with = "serde_array::deserialize_array1"
                    )]
                    value: &'a Array1<f64>,
                }

                Wrapper { value: array }.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array1<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(
                serialize_with = "serde_array::serialize_array1",
                deserialize_with = "serde_array::deserialize_array1"
            )]
            #[allow(dead_code)]
            value: Array1<f64>,
        }

        Option::<Wrapper>::deserialize(deserializer).map(|opt_wrapper| opt_wrapper.map(|w| w.value))
    }
}

impl Dataset {
    /// Create a new dataset with the given data and target
    pub fn new(data: Array2<f64>, target: Option<Array1<f64>>) -> Self {
        Dataset {
            data,
            target,
            target_names: None,
            feature_names: None,
            feature_descriptions: None,
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Add target names to the dataset
    pub fn with_target_names(mut self, target_names: Vec<String>) -> Self {
        self.target_names = Some(target_names);
        self
    }

    /// Add feature names to the dataset
    pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Self {
        self.feature_names = Some(feature_names);
        self
    }

    /// Add feature descriptions to the dataset
    pub fn with_feature_descriptions(mut self, feature_descriptions: Vec<String>) -> Self {
        self.feature_descriptions = Some(feature_descriptions);
        self
    }

    /// Add a description to the dataset
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add metadata to the dataset
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the number of samples in the dataset
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of features in the dataset
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Split the dataset into training and test sets
    pub fn train_test_split(
        &self,
        test_size: f64,
        random_seed: Option<u64>,
    ) -> Result<(Dataset, Dataset)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(DatasetsError::InvalidFormat(
                "test_size must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = self.n_samples();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Both train and test sets must have at least one sample".to_string(),
            ));
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };
        indices.shuffle(&mut rng);

        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];

        // Create training dataset
        let train_data = self.data.select(ndarray::Axis(0), train_indices);
        let train_target = self
            .target
            .as_ref()
            .map(|t| t.select(ndarray::Axis(0), train_indices));

        let mut train_dataset = Dataset::new(train_data, train_target);
        if let Some(feature_names) = &self.feature_names {
            train_dataset = train_dataset.with_feature_names(feature_names.clone());
        }
        if let Some(description) = &self.description {
            train_dataset = train_dataset.with_description(description.clone());
        }

        // Create test dataset
        let test_data = self.data.select(ndarray::Axis(0), test_indices);
        let test_target = self
            .target
            .as_ref()
            .map(|t| t.select(ndarray::Axis(0), test_indices));

        let mut test_dataset = Dataset::new(test_data, test_target);
        if let Some(feature_names) = &self.feature_names {
            test_dataset = test_dataset.with_feature_names(feature_names.clone());
        }
        if let Some(description) = &self.description {
            test_dataset = test_dataset.with_description(description.clone());
        }

        Ok((train_dataset, test_dataset))
    }
}

/// Helper function to normalize data (zero mean, unit variance)
pub fn normalize(data: &mut Array2<f64>) {
    let n_features = data.ncols();

    for j in 0..n_features {
        let mut column = data.column_mut(j);

        // Calculate mean and std
        let mean = column.mean().unwrap_or(0.0);
        let std = column.std(0.0);

        // Avoid division by zero
        if std > 1e-10 {
            column.mapv_inplace(|x| (x - mean) / std);
        }
    }
}

/// Trait extension for Array2 to calculate mean and standard deviation
#[allow(dead_code)]
trait StatsExt {
    fn mean(&self) -> Option<f64>;
    fn std(&self, ddof: f64) -> f64;
}

impl StatsExt for ndarray::ArrayView1<'_, f64> {
    fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        let sum: f64 = self.sum();
        Some(sum / self.len() as f64)
    }

    fn std(&self, ddof: f64) -> f64 {
        if self.is_empty() {
            return 0.0;
        }

        let n = self.len() as f64;
        let mean = self.mean().unwrap_or(0.0);

        let mut sum_sq = 0.0;
        for &x in self.iter() {
            let diff = x - mean;
            sum_sq += diff * diff;
        }

        let divisor = n - ddof;
        if divisor <= 0.0 {
            return 0.0;
        }

        (sum_sq / divisor).sqrt()
    }
}
