//! Data loading utilities

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Load a dataset from a CSV file
pub fn load_csv<P: AsRef<Path>>(
    path: P,
    has_header: bool,
    target_column: Option<usize>,
) -> Result<Dataset> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(file);

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut header: Option<Vec<String>> = None;

    // Read header if needed
    if has_header {
        let headers = reader.headers().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to read CSV headers: {}", e))
        })?;
        header = Some(headers.iter().map(|s| s.to_string()).collect());
    }

    // Read rows
    for result in reader.records() {
        let record = result.map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to read CSV record: {}", e))
        })?;

        let values: Vec<f64> = record
            .iter()
            .map(|s| {
                s.parse::<f64>().map_err(|_| {
                    DatasetsError::InvalidFormat(format!("Failed to parse value: {}", s))
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        if !values.is_empty() {
            records.push(values);
        }
    }

    if records.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "CSV file is empty".to_string(),
        ));
    }

    // Create data array and target array if needed
    let n_rows = records.len();
    let n_cols = records[0].len();

    let (data, target, feature_names, _target_name) = if let Some(idx) = target_column {
        if idx >= n_cols {
            return Err(DatasetsError::InvalidFormat(format!(
                "Target column index {} is out of bounds (max: {})",
                idx,
                n_cols - 1
            )));
        }

        let mut data_array = Array2::zeros((n_rows, n_cols - 1));
        let mut target_array = Array1::zeros(n_rows);

        for (i, row) in records.iter().enumerate() {
            let mut data_col = 0;
            for (j, &val) in row.iter().enumerate() {
                if j == idx {
                    target_array[i] = val;
                } else {
                    data_array[[i, data_col]] = val;
                    data_col += 1;
                }
            }
        }

        let feature_names = header.as_ref().map(|h| {
            let mut names = Vec::new();
            for (j, name) in h.iter().enumerate() {
                if j != idx {
                    names.push(name.clone());
                }
            }
            names
        });

        (
            data_array,
            Some(target_array),
            feature_names,
            header.as_ref().map(|h| h[idx].clone()),
        )
    } else {
        let mut data_array = Array2::zeros((n_rows, n_cols));

        for (i, row) in records.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data_array[[i, j]] = val;
            }
        }

        (data_array, None, header, None)
    };

    let mut dataset = Dataset::new(data, target);

    if let Some(names) = feature_names {
        dataset = dataset.with_feature_names(names);
    }

    Ok(dataset)
}

/// Load a dataset from a JSON file
pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let reader = BufReader::new(file);

    let dataset: Dataset = serde_json::from_reader(reader)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to parse JSON: {}", e)))?;

    Ok(dataset)
}

/// Save a dataset to a JSON file
pub fn save_json<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<()> {
    let file = File::create(path).map_err(DatasetsError::IoError)?;

    serde_json::to_writer_pretty(file, dataset)
        .map_err(|e| DatasetsError::SerdeError(format!("Failed to write JSON: {}", e)))?;

    Ok(())
}

/// Load raw data from a file
pub fn load_raw<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut buffer = Vec::new();

    file.read_to_end(&mut buffer)
        .map_err(DatasetsError::IoError)?;

    Ok(buffer)
}
