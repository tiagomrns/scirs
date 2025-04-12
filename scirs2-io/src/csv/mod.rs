//! CSV file format support
//!
//! This module provides functionality for reading and writing CSV (Comma-Separated Values)
//! files, commonly used for storing tabular data.
//!
//! Features:
//! - Reading and writing CSV files with various configuration options
//! - Support for custom delimiters, quotes, and line endings
//! - Handling of missing values and type conversions
//! - Memory-efficient processing of large files
//! - Column-based I/O operations

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// CSV reader configuration
#[derive(Debug, Clone)]
pub struct CsvReaderConfig {
    /// Delimiter character (default: ',')
    pub delimiter: char,
    /// Quote character (default: '"')
    pub quote_char: char,
    /// Whether to trim whitespace from fields (default: false)
    pub trim: bool,
    /// Whether the file has a header row (default: true)
    pub has_header: bool,
    /// Comment character, lines starting with this will be ignored (default: None)
    pub comment_char: Option<char>,
    /// Skip rows at the beginning of the file (default: 0)
    pub skip_rows: usize,
    /// Maximum number of rows to read (default: None = all rows)
    pub max_rows: Option<usize>,
}

impl Default for CsvReaderConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            trim: false,
            has_header: true,
            comment_char: None,
            skip_rows: 0,
            max_rows: None,
        }
    }
}

/// Read a CSV file into a 2D array of strings
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `config` - Optional CSV reader configuration
///
/// # Returns
///
/// * `Result<(Vec<String>, Array2<String>)>` - Header labels and data as strings
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::csv::{read_csv, CsvReaderConfig};
///
/// // Read with default configuration
/// let (headers, data) = read_csv("data.csv", None).unwrap();
/// println!("Headers: {:?}", headers);
/// println!("Data shape: {:?}", data.shape());
///
/// // Read with custom configuration
/// let config = CsvReaderConfig {
///     delimiter: ';',
///     has_header: false,
///     ..Default::default()
/// };
/// let (_, data) = read_csv("data.csv", Some(config)).unwrap();
/// ```
pub fn read_csv<P: AsRef<Path>>(
    path: P,
    config: Option<CsvReaderConfig>,
) -> Result<(Vec<String>, Array2<String>)> {
    let config = config.unwrap_or_default();

    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let mut rows = Vec::new();

    // Skip rows if needed
    for _ in 0..config.skip_rows {
        if lines.next().is_none() {
            return Err(IoError::FormatError("Not enough rows in file".to_string()));
        }
    }

    // Read header if present
    let headers = if config.has_header {
        match lines.next() {
            Some(Ok(line)) => parse_csv_line(&line, &config),
            Some(Err(e)) => return Err(IoError::FileError(e.to_string())),
            None => return Err(IoError::FormatError("Empty file".to_string())),
        }
    } else {
        Vec::new()
    };

    // Read data
    let mut row_count = 0;
    for line_result in lines {
        // Break if we've read enough rows
        if let Some(max) = config.max_rows {
            if row_count >= max {
                break;
            }
        }

        let line = line_result.map_err(|e| IoError::FileError(e.to_string()))?;

        // Skip comment lines
        if let Some(comment_char) = config.comment_char {
            if line.trim().starts_with(comment_char) {
                continue;
            }
        }

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        let row = parse_csv_line(&line, &config);
        rows.push(row);
        row_count += 1;
    }

    // Check if we have any data
    if rows.is_empty() {
        return Err(IoError::FormatError("No data rows in file".to_string()));
    }

    // Determine the number of columns from the first row
    let num_cols = rows[0].len();

    // Ensure all rows have the same number of columns
    for (i, row) in rows.iter().enumerate() {
        if row.len() != num_cols {
            return Err(IoError::FormatError(format!(
                "Inconsistent number of columns: row {} has {} columns, expected {}",
                i + 1,
                row.len(),
                num_cols
            )));
        }
    }

    // Convert to Array2
    let num_rows = rows.len();
    let mut data = Array2::from_elem((num_rows, num_cols), String::new());

    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            data[[i, j]] = value.clone();
        }
    }

    Ok((headers, data))
}

/// Parse a CSV line into fields
fn parse_csv_line(line: &str, config: &CsvReaderConfig) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        // Handle quotes
        if c == config.quote_char {
            // Check for escaped quotes (double quotes)
            if in_quotes && chars.peek() == Some(&config.quote_char) {
                chars.next(); // Consume the second quote
                field.push(config.quote_char);
            } else {
                in_quotes = !in_quotes;
            }
        }
        // Handle delimiters
        else if c == config.delimiter && !in_quotes {
            let processed_field = if config.trim {
                field.trim().to_string()
            } else {
                field
            };
            fields.push(processed_field);
            field = String::new();
        }
        // Handle regular characters
        else {
            field.push(c);
        }
    }

    // Add the last field
    let processed_field = if config.trim {
        field.trim().to_string()
    } else {
        field
    };
    fields.push(processed_field);

    fields
}

/// Read a CSV file and convert to numeric arrays
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `config` - Optional CSV reader configuration
///
/// # Returns
///
/// * `Result<(Vec<String>, Array2<f64>)>` - Header labels and data as f64 values
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::csv::{read_csv_numeric, CsvReaderConfig};
///
/// let (headers, data) = read_csv_numeric("data.csv", None).unwrap();
/// println!("Numeric data shape: {:?}", data.shape());
/// ```
pub fn read_csv_numeric<P: AsRef<Path>>(
    path: P,
    config: Option<CsvReaderConfig>,
) -> Result<(Vec<String>, Array2<f64>)> {
    let (headers, string_data) = read_csv(path, config)?;

    let shape = string_data.shape();
    let mut numeric_data = Array2::<f64>::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let value = string_data[[i, j]].parse::<f64>().map_err(|_| {
                IoError::FormatError(format!(
                    "Could not convert value '{}' at position [{}, {}] to number",
                    string_data[[i, j]],
                    i,
                    j
                ))
            })?;
            numeric_data[[i, j]] = value;
        }
    }

    Ok((headers, numeric_data))
}

/// CSV writer configuration
#[derive(Debug, Clone)]
pub struct CsvWriterConfig {
    /// Delimiter character (default: ',')
    pub delimiter: char,
    /// Quote character (default: '"')
    pub quote_char: char,
    /// Always quote fields (default: false)
    pub always_quote: bool,
    /// Quote fields containing special characters (default: true)
    pub quote_special: bool,
    /// Write header row (default: true)
    pub write_header: bool,
    /// Line ending (default: LF)
    pub line_ending: LineEnding,
}

impl Default for CsvWriterConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            always_quote: false,
            quote_special: true,
            write_header: true,
            line_ending: LineEnding::default(),
        }
    }
}

/// Line ending options for CSV files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineEnding {
    /// LF (Unix) line ending: \n
    #[default]
    LF,
    /// CRLF (Windows) line ending: \r\n
    CRLF,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::LF => "\n",
            LineEnding::CRLF => "\r\n",
        }
    }
}

/// Types of missing values that can be recognized in CSV files
#[derive(Debug, Clone)]
pub struct MissingValueOptions {
    /// Strings to interpret as missing values (default: NA, N/A, NaN, null, "")
    pub values: Vec<String>,
    /// Replace missing values with a default value (default: None)
    pub fill_value: Option<f64>,
}

impl Default for MissingValueOptions {
    fn default() -> Self {
        Self {
            values: vec![
                "NA".to_string(),
                "N/A".to_string(),
                "NaN".to_string(),
                "null".to_string(),
                "".to_string(),
            ],
            fill_value: None,
        }
    }
}

/// Column data type specification for type conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// String type (default)
    String,
    /// Integer type (i64)
    Integer,
    /// Float type (f64)
    Float,
    /// Boolean type (true/false, yes/no, 1/0)
    Boolean,
    /// Date type (YYYY-MM-DD)
    Date,
    /// Time type (HH:MM:SS)
    Time,
    /// DateTime type (YYYY-MM-DDThh:mm:ss)
    DateTime,
    /// Complex number (real+imagi)
    Complex,
}

/// Column specification for CSV reading
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    /// Column index
    pub index: usize,
    /// Column name (optional)
    pub name: Option<String>,
    /// Column data type
    pub dtype: ColumnType,
    /// Custom missing values for this column
    pub missing_values: Option<MissingValueOptions>,
}

/// Data value type for mixed type columns
#[derive(Debug, Clone)]
pub enum DataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date value (year, month, day)
    Date(NaiveDate),
    /// Time value (hour, minute, second)
    Time(NaiveTime),
    /// DateTime value
    DateTime(NaiveDateTime),
    /// Complex number value (real, imaginary)
    Complex(Complex64),
    /// Missing value
    Missing,
}

impl std::fmt::Display for DataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataValue::String(s) => write!(f, "{}", s),
            DataValue::Integer(i) => write!(f, "{}", i),
            DataValue::Float(v) => write!(f, "{}", v),
            DataValue::Boolean(b) => write!(f, "{}", b),
            DataValue::Date(d) => write!(f, "{}", d.format("%Y-%m-%d")),
            DataValue::Time(t) => write!(f, "{}", t.format("%H:%M:%S%.f")),
            DataValue::DateTime(dt) => write!(f, "{}", dt.format("%Y-%m-%dT%H:%M:%S%.f")),
            DataValue::Complex(c) => {
                if c.im >= 0.0 {
                    write!(f, "{}+{}i", c.re, c.im)
                } else {
                    write!(f, "{}{}i", c.re, c.im)
                }
            }
            DataValue::Missing => write!(f, "NA"),
        }
    }
}

/// Automatically detect column types from data
pub fn detect_column_types(data: &Array2<String>) -> Vec<ColumnType> {
    let (rows, cols) = (data.shape()[0], data.shape()[1]);

    // Default to String if we can't determine type
    if rows == 0 {
        return vec![ColumnType::String; cols];
    }

    let mut col_types = vec![ColumnType::String; cols];

    for col in 0..cols {
        let mut is_int = true;
        let mut is_float = true;
        let mut is_bool = true;
        let mut is_date = true;
        let mut is_time = true;
        let mut is_datetime = true;
        let mut is_complex = true;
        let mut non_empty_rows = 0;

        for row in 0..rows {
            let val = data[[row, col]].trim();

            // Skip empty values for type detection
            if val.is_empty() {
                continue;
            }

            non_empty_rows += 1;

            // Check if could be a boolean
            let lower_val = val.to_lowercase();
            let is_valid_bool =
                ["true", "false", "yes", "no", "1", "0"].contains(&lower_val.as_str());
            if !is_valid_bool {
                is_bool = false;
            }

            // Check if could be an integer
            if is_int && val.parse::<i64>().is_err() {
                is_int = false;
            }

            // Check if could be a float
            if is_float && val.parse::<f64>().is_err() {
                is_float = false;
            }

            // Check if could be a date (YYYY-MM-DD)
            if is_date && NaiveDate::parse_from_str(val, "%Y-%m-%d").is_err() {
                is_date = false;
            }

            // Check if could be a time (HH:MM:SS)
            if is_time
                && NaiveTime::parse_from_str(val, "%H:%M:%S").is_err()
                && NaiveTime::parse_from_str(val, "%H:%M:%S%.f").is_err()
            {
                is_time = false;
            }

            // Check if could be a datetime (YYYY-MM-DDThh:mm:ss)
            if is_datetime
                && NaiveDateTime::parse_from_str(val, "%Y-%m-%dT%H:%M:%S").is_err()
                && NaiveDateTime::parse_from_str(val, "%Y-%m-%d %H:%M:%S").is_err()
                && NaiveDateTime::parse_from_str(val, "%Y-%m-%dT%H:%M:%S%.f").is_err()
                && NaiveDateTime::parse_from_str(val, "%Y-%m-%d %H:%M:%S%.f").is_err()
            {
                is_datetime = false;
            }

            // Check if could be a complex number
            if is_complex {
                // Try to parse as complex number with patterns like "3+4i", "3-4i", etc.
                is_complex = parse_complex(val).is_some();
            }
        }

        // Don't auto-detect special types if we have too few samples
        if non_empty_rows < 2 {
            is_date = false;
            is_time = false;
            is_datetime = false;
            is_complex = false;
        }

        // Assign most specific type, with priority
        if is_bool {
            col_types[col] = ColumnType::Boolean;
        } else if is_int {
            col_types[col] = ColumnType::Integer;
        } else if is_float {
            col_types[col] = ColumnType::Float;
        } else if is_date {
            col_types[col] = ColumnType::Date;
        } else if is_time {
            col_types[col] = ColumnType::Time;
        } else if is_datetime {
            col_types[col] = ColumnType::DateTime;
        } else if is_complex {
            col_types[col] = ColumnType::Complex;
        }
    }

    col_types
}

/// Parse a complex number from string like "3+4i", "-1-2i"
fn parse_complex(s: &str) -> Option<Complex64> {
    // Common complex number formats:
    // 1. "a+bi" or "a-bi" - standard form
    // 2. "(a,b)" - coordinate form

    if s.contains('i') {
        // Handle standard form
        let s = s.trim().replace(" ", "");

        // Remove trailing 'i'
        let s = if s.ends_with('i') {
            &s[0..s.len() - 1]
        } else {
            return None;
        };

        // Find the position of + or - that isn't at the start
        let mut split_pos = None;
        let mut in_first_number = true;

        for (i, c) in s.chars().enumerate() {
            if i == 0 {
                continue; // Skip first character which might be a sign
            }

            if c == '+' || c == '-' {
                split_pos = Some((i, c));
                break;
            }

            if !c.is_ascii_digit()
                && c != '.'
                && c != 'e'
                && c != 'E'
                && !(c == '-' && (s.as_bytes()[i - 1] == b'e' || s.as_bytes()[i - 1] == b'E'))
            {
                in_first_number = false;
            }
        }

        if let Some((pos, sign)) = split_pos {
            let real_part = s[0..pos].parse::<f64>().ok()?;
            let imag_part = if sign == '+' {
                s[pos + 1..].parse::<f64>().ok()?
            } else {
                -s[pos + 1..].parse::<f64>().ok()?
            };

            Some(Complex64::new(real_part, imag_part))
        } else if in_first_number {
            // Form like "0i" (just imaginary part)
            Some(Complex64::new(0.0, s.parse::<f64>().ok()?))
        } else {
            None
        }
    } else if s.starts_with('(') && s.ends_with(')') && s.contains(',') {
        // Handle coordinate form (a,b)
        let contents = &s[1..s.len() - 1];
        let parts: Vec<&str> = contents.split(',').collect();

        if parts.len() == 2 {
            let real = parts[0].trim().parse::<f64>().ok()?;
            let imag = parts[1].trim().parse::<f64>().ok()?;
            Some(Complex64::new(real, imag))
        } else {
            None
        }
    } else {
        None
    }
}

/// Convert a string to a specified type with missing value handling
fn convert_value(
    value: &str,
    col_type: ColumnType,
    missing_values: &MissingValueOptions,
) -> Result<DataValue> {
    let trimmed = value.trim();

    // Check for missing values
    if missing_values
        .values
        .iter()
        .any(|mv| mv.eq_ignore_ascii_case(trimmed))
    {
        if let (Some(fill), ColumnType::Float) = (missing_values.fill_value, col_type) {
            return Ok(DataValue::Float(fill));
        }
        return Ok(DataValue::Missing);
    }

    // Empty string check
    if trimmed.is_empty() {
        return Ok(DataValue::Missing);
    }

    // Type conversion
    match col_type {
        ColumnType::String => Ok(DataValue::String(trimmed.to_string())),
        ColumnType::Integer => match trimmed.parse::<i64>() {
            Ok(val) => Ok(DataValue::Integer(val)),
            Err(_) => Err(IoError::FormatError(format!(
                "Cannot convert '{}' to integer",
                value
            ))),
        },
        ColumnType::Float => match trimmed.parse::<f64>() {
            Ok(val) => Ok(DataValue::Float(val)),
            Err(_) => Err(IoError::FormatError(format!(
                "Cannot convert '{}' to float",
                value
            ))),
        },
        ColumnType::Boolean => {
            let lower = trimmed.to_lowercase();
            match lower.as_str() {
                "true" | "yes" | "1" => Ok(DataValue::Boolean(true)),
                "false" | "no" | "0" => Ok(DataValue::Boolean(false)),
                _ => Err(IoError::FormatError(format!(
                    "Cannot convert '{}' to boolean",
                    value
                ))),
            }
        }
        ColumnType::Date => match NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
            Ok(date) => Ok(DataValue::Date(date)),
            Err(_) => Err(IoError::FormatError(format!(
                "Cannot convert '{}' to date (expected YYYY-MM-DD)",
                value
            ))),
        },
        ColumnType::Time => {
            let result = NaiveTime::parse_from_str(trimmed, "%H:%M:%S")
                .or_else(|_| NaiveTime::parse_from_str(trimmed, "%H:%M:%S%.f"));

            match result {
                Ok(time) => Ok(DataValue::Time(time)),
                Err(_) => Err(IoError::FormatError(format!(
                    "Cannot convert '{}' to time (expected HH:MM:SS[.f])",
                    value
                ))),
            }
        }
        ColumnType::DateTime => {
            let result = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S")
                .or_else(|_| NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S"))
                .or_else(|_| NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S%.f"))
                .or_else(|_| NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f"));

            match result {
                Ok(dt) => Ok(DataValue::DateTime(dt)),
                Err(_) => Err(IoError::FormatError(format!(
                    "Cannot convert '{}' to datetime (expected YYYY-MM-DD[T ]HH:MM:SS[.f])",
                    value
                ))),
            }
        }
        ColumnType::Complex => match parse_complex(trimmed) {
            Some(complex) => Ok(DataValue::Complex(complex)),
            None => Err(IoError::FormatError(format!(
                "Cannot convert '{}' to complex number (expected a+bi or (a,b))",
                value
            ))),
        },
    }
}

/// Write a 2D array to a CSV file
///
/// # Arguments
///
/// * `path` - Path to the output CSV file
/// * `data` - 2D array to write
/// * `headers` - Optional column headers
/// * `config` - Optional CSV writer configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_io::csv::{write_csv, CsvWriterConfig};
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let headers = vec!["A".to_string(), "B".to_string(), "C".to_string()];
///
/// // Write with default configuration
/// write_csv("output.csv", &data, Some(&headers), None).unwrap();
///
/// // Write with custom configuration
/// let config = CsvWriterConfig {
///     delimiter: ';',
///     always_quote: true,
///     ..Default::default()
/// };
/// write_csv("output_custom.csv", &data, Some(&headers), Some(config)).unwrap();
/// ```
pub fn write_csv<P: AsRef<Path>, T: std::fmt::Display>(
    path: P,
    data: &Array2<T>,
    headers: Option<&Vec<String>>,
    config: Option<CsvWriterConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();

    // Get data shape
    let shape = data.shape();
    let (rows, cols) = (shape[0], shape[1]);

    // Check headers match data width
    if let Some(hdrs) = headers {
        if hdrs.len() != cols && config.write_header {
            return Err(IoError::FormatError(format!(
                "Header length ({}) does not match data width ({})",
                hdrs.len(),
                cols
            )));
        }
    }

    // Open file for writing
    let mut file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;

    // Write headers if provided and enabled
    if let Some(hdrs) = headers {
        if config.write_header {
            let header_line = format_csv_line(hdrs, &config);
            file.write_all(header_line.as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
            file.write_all(config.line_ending.as_str().as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    // Write data rows
    for i in 0..rows {
        let row: Vec<String> = (0..cols).map(|j| data[[i, j]].to_string()).collect();

        let line = format_csv_line(&row, &config);
        file.write_all(line.as_bytes())
            .map_err(|e| IoError::FileError(e.to_string()))?;

        if i < rows - 1 || config.line_ending == LineEnding::CRLF {
            file.write_all(config.line_ending.as_str().as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
        } else {
            // For LF, ensure file ends with a newline but avoid extra newline
            file.write_all(b"\n")
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    Ok(())
}

/// Format a row as a CSV line
fn format_csv_line(fields: &[String], config: &CsvWriterConfig) -> String {
    let mut result = String::new();

    for (i, field) in fields.iter().enumerate() {
        let need_quotes = config.always_quote
            || (config.quote_special
                && (field.contains(config.delimiter)
                    || field.contains(config.quote_char)
                    || field.contains('\n')
                    || field.contains('\r')));

        if need_quotes {
            // Add quote character
            result.push(config.quote_char);

            // Add the field with escaped quotes
            let escaped = field.replace(
                config.quote_char,
                &format!("{}{}", config.quote_char, config.quote_char),
            );
            result.push_str(&escaped);

            // Close quotes
            result.push(config.quote_char);
        } else {
            result.push_str(field);
        }

        // Add delimiter if not the last field
        if i < fields.len() - 1 {
            result.push(config.delimiter);
        }
    }

    result
}

/// Read a CSV file with type conversion and missing value handling
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `config` - Optional CSV reader configuration
/// * `col_types` - Optional column data types
/// * `missing_values` - Optional missing value handling options
///
/// # Returns
///
/// * `Result<(Vec<String>, Vec<Vec<DataValue>>)>` - Headers and typed data values
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::csv::{read_csv_typed, ColumnType, CsvReaderConfig, MissingValueOptions};
///
/// // Read with automatic type detection
/// let (headers, data) = read_csv_typed("data.csv", None, None, None).unwrap();
///
/// // Read with specified column types
/// let col_types = vec![
///     ColumnType::String,
///     ColumnType::Integer,
///     ColumnType::Float,
///     ColumnType::Boolean,
/// ];
/// let (headers, data) = read_csv_typed("data.csv", None, Some(&col_types), None).unwrap();
///
/// // Read with custom missing value handling
/// let missing_opts = MissingValueOptions {
///     values: vec!["missing".to_string(), "unknown".to_string()],
///     fill_value: Some(0.0),
/// };
/// let (headers, data) = read_csv_typed("data.csv", None, None, Some(missing_opts)).unwrap();
/// ```
pub fn read_csv_typed<P: AsRef<Path>>(
    path: P,
    config: Option<CsvReaderConfig>,
    col_types: Option<&[ColumnType]>,
    missing_values: Option<MissingValueOptions>,
) -> Result<(Vec<String>, Vec<Vec<DataValue>>)> {
    // Get string data first
    let (headers, string_data) = read_csv(path, config)?;

    // If no data, return early
    if string_data.shape()[0] == 0 || string_data.shape()[1] == 0 {
        return Ok((headers, Vec::new()));
    }

    // Determine column types if not provided
    let types = match col_types {
        Some(types) => {
            if types.len() != string_data.shape()[1] {
                return Err(IoError::FormatError(format!(
                    "Number of column types ({}) does not match data width ({})",
                    types.len(),
                    string_data.shape()[1]
                )));
            }
            types.to_vec()
        }
        None => detect_column_types(&string_data),
    };

    let missing_opts = missing_values.unwrap_or_default();

    // Convert data
    let mut typed_data = Vec::with_capacity(string_data.shape()[0]);

    for i in 0..string_data.shape()[0] {
        let mut row = Vec::with_capacity(string_data.shape()[1]);

        for j in 0..string_data.shape()[1] {
            let value = convert_value(&string_data[[i, j]], types[j], &missing_opts)?;
            row.push(value);
        }

        typed_data.push(row);
    }

    Ok((headers, typed_data))
}

/// Read a CSV file in chunks to process large files memory-efficiently
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `config` - Optional CSV reader configuration
/// * `chunk_size` - Number of rows to read in each chunk
/// * `callback` - Function to process each chunk
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::csv::{read_csv_chunked, CsvReaderConfig};
/// use ndarray::Array2;
///
/// let config = CsvReaderConfig::default();
/// let mut total_rows = 0;
///
/// read_csv_chunked("large_data.csv", Some(config), 1000, |headers, chunk| {
///     println!("Processing chunk with {} rows", chunk.shape()[0]);
///     total_rows += chunk.shape()[0];
///     true // continue processing
/// }).unwrap();
///
/// println!("Total rows processed: {}", total_rows);
/// ```
pub fn read_csv_chunked<P, F>(
    path: P,
    config: Option<CsvReaderConfig>,
    chunk_size: usize,
    mut callback: F,
) -> Result<()>
where
    P: AsRef<Path>,
    F: FnMut(&[String], &Array2<String>) -> bool,
{
    let config = config.unwrap_or_default();

    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip rows if needed
    for _ in 0..config.skip_rows {
        if lines.next().is_none() {
            return Err(IoError::FormatError("Not enough rows in file".to_string()));
        }
    }

    // Read header if present
    let headers = if config.has_header {
        match lines.next() {
            Some(Ok(line)) => parse_csv_line(&line, &config),
            Some(Err(e)) => return Err(IoError::FileError(e.to_string())),
            None => return Err(IoError::FormatError("Empty file".to_string())),
        }
    } else {
        Vec::new()
    };

    let mut buffer = Vec::with_capacity(chunk_size);
    let mut num_cols = 0;

    // Process file in chunks
    for line_result in lines {
        // Process comment lines, empty lines
        let line = line_result.map_err(|e| IoError::FileError(e.to_string()))?;

        if let Some(comment_char) = config.comment_char {
            if line.trim().starts_with(comment_char) {
                continue;
            }
        }

        if line.trim().is_empty() {
            continue;
        }

        // Parse the line
        let row = parse_csv_line(&line, &config);

        // Determine number of columns from first data row
        if buffer.is_empty() {
            num_cols = row.len();
        } else if row.len() != num_cols {
            return Err(IoError::FormatError(format!(
                "Inconsistent number of columns: got {}, expected {}",
                row.len(),
                num_cols
            )));
        }

        buffer.push(row);

        // Process chunk when we've reached chunk_size
        if buffer.len() >= chunk_size
            && !process_chunk(&headers, &mut buffer, num_cols, &mut callback)?
        {
            return Ok(()); // Callback returned false, stop processing
        }
    }

    // Process remaining rows
    if !buffer.is_empty() {
        process_chunk(&headers, &mut buffer, num_cols, &mut callback)?;
    }

    Ok(())
}

/// Helper function to process a chunk of data
fn process_chunk<F>(
    headers: &[String],
    buffer: &mut Vec<Vec<String>>,
    num_cols: usize,
    callback: &mut F,
) -> Result<bool>
where
    F: FnMut(&[String], &Array2<String>) -> bool,
{
    let num_rows = buffer.len();
    let mut data = Array2::<String>::from_elem((num_rows, num_cols), String::new());

    for (i, row) in buffer.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            data[[i, j]] = value.clone();
        }
    }

    buffer.clear();

    Ok(callback(headers, &data))
}

/// Write typed data to a CSV file
///
/// # Arguments
///
/// * `path` - Path to the output CSV file
/// * `data` - Vector of vectors containing typed data values
/// * `headers` - Optional column headers
/// * `config` - Optional CSV writer configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::csv::{write_csv_typed, DataValue, CsvWriterConfig};
///
/// // Create mixed-type data
/// let row1 = vec![
///     DataValue::String("Alice".to_string()),
///     DataValue::Integer(25),
///     DataValue::Float(168.5),
///     DataValue::Boolean(true),
/// ];
/// let row2 = vec![
///     DataValue::String("Bob".to_string()),
///     DataValue::Integer(32),
///     DataValue::Float(175.0),
///     DataValue::Boolean(false),
/// ];
///
/// let data = vec![row1, row2];
/// let headers = vec![
///     "Name".to_string(),
///     "Age".to_string(),
///     "Height".to_string(),
///     "Active".to_string(),
/// ];
///
/// write_csv_typed("typed_data.csv", &data, Some(&headers), None).unwrap();
/// ```
pub fn write_csv_typed<P: AsRef<Path>>(
    path: P,
    data: &[Vec<DataValue>],
    headers: Option<&Vec<String>>,
    config: Option<CsvWriterConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();

    if data.is_empty() {
        return Err(IoError::FormatError("No data provided".to_string()));
    }

    // Check all rows have the same length
    let num_cols = data[0].len();
    for (i, row) in data.iter().enumerate().skip(1) {
        if row.len() != num_cols {
            return Err(IoError::FormatError(format!(
                "Row {} has {} columns, expected {}",
                i,
                row.len(),
                num_cols
            )));
        }
    }

    // Check headers match data width
    if let Some(hdrs) = headers {
        if hdrs.len() != num_cols && config.write_header {
            return Err(IoError::FormatError(format!(
                "Header length ({}) does not match data width ({})",
                hdrs.len(),
                num_cols
            )));
        }
    }

    // Open file for writing with buffering for better performance
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write headers if provided and enabled
    if let Some(hdrs) = headers {
        if config.write_header {
            let header_line = format_csv_line(hdrs, &config);
            writer
                .write_all(header_line.as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
            writer
                .write_all(config.line_ending.as_str().as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    // Write data rows
    for (i, row) in data.iter().enumerate() {
        let string_row: Vec<String> = row.iter().map(|val| val.to_string()).collect();

        let line = format_csv_line(&string_row, &config);
        writer
            .write_all(line.as_bytes())
            .map_err(|e| IoError::FileError(e.to_string()))?;

        if i < data.len() - 1 || config.line_ending == LineEnding::CRLF {
            writer
                .write_all(config.line_ending.as_str().as_bytes())
                .map_err(|e| IoError::FileError(e.to_string()))?;
        } else {
            // For LF, ensure file ends with a newline but avoid extra newline
            writer
                .write_all(b"\n")
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    // Ensure data is written to disk
    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

/// Write multiple 1D arrays to a CSV file as columns
///
/// # Arguments
///
/// * `path` - Path to the output CSV file
/// * `columns` - Vector of 1D arrays to write as columns
/// * `headers` - Optional column headers
/// * `config` - Optional CSV writer configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array1, array};
/// use scirs2_io::csv::{write_csv_columns, CsvWriterConfig};
///
/// let col1 = array![1.0, 2.0, 3.0];
/// let col2 = array![4.0, 5.0, 6.0];
/// let columns = vec![col1, col2];
/// let headers = vec!["X".to_string(), "Y".to_string()];
///
/// write_csv_columns("columns.csv", &columns, Some(&headers), None).unwrap();
/// ```
pub fn write_csv_columns<P: AsRef<Path>, T: std::fmt::Display + Clone>(
    path: P,
    columns: &[Array1<T>],
    headers: Option<&Vec<String>>,
    config: Option<CsvWriterConfig>,
) -> Result<()> {
    if columns.is_empty() {
        return Err(IoError::FormatError("No columns provided".to_string()));
    }

    // Check all columns have the same length
    let num_rows = columns[0].len();
    for (i, col) in columns.iter().enumerate().skip(1) {
        if col.len() != num_rows {
            return Err(IoError::FormatError(format!(
                "Column {} has length {}, expected {}",
                i,
                col.len(),
                num_rows
            )));
        }
    }

    // Check headers match column count
    if let Some(hdrs) = headers {
        if hdrs.len() != columns.len() {
            return Err(IoError::FormatError(format!(
                "Header length ({}) does not match column count ({})",
                hdrs.len(),
                columns.len()
            )));
        }
    }

    // Convert to Array2
    let num_cols = columns.len();
    let mut data = Array2::<String>::from_elem((num_rows, num_cols), String::new());

    for (j, col) in columns.iter().enumerate() {
        for (i, val) in col.iter().enumerate() {
            data[[i, j]] = val.to_string();
        }
    }

    // Write to CSV
    write_csv(path, &data, headers, config)
}
