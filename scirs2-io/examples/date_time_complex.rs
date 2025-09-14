use scirs2_io::csv::{read_csv_typed, write_csv_typed, ColumnType};
use std::error::Error;
use std::fs::File;
use std::io::Write;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Create a CSV file with the advanced data types
    println!("Creating CSV files with date, time, and complex number data types...");

    let date_time_csv = r#"Date,Time,DateTime
2023-01-15,13:45:30,2023-01-15T13:45:30
2023-02-20,08:15:00,2023-02-20T08:15:00
2023-03-25,18:30:45.123,2023-03-25T18:30:45.123"#;

    let complex_csv = r#"Complex
"3+4i"
"-1.5-2.7i"
"(1.2,3.4)"
"0i""#;

    // Write test CSV files
    let mut date_time_file =
        File::create("/media/kitasan/Backup/scirs/scirs2-io/examples/date_time.csv")?;
    date_time_file.write_all(date_time_csv.as_bytes())?;

    let mut complex_file =
        File::create("/media/kitasan/Backup/scirs/scirs2-io/examples/complex.csv")?;
    complex_file.write_all(complex_csv.as_bytes())?;

    // Read date and time CSV file
    println!("\nReading date and time data...");

    let date_time_types = vec![ColumnType::Date, ColumnType::Time, ColumnType::DateTime];

    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/date_time.csv",
        None,
        Some(&date_time_types),
        None,
    )?;

    println!("Headers: {:?}", headers);
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Test writing date/time data back to CSV
    println!("\nWriting date and time data back to CSV...");

    let output_path = "/media/kitasan/Backup/scirs/scirs2-io/examples/date_time_output.csv";
    write_csv_typed(output_path, &data, Some(&headers), None)?;

    println!("Date/time data written to: {}", output_path);

    // Read complex number CSV file
    println!("\nReading complex number data...");

    let complex_types = vec![ColumnType::Complex];

    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/complex.csv",
        None,
        Some(&complex_types),
        None,
    )?;

    println!("Headers: {:?}", headers);
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Test writing complex data back to CSV
    println!("\nWriting complex number data back to CSV...");

    let output_path = "/media/kitasan/Backup/scirs/scirs2-io/examples/complex_output.csv";
    write_csv_typed(output_path, &data, Some(&headers), None)?;

    println!("Complex number data written to: {}", output_path);

    // Demonstrate auto type detection
    println!("\nTesting automatic type detection...");

    // Read date/time file with auto detection
    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/date_time.csv",
        None,
        None,
        None,
    )?;

    println!("Auto-detected date/time types:");
    println!("Headers: {:?}", headers);
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Read complex file with auto detection
    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/complex.csv",
        None,
        None,
        None,
    )?;

    println!("Auto-detected complex types:");
    println!("Headers: {:?}", headers);
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    println!("\nExample completed successfully!");

    Ok(())
}
