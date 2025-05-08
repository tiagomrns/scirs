use scirs2_io::csv::{read_csv_typed, write_csv_typed, ColumnType, CsvWriterConfig};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Reading CSV file with advanced data types...");

    // Read the CSV file with automatic type detection
    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/simple_types.csv",
        None,
        None,
        None,
    )?;

    println!("Headers: {:?}", headers);
    println!("Detected types:");

    // Display the detected types and values for each row
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Read the CSV file with specified types
    let col_types = vec![
        ColumnType::String,
        ColumnType::Integer,
        ColumnType::Float,
        ColumnType::Boolean,
    ];

    println!("\nReading with explicit types...");

    let (headers, data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/simple_types.csv",
        None,
        Some(&col_types),
        None,
    )?;

    println!("Headers: {:?}", headers);

    // Display the values for each row
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Write the data back to a new CSV file
    println!("\nWriting data back to CSV...");

    let writer_config = CsvWriterConfig {
        delimiter: ',',
        quote_char: '"',
        always_quote: true,
        ..Default::default()
    };

    write_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/advanced_types_output.csv",
        &data,
        Some(&headers),
        Some(writer_config),
    )?;

    println!(
        "Data written to /media/kitasan/Backup/scirs/scirs2-io/examples/advanced_types_output.csv"
    );

    println!("\nReading back the written file for verification...");

    let (output_headers, output_data) = read_csv_typed(
        "/media/kitasan/Backup/scirs/scirs2-io/examples/advanced_types_output.csv",
        None,
        Some(&col_types),
        None,
    )?;

    println!("Output headers: {:?}", output_headers);

    // Display the values for each row
    for (i, row) in output_data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    println!("\nExample completed successfully!");

    Ok(())
}
