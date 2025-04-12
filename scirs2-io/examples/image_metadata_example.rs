use ndarray::Array3;
use scirs2_io::image::{read_image, write_image, ImageFormat, ImageMetadata};
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Image Metadata Example ===\n");

    // Create a simple test image
    println!("Creating a test image with metadata...");
    let mut test_img = Array3::zeros((50, 50, 3));

    // Create a colorful pattern
    for y in 0..50 {
        for x in 0..50 {
            test_img[[y, x, 0]] = ((x as f32 / 50.0) * 255.0) as u8; // Red increases with x
            test_img[[y, x, 1]] = ((y as f32 / 50.0) * 255.0) as u8; // Green increases with y
            test_img[[y, x, 2]] = (((x + y) as f32 / 100.0) * 255.0) as u8; // Blue increases with x+y
        }
    }

    // Create custom metadata
    let mut custom_metadata = HashMap::new();
    custom_metadata.insert("title".to_string(), "Test Image".to_string());
    custom_metadata.insert("author".to_string(), "SciRS2 IO Module".to_string());
    custom_metadata.insert(
        "description".to_string(),
        "A test image with custom metadata".to_string(),
    );
    custom_metadata.insert("created".to_string(), "2025-04-10".to_string());

    let metadata = ImageMetadata {
        width: 50,
        height: 50,
        color_mode: None, // Will be determined from the image data
        bits_per_channel: Some(8),
        dpi: Some((300, 300)),
        format: Some("png".to_string()),
        custom: custom_metadata,
    };

    // Write the image with metadata
    println!("Writing image with metadata...");
    write_image(
        "scirs2-io/examples/metadata_test.png",
        &test_img,
        Some(ImageFormat::PNG),
        Some(&metadata),
    )?;

    // Read the image back
    println!("\nReading image and checking metadata...");
    let (_, read_metadata) = read_image("scirs2-io/examples/metadata_test.png", None)?;

    println!("Image Metadata:");
    println!(
        "  Dimensions: {}x{}",
        read_metadata.width, read_metadata.height
    );
    println!("  Format: {:?}", read_metadata.format);
    println!("  Color Mode: {:?}", read_metadata.color_mode);
    println!("  Bits Per Channel: {:?}", read_metadata.bits_per_channel);

    // Note: Most custom metadata is not preserved in standard image formats
    // This is a limitation of the underlying image formats and libraries
    println!("\nCustom Metadata (may not be preserved in the image file):");
    for (key, value) in metadata.custom.iter() {
        println!("  {}: {}", key, value);
    }

    // Example of extracting EXIF data from JPEG images
    println!(
        "\nNote: For real-world images, EXIF data can be extracted from JPEG and other formats."
    );
    println!("This example doesn't include EXIF extraction for simplicity.");

    println!("\nImage metadata example completed successfully!");
    Ok(())
}
