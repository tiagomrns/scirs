use ndarray::Array3;
use scirs2_io::image::{
    convert_image, get_grayscale, read_image, read_image_metadata, write_image, ImageFormat,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Image Module Example ===\n");

    // Create a simple RGB test image (100x100 pixels)
    println!("Creating a test image...");
    let mut test_img = Array3::zeros((100, 100, 3));

    // Create a red square in the top-left corner
    for y in 0..30 {
        for x in 0..30 {
            test_img[[y, x, 0]] = 255;
        }
    }

    // Create a green square in the top-right corner
    for y in 0..30 {
        for x in 70..100 {
            test_img[[y, x, 1]] = 255;
        }
    }

    // Create a blue square in the bottom-left corner
    for y in 70..100 {
        for x in 0..30 {
            test_img[[y, x, 2]] = 255;
        }
    }

    // Create a white square in the bottom-right corner
    for y in 70..100 {
        for x in 70..100 {
            test_img[[y, x, 0]] = 255;
            test_img[[y, x, 1]] = 255;
            test_img[[y, x, 2]] = 255;
        }
    }

    // Create a diagonal line
    for i in 0..100 {
        if i < test_img.shape()[0] && i < test_img.shape()[1] {
            test_img[[i, i, 0]] = 255;
            test_img[[i, i, 1]] = 255;
            test_img[[i, i, 2]] = 0;
        }
    }

    println!("Test image created with shape: {:?}", test_img.shape());

    // Write test image in different formats
    println!("\nWriting test image in different formats...");

    // Write as PNG
    println!("Writing as PNG...");
    write_image(
        "scirs2-io/examples/test_image.png",
        &test_img,
        Some(ImageFormat::PNG),
        None,
    )?;

    // Write as JPEG
    println!("Writing as JPEG...");
    write_image(
        "scirs2-io/examples/test_image.jpg",
        &test_img,
        Some(ImageFormat::JPEG),
        None,
    )?;

    // Write as BMP
    println!("Writing as BMP...");
    write_image(
        "scirs2-io/examples/test_image.bmp",
        &test_img,
        Some(ImageFormat::BMP),
        None,
    )?;

    // Read image metadata
    println!("\nReading image metadata...");
    let meta = read_image_metadata("scirs2-io/examples/test_image.png")?;
    println!("PNG Metadata:");
    println!("  Dimensions: {}x{}", meta.width, meta.height);
    println!("  Format: {:?}", meta.format);

    // Read the image
    println!("\nReading the image back...");
    let (img_data, meta) = read_image("scirs2-io/examples/test_image.png", None)?;
    println!("Read image with shape: {:?}", img_data.shape());
    println!("Color mode: {:?}", meta.color_mode);

    // Convert to grayscale
    println!("\nConverting to grayscale...");
    let gray_img = get_grayscale(&img_data);
    println!("Grayscale image shape: {:?}", gray_img.shape());

    // Create a 3D array from the grayscale image for writing
    let mut gray_3d = Array3::zeros((gray_img.shape()[0], gray_img.shape()[1], 1));
    for y in 0..gray_img.shape()[0] {
        for x in 0..gray_img.shape()[1] {
            gray_3d[[y, x, 0]] = gray_img[[y, x]];
        }
    }

    // Write grayscale image
    println!("Writing grayscale image...");
    write_image(
        "scirs2-io/examples/test_image_gray.png",
        &gray_3d,
        Some(ImageFormat::PNG),
        None,
    )?;

    // Convert between formats
    println!("\nConverting between formats...");
    convert_image(
        "scirs2-io/examples/test_image.png",
        "scirs2-io/examples/test_image_converted.jpg",
        Some(ImageFormat::JPEG),
    )?;
    println!("Converted PNG to JPEG");

    println!("\nImage module example completed successfully!");
    Ok(())
}
