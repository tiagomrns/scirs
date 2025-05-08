use ndarray::Array4;
use scirs2_io::image::{
    create_animated_gif, extract_gif_frames, read_image_sequence, write_image_sequence,
    AnimationMetadata, ImageFormat,
};
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    // Create output directories
    fs::create_dir_all("examples/animation/frames")?;

    println!("Creating a simple animation...");

    // Create a simple animation with 10 frames, 50x50 pixels, RGB
    let frames = 10;
    let size = 50;
    let mut animation = Array4::zeros((frames, size, size, 3));

    // Create a moving colored ball
    for frame in 0..frames {
        // Position of the ball, moves from left to right
        let x = (frame * (size - 10) / (frames - 1)) + 5;

        // Draw a colored ball (circle)
        for y in 0..size {
            for px in 0..size {
                // Distance from center of the ball
                let dx = (px as isize - x as isize).abs() as usize;
                let dy = (y as isize - size as isize / 2).abs() as usize;
                let dist = (dx * dx + dy * dy) as f32;
                let radius = 8.0;

                if dist < radius * radius {
                    // Inside the ball - rainbow coloring that changes per frame
                    let r = (frame * 255 / frames) as u8;
                    let g = ((frames - frame) * 255 / frames) as u8;
                    let b = ((frame + frames / 2) % frames * 255 / frames) as u8;

                    animation[[frame, y, px, 0]] = r;
                    animation[[frame, y, px, 1]] = g;
                    animation[[frame, y, px, 2]] = b;
                } else if dx < 1 || dy < 1 || px == size - 1 || y == size - 1 {
                    // Border of the image - white
                    animation[[frame, y, px, 0]] = 255;
                    animation[[frame, y, px, 1]] = 255;
                    animation[[frame, y, px, 2]] = 255;
                }
            }
        }
    }

    // Create metadata for animation
    let metadata = AnimationMetadata {
        frame_count: frames,
        width: size,
        height: size,
        frame_delays: vec![100; frames], // 100ms per frame
        loop_forever: true,
        ..Default::default()
    };

    // Create an animated GIF
    println!("Saving as GIF...");
    let gif_path = "examples/animation/animation.gif";
    create_animated_gif(gif_path, &animation, &metadata)?;
    println!("GIF created at: {}", gif_path);

    // Extract frames from the GIF we just created
    println!("Extracting frames from GIF...");
    let output_pattern = "examples/animation/frames/extracted_{:04d}.png";
    let extracted_paths = extract_gif_frames(gif_path, output_pattern, Some(ImageFormat::PNG))?;
    println!("Extracted {} frames from GIF to PNG", extracted_paths.len());

    // Save frames as a sequence of individual PNG files
    println!("Saving frames as individual PNG files...");
    let frames_pattern = "examples/animation/frames/frame_{:04d}.png";
    let frame_paths = write_image_sequence(
        frames_pattern,
        &animation,
        Some(ImageFormat::PNG),
        Some(&metadata),
    )?;
    println!("Saved {} individual frame files", frame_paths.len());

    // Read the frames back as a sequence
    println!("Reading frames back as a sequence...");
    let (read_frames, read_metadata) = read_image_sequence(frames_pattern, None)?;
    println!(
        "Read {} frames of {}x{}",
        read_metadata.frame_count, read_metadata.width, read_metadata.height
    );

    // Create a new animation with the read frames, but reverse it
    println!("Creating a reversed animation...");
    let reversed_metadata = read_metadata.clone();

    // Create a 4D array for the reversed animation
    let num_frames = read_frames.shape()[0];
    let height = read_frames.shape()[1];
    let width = read_frames.shape()[2];
    let channels = read_frames.shape()[3];

    let mut reversed_frames = Array4::zeros((num_frames, height, width, channels));

    // Reverse the frame order
    for i in 0..num_frames {
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    reversed_frames[[i, y, x, c]] = read_frames[[num_frames - 1 - i, y, x, c]];
                }
            }
        }
    }

    // Save the reversed animation
    println!("Saving reversed animation...");
    create_animated_gif(
        "examples/animation/reversed.gif",
        &reversed_frames,
        &reversed_metadata,
    )?;
    println!("Reversed animation saved as examples/animation/reversed.gif");

    println!("Example completed successfully!");

    Ok(())
}
