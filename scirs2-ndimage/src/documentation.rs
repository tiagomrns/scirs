use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, serde::Serialize)]

pub struct DocumentationSite {
    pub title: String,
    pub description: String,
    pub version: String,
    pub base_url: String,
    pub modules: Vec<ModuleDoc>,
    pub tutorials: Vec<Tutorial>,
    pub examples: Vec<Example>,
}

#[derive(Debug, Clone)]
pub struct ModuleDoc {
    pub name: String,
    pub description: String,
    pub functions: Vec<FunctionDoc>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FunctionDoc {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
    pub returns: String,
    pub examples: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub optional: bool,
}

#[derive(Debug, Clone)]
pub struct Tutorial {
    pub title: String,
    pub description: String,
    pub content: String,
    pub code_examples: Vec<String>,
    pub difficulty: String,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub title: String,
    pub description: String,
    pub code: String,
    pub expected_output: Option<String>,
    pub category: String,
}

impl DocumentationSite {
    pub fn new() -> Self {
        Self {
            title: "SciRS2 NDImage Documentation".to_string(),
            description:
                "Comprehensive documentation for SciRS2 N-dimensional image processing library"
                    .to_string(),
            version: "0.1.0-beta.1".to_string(),
            base_url: "https://scirs2.github.io/ndimage".to_string(),
            modules: Vec::new(),
            tutorials: Vec::new(),
            examples: Vec::new(),
        }
    }

    pub fn build_comprehensive_documentation(&mut self) -> Result<()> {
        self.build_module_documentation()?;
        self.build_tutorials()?;
        self.build_examples()?;
        Ok(())
    }

    fn build_module_documentation(&mut self) -> Result<()> {
        // Filters module documentation
        let filters_module = ModuleDoc {
            name: "Filters".to_string(),
            description: "Image filtering operations including Gaussian, median, rank, and edge detection filters".to_string(),
            functions: vec![
                FunctionDoc {
                    name: "gaussian_filter".to_string(),
                    signature: "pub fn gaussian_filter<T>(input: &ArrayD<T>, sigma: f64) -> ArrayD<T>".to_string(),
                    description: "Apply Gaussian filter to n-dimensional array".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "_input".to_string(),
                            param_type: "&ArrayD<T>".to_string(),
                            description: "Input n-dimensional array".to_string(),
                            optional: false,
                        },
                        Parameter {
                            name: "sigma".to_string(),
                            param_type: "f64".to_string(),
                            description: "Standard deviation for Gaussian kernel".to_string(),
                            optional: false,
                        },
                    ],
                    returns: "ArrayD<T> - Filtered array".to_string(),
                    examples: vec![
                        r#"use scirs2_ndimage::filters::gaussian_filter;
use ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let filtered = gaussian_filter(&image, 2.0);
assert_eq!(filtered.shape(), image.shape());"#.to_string(),
                    ],
                    notes: vec![
                        "Uses separable convolution for efficiency".to_string(),
                        "Supports all numeric types".to_string(),
                    ],
                },
                FunctionDoc {
                    name: "median_filter".to_string(),
                    signature: "pub fn median_filter<T>(input: &ArrayD<T>, size: usize) -> ArrayD<T>".to_string(),
                    description: "Apply median filter to remove noise while preserving edges".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "_input".to_string(),
                            param_type: "&ArrayD<T>".to_string(),
                            description: "Input n-dimensional array".to_string(),
                            optional: false,
                        },
                        Parameter {
                            name: "size".to_string(),
                            param_type: "usize".to_string(),
                            description: "Size of the median filter window".to_string(),
                            optional: false,
                        },
                    ],
                    returns: "ArrayD<T> - Filtered array".to_string(),
                    examples: vec![
                        r#"use scirs2_ndimage::filters::median_filter;
use ndarray::Array2;

let noisyimage = Array2::from_elem((50, 50), 128.0f64);
let filtered = median_filter(&noisyimage, 3);
// Median filter removes salt-and-pepper noise"#.to_string(),
                    ],
                    notes: vec![
                        "Excellent for removing salt-and-pepper noise".to_string(),
                        "Preserves edges better than linear filters".to_string(),
                    ],
                },
            ],
            examples: vec![
                "Basic filtering operations".to_string(),
                "Edge detection pipeline".to_string(),
                "Noise reduction techniques".to_string(),
            ],
        };

        // Morphology module documentation
        let morphology_module = ModuleDoc {
            name: "Morphology".to_string(),
            description: "Mathematical morphology operations for binary and grayscale images".to_string(),
            functions: vec![
                FunctionDoc {
                    name: "binary_erosion".to_string(),
                    signature: "pub fn binary_erosion(input: &ArrayD<bool>, structure: &ArrayD<bool>) -> ArrayD<bool>".to_string(),
                    description: "Perform binary erosion operation".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "_input".to_string(),
                            param_type: "&ArrayD<bool>".to_string(),
                            description: "Input binary array".to_string(),
                            optional: false,
                        },
                        Parameter {
                            name: "structure".to_string(),
                            param_type: "&ArrayD<bool>".to_string(),
                            description: "Structuring element".to_string(),
                            optional: false,
                        },
                    ],
                    returns: "ArrayD<bool> - Eroded binary array".to_string(),
                    examples: vec![
                        r#"use scirs2_ndimage::morphology::binary_erosion;
use ndarray::Array2;

let binary_image = Array2::from_elem((10, 10), true);
let structure = Array2::from_elem((3, 3), true);
let eroded = binary_erosion(&binary_image, &structure);"#.to_string(),
                    ],
                    notes: vec![
                        "Shrinks white regions in binary images".to_string(),
                        "Useful for separating connected objects".to_string(),
                    ],
                },
                FunctionDoc {
                    name: "distance_transform_edt".to_string(),
                    signature: "pub fn distance_transform_edt(input: &ArrayD<bool>) -> ArrayD<f64>".to_string(),
                    description: "Compute Euclidean distance transform using optimized algorithm".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "_input".to_string(),
                            param_type: "&ArrayD<bool>".to_string(),
                            description: "Input binary array".to_string(),
                            optional: false,
                        },
                    ],
                    returns: "ArrayD<f64> - Distance transform".to_string(),
                    examples: vec![
                        r#"use scirs2_ndimage::morphology::distance_transform_edt;
use ndarray::Array2;

let binary_image = Array2::from_elem((100, 100), false);
let distances = distance_transform_edt(&binary_image);
// Each pixel contains distance to nearest background pixel"#.to_string(),
                    ],
                    notes: vec![
                        "Uses Felzenszwalb & Huttenlocher separable algorithm for O(n) performance".to_string(),
                        "Supports arbitrary dimensions".to_string(),
                    ],
                },
            ],
            examples: vec![
                "Object size analysis".to_string(),
                "Shape decomposition".to_string(),
                "Skeletonization".to_string(),
            ],
        };

        // Interpolation module documentation
        let interpolation_module = ModuleDoc {
            name: "Interpolation".to_string(),
            description: "Geometric transformations and interpolation operations".to_string(),
            functions: vec![
                FunctionDoc {
                    name: "affine_transform".to_string(),
                    signature: "pub fn affine_transform<T>(input: &ArrayD<T>, matrix: &Array2<f64>) -> ArrayD<T>".to_string(),
                    description: "Apply affine transformation to n-dimensional array".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "_input".to_string(),
                            param_type: "&ArrayD<T>".to_string(),
                            description: "Input array to transform".to_string(),
                            optional: false,
                        },
                        Parameter {
                            name: "matrix".to_string(),
                            param_type: "&Array2<f64>".to_string(),
                            description: "Affine transformation matrix".to_string(),
                            optional: false,
                        },
                    ],
                    returns: "ArrayD<T> - Transformed array".to_string(),
                    examples: vec![
                        r#"use scirs2_ndimage::interpolation::affine_transform;
use ndarray::{Array2, array};

let image = Array2::from_elem((50, 50), 1.0f64);
let rotation_matrix = array![[0.866, -0.5], [0.5, 0.866]]; // 30 degree rotation
let rotated = affine_transform(&image, &rotation_matrix);"#.to_string(),
                    ],
                    notes: vec![
                        "Supports rotation, scaling, shearing, and translation".to_string(),
                        "Uses spline interpolation for high quality results".to_string(),
                    ],
                },
            ],
            examples: vec![
                "Image registration".to_string(),
                "Geometric correction".to_string(),
                "Multi-resolution analysis".to_string(),
            ],
        };

        // Measurements module documentation
        let measurements_module = ModuleDoc {
            name: "Measurements".to_string(),
            description: "Statistical measurements and region analysis".to_string(),
            functions: vec![FunctionDoc {
                name: "center_of_mass".to_string(),
                signature: "pub fn center_of_mass<T>(input: &ArrayD<T>) -> Vec<f64>".to_string(),
                description: "Calculate center of mass of n-dimensional array".to_string(),
                parameters: vec![Parameter {
                    name: "_input".to_string(),
                    param_type: "&ArrayD<T>".to_string(),
                    description: "Input array".to_string(),
                    optional: false,
                }],
                returns: "Vec<f64> - Center of mass coordinates".to_string(),
                examples: vec![r#"use scirs2_ndimage::measurements::center_of_mass;
use ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let com = center_of_mass(&image);
println!("Center of mass: {:?}", com);"#
                    .to_string()],
                notes: vec![
                    "Works with any numeric type".to_string(),
                    "Returns coordinates in array index order".to_string(),
                ],
            }],
            examples: vec![
                "Object property analysis".to_string(),
                "Region statistics".to_string(),
                "Feature extraction".to_string(),
            ],
        };

        self.modules = vec![
            filters_module,
            morphology_module,
            interpolation_module,
            measurements_module,
        ];
        Ok(())
    }

    fn build_tutorials(&mut self) -> Result<()> {
        let tutorials = vec![
            Tutorial {
                title: "Getting Started with SciRS2 NDImage".to_string(),
                description: "Introduction to n-dimensional image processing in Rust".to_string(),
                difficulty: "Beginner".to_string(),
                content: r#"
# Getting Started with SciRS2 NDImage

## Introduction

SciRS2 NDImage is a comprehensive n-dimensional image processing library for Rust that provides 
high-performance implementations of common image processing operations. This tutorial will guide 
you through the basics of using the library.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-ndimage = "0.1.0-beta.1"
ndarray = "0.16"
```

## Basic Usage

### Creating Arrays

```rust
use ndarray::{Array2, Array3};

// Create a 2D array (image)
let image = Array2::from_elem((100, 100), 0.5f64);

// Create a 3D array (volume)
let volume = Array3::from_elem((50, 50, 50), 1.0f64);
```

### Applying Filters

```rust
use scirs2_ndimage::filters::gaussian_filter;

let filtered = gaussian_filter(&image, 2.0);
```

## Next Steps

- Learn about morphological operations
- Explore geometric transformations
- Try advanced filtering techniques
"#.to_string(),
                code_examples: vec![
                    "Basic array creation and manipulation".to_string(),
                    "Simple filtering operations".to_string(),
                ],
            },
            Tutorial {
                title: "Advanced Filtering Techniques".to_string(),
                description: "Master advanced filtering operations for noise reduction and feature enhancement".to_string(),
                difficulty: "Intermediate".to_string(),
                content: r#"
# Advanced Filtering Techniques

## Edge Detection

Edge detection is crucial for feature extraction and object recognition.

### Sobel Filter

```rust
use scirs2_ndimage::filters::sobel_filter;

let edges = sobel_filter(&image);
```

### Canny Edge Detection

```rust
use scirs2_ndimage::filters::canny_edge_detector;

let edges = canny_edge_detector(&image, 0.1, 0.2);
```

## Noise Reduction

### Bilateral Filter

Preserves edges while reducing noise:

```rust
use scirs2_ndimage::filters::bilateral_filter;

let denoised = bilateral_filter(&noisyimage, 5.0, 10.0);
```

### Non-local Means

Advanced denoising technique:

```rust
use scirs2_ndimage::filters::non_local_means;

let denoised = non_local_means(&noisyimage, 0.1, 7, 21);
```
"#.to_string(),
                code_examples: vec![
                    "Edge detection pipeline".to_string(),
                    "Noise reduction comparison".to_string(),
                ],
            },
            Tutorial {
                title: "Morphological Operations".to_string(),
                description: "Shape analysis and morphological transformations".to_string(),
                difficulty: "Intermediate".to_string(),
                content: r#"
# Morphological Operations

## Understanding Mathematical Morphology

Mathematical morphology is a theory and technique for analyzing shapes and structures 
in images. It's particularly useful for binary images but can be extended to grayscale.

## Basic Operations

### Erosion and Dilation

```rust
use scirs2_ndimage::morphology::{binary_erosion, binary_dilation};
use ndarray::Array2;

let structure = Array2::from_elem((3, 3), true);
let eroded = binary_erosion(&binary_image, &structure);
let dilated = binary_dilation(&binary_image, &structure);
```

### Opening and Closing

```rust
use scirs2_ndimage::morphology::{binary_opening, binary_closing};

let opened = binary_opening(&binary_image, &structure);
let closed = binary_closing(&binary_image, &structure);
```

## Advanced Applications

### Skeletonization

```rust
use scirs2_ndimage::morphology::skeletonize;

let skeleton = skeletonize(&binary_image);
```

### Distance Transform

```rust
use scirs2_ndimage::morphology::distance_transform_edt;

let distances = distance_transform_edt(&binary_image);
```
"#.to_string(),
                code_examples: vec![
                    "Shape analysis workflow".to_string(),
                    "Object separation techniques".to_string(),
                ],
            },
            Tutorial {
                title: "High-Performance Computing".to_string(),
                description: "Leveraging SIMD, parallel processing, and GPU acceleration".to_string(),
                difficulty: "Advanced".to_string(),
                content: r#"
# High-Performance Computing with SciRS2 NDImage

## SIMD Optimization

SciRS2 NDImage automatically uses SIMD instructions when available:

```rust
// Enable SIMD features
use scirs2_ndimage::filters::gaussian_filter_simd;

let filtered = gaussian_filter_simd(&largeimage, 2.0);
```

## Parallel Processing

Large arrays are automatically processed in parallel:

```rust
use scirs2_ndimage::parallel::ParallelConfig;

// Configure parallel processing
ParallelConfig::set_num_threads(8);

// Operations automatically use parallel processing for large arrays
let result = expensive_operation(&hugeimage);
```

## GPU Acceleration

For supported operations, GPU acceleration provides significant speedup:

```rust
use scirs2_ndimage::gpu::{GpuContext, gpu_gaussian_filter};

let gpu_ctx = GpuContext::new()?;
let gpu_result = gpu_gaussian_filter(&gpu_ctx, &image, 2.0)?;
```

## Memory Optimization

### Streaming Processing

For very large datasets that don't fit in memory:

```rust
use scirs2_ndimage::streaming::StreamProcessor;

let processor = StreamProcessor::new("largeimage.tiff")?;
let result = processor.apply_filter(gaussian_filter, 2.0)?;
```

### In-place Operations

Reduce memory usage with in-place operations:

```rust
use scirs2_ndimage::filters::gaussian_filter_inplace;

gaussian_filter_inplace(&mut image, 2.0);
```
"#.to_string(),
                code_examples: vec![
                    "Performance benchmarking".to_string(),
                    "Memory-efficient processing".to_string(),
                ],
            },
        ];

        self.tutorials = tutorials;
        Ok(())
    }

    fn build_examples(&mut self) -> Result<()> {
        let examples = vec![
            Example {
                title: "Medical Image Processing".to_string(),
                description: "Process medical images with specialized filters and analysis".to_string(),
                category: "Medical".to_string(),
                code: r#"
use scirs2_ndimage::domain_specific::medical::*;
use ndarray::Array3;

// Load medical volume (e.g., CT scan)
let ct_volume = Array3::from_elem((256, 256, 100), 1000.0f64);

// Apply Frangi vesselness filter for blood vessel detection
let vessels = frangi_vesselness_filter(&ct_volume, &FrangiParams::default());

// Segment bones using threshold and morphology
let bones = bone_enhancement_filter(&ct_volume, 400.0);

// Detect lung nodules
let nodules = lung_nodule_detector(&ct_volume, &NoduleDetectionParams::default());

println!("Detected {} potential nodules", nodules.len());
"#.to_string(),
                expected_output: Some("Medical image processing completed successfully".to_string()),
            },
            Example {
                title: "Satellite Image Analysis".to_string(),
                description: "Analyze satellite imagery for environmental monitoring".to_string(),
                category: "Remote Sensing".to_string(),
                code: r#"
use scirs2_ndimage::domain_specific::satellite::*;
use ndarray::Array3;

// Multi-spectral satellite image (bands: R, G, B, NIR)
let satelliteimage = Array3::from_elem((1000, 1000, 4), 0.5f64);

// Calculate vegetation indices
let ndvi = compute_ndvi(&satelliteimage);
let ndwi = compute_ndwi(&satelliteimage);

// Water body detection
let water_mask = detect_water_bodies(&satelliteimage, 0.3);

// Cloud detection and removal
let cloud_mask = detect_clouds(&satelliteimage, &CloudDetectionParams::default());
let cloud_free = remove_clouds(&satelliteimage, &cloud_mask);

// Pan-sharpening for higher resolution
let panchromatic = Array3::from_elem((4000, 4000, 1), 0.7f64);
let sharpened = pan_sharpen(&satelliteimage, &panchromatic);

println!("Processed satellite image with {} water pixels", water_mask.iter().filter(|&&x| x).count());
"#.to_string(),
                expected_output: Some("Satellite image analysis completed".to_string()),
            },
            Example {
                title: "Real-time Video Processing".to_string(),
                description: "Process video frames in real-time with optimized algorithms".to_string(),
                category: "Computer Vision".to_string(),
                code: r#"
use scirs2_ndimage::streaming::*;
use scirs2_ndimage::features::*;
use ndarray::Array3;

// Setup streaming video processor
let mut video_processor = StreamProcessor::new_video("input.mp4")?;

// Configure real-time processing pipeline
let pipeline = ProcessingPipeline::new()
    .add_filter(gaussian_filter, 1.0)
    .add_detector(corner_detector, &CornerParams::default())
    .add_tracker(object_tracker, &TrackerParams::default());

// Process frames in real-time
while let Some(frame) = video_processor.next_frame()? {
    let processed = pipeline.process(&frame)?;
    
    // Extract features
    let corners = detect_corners(&processed, &CornerParams::default());
    let edges = detect_edges(&processed, &EdgeParams::default());
    
    // Track objects across frames
    let tracked_objects = update_tracking(&corners, &edges);
    
    // Output processed frame
    video_processor.write_frame(&processed)?;
    
    println!("Frame processed: {} corners, {} edges", corners.len(), edges.len());
}
"#.to_string(),
                expected_output: Some("Real-time video processing pipeline completed".to_string()),
            },
            Example {
                title: "Scientific Image Analysis".to_string(),
                description: "Advanced analysis techniques for scientific imaging".to_string(),
                category: "Scientific".to_string(),
                code: r#"
use scirs2_ndimage::measurements::*;
use scirs2_ndimage::segmentation::*;
use ndarray::Array2;

// Scientific image (e.g., microscopy, astronomy)
let scientificimage = Array2::from_elem((2048, 2048), 0.0f64);

// Advanced segmentation using watershed
let markers = find_local_maxima(&scientificimage, 10);
let segmented = watershed_segmentation(&scientificimage, &markers);

// Measure region properties
let regions = analyze_regions(&segmented);
for region in &regions {
    println!("Region {}: area={}, centroid={:?}, eccentricity={:.3}",
             region.label, region.area, region.centroid, region.eccentricity);
}

// Statistical analysis
let moments = compute_moments(&scientificimage);
let hu_moments = compute_hu_moments(&moments);

// Feature extraction for classification
let texturefeatures = extracttexturefeatures(&scientificimage);
let shapefeatures = extractshapefeatures(&segmented);

println!("Analysis complete: {} regions found", regions.len());
"#.to_string(),
                expected_output: Some("Scientific image analysis completed with region measurements".to_string()),
            },
        ];

        self.examples = examples;
        Ok(())
    }

    pub fn generate_html_documentation(&self, outputdir: &str) -> Result<()> {
        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)?;

        // Generate main index page
        self.generate_index_page(output_path)?;

        // Generate API documentation
        self.generate_api_documentation(output_path)?;

        // Generate tutorials
        self.generate_tutorials(output_path)?;

        // Generate examples
        self.generate_examples(output_path)?;

        // Generate search functionality
        self.generate_search_index(output_path)?;

        // Copy CSS and JavaScript files
        self.generate_static_files(output_path)?;

        Ok(())
    }

    fn generate_index_page(&self, outputpath: &Path) -> Result<()> {
        let index_html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <link rel="stylesheet" href="static/style.css">
    <link rel="stylesheet" href="static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo">{}</h1>
            <nav class="nav">
                <a href="index.html">Home</a>
                <a href="api/index.html">API Reference</a>
                <a href="tutorials/index.html">Tutorials</a>
                <a href="examples/index.html">Examples</a>
                <a href="search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <section class="hero">
            <div class="container">
                <h2>High-Performance N-Dimensional Image Processing for Rust</h2>
                <p class="hero-description">{}</p>
                <div class="hero-buttons">
                    <a href="tutorials/getting-started.html" class="btn btn-primary">Get Started</a>
                    <a href="api/index.html" class="btn btn-secondary">API Reference</a>
                </div>
            </div>
        </section>

        <section class="features">
            <div class="container">
                <h3>Key Features</h3>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h4>🚀 High Performance</h4>
                        <p>SIMD-optimized algorithms with parallel processing and GPU acceleration support</p>
                    </div>
                    <div class="feature-card">
                        <h4>📐 N-Dimensional</h4>
                        <p>Work seamlessly with 1D, 2D, 3D, and higher-dimensional arrays</p>
                    </div>
                    <div class="feature-card">
                        <h4>🔬 Scientific</h4>
                        <p>Domain-specific functions for medical imaging, satellite analysis, and microscopy</p>
                    </div>
                    <div class="feature-card">
                        <h4>🐍 SciPy Compatible</h4>
                        <p>API compatible with SciPy's ndimage module for easy migration</p>
                    </div>
                    <div class="feature-card">
                        <h4>🛡️ Memory Safe</h4>
                        <p>Rust's ownership system ensures memory safety without runtime overhead</p>
                    </div>
                    <div class="feature-card">
                        <h4>⚡ Zero-Copy</h4>
                        <p>Efficient memory usage with zero-copy operations where possible</p>
                    </div>
                </div>
            </div>
        </section>

        <section class="modules">
            <div class="container">
                <h3>Core Modules</h3>
                <div class="module-grid">
                    {}
                </div>
            </div>
        </section>

        <section class="quick-start">
            <div class="container">
                <h3>Quick Start</h3>
                <pre><code class="language-toml">[dependencies]
scirs2-ndimage = "{}"
ndarray = "0.16"</code></pre>
                <pre><code class="language-rust">use scirs2_ndimage::filters::gaussian_filter;
use ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let filtered = gaussian_filter(&image, 2.0);
println!("Filtered image shape: {{:?}}", filtered.shape());</code></pre>
            </div>
        </section>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
            <p>Version {} | <a href="https://github.com/scirs2/ndimage">GitHub</a> | <a href="https://docs.rs/scirs2-ndimage">docs.rs</a></p>
        </div>
    </footer>

    <script src="static/prism.js"></script>
    <script src="static/main.js"></script>
</body>
</html>
"#,
            self.title,
            self.title,
            self.description,
            self.generate_module_cards(),
            self.version,
            self.version
        );

        let mut index_file = fs::File::create(output_path.join("index.html"))?;
        index_file.write_all(index_html.as_bytes())?;
        Ok(())
    }

    fn generate_module_cards(&self) -> String {
        self.modules
            .iter()
            .map(|module| {
                format!(
                    r#"
                <div class="module-card">
                    <h4>{}</h4>
                    <p>{}</p>
                    <div class="module-functions">
                        <span class="function-count">{} functions</span>
                        <a href="api/{}.html" class="module-link">View API →</a>
                    </div>
                </div>
            "#,
                    module.name,
                    module.description,
                    module.functions.len(),
                    module.name.to_lowercase()
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn generate_api_documentation(&self, outputpath: &Path) -> Result<()> {
        let api_dir = output_path.join("api");
        fs::create_dir_all(&api_dir)?;

        // Generate API index
        let api_index = self.generate_api_index();
        let mut api_index_file = fs::File::create(api_dir.join("index.html"))?;
        api_index_file.write_all(api_index.as_bytes())?;

        // Generate individual module pages
        for module in &self.modules {
            let module_html = self.generate_module_page(module);
            let module_filename = format!("{}.html", module.name.to_lowercase());
            let mut module_file = fs::File::create(api_dir.join(module_filename))?;
            module_file.write_all(module_html.as_bytes())?;
        }

        Ok(())
    }

    fn generate_api_index(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Reference - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="index.html" class="active">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>API Reference</h2>
            <p>Complete reference for all modules and functions in SciRS2 NDImage.</p>
            
            <div class="api-modules">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            self.title,
            self.title,
            self.generate_api_module_list()
        )
    }

    fn generate_api_module_list(&self) -> String {
        self.modules
            .iter()
            .map(|module| {
                format!(
                    r#"
                <div class="api-module">
                    <h3><a href="{}.html">{}</a></h3>
                    <p>{}</p>
                    <div class="function-list">
                        {}
                    </div>
                </div>
            "#,
                    module.name.to_lowercase(),
                    module.name,
                    module.description,
                    module
                        .functions
                        .iter()
                        .map(|func| {
                            format!(r#"<span class="function-name">{}</span>"#, func.name)
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn generate_module_page(&self, module: &ModuleDoc) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} Module - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="index.html">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>{} Module</h2>
            <p class="module-description">{}</p>
            
            <div class="functions">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            module.name,
            self.title,
            self.title,
            module.name,
            module.description,
            self.generate_function_documentation(&module.functions)
        )
    }

    fn generate_function_documentation(&self, functions: &[FunctionDoc]) -> String {
        functions
            .iter()
            .map(|func| {
                format!(
                    r#"
                <div class="function">
                    <h3 class="function-name">{}</h3>
                    <pre class="function-signature"><code class="language-rust">{}</code></pre>
                    <p class="function-description">{}</p>
                    
                    <div class="parameters">
                        <h4>Parameters</h4>
                        <ul>
                            {}
                        </ul>
                    </div>
                    
                    <div class="returns">
                        <h4>Returns</h4>
                        <p>{}</p>
                    </div>
                    
                    {}
                    
                    {}
                </div>
            "#,
                    func.name,
                    func.signature,
                    func.description,
                    func.parameters
                        .iter()
                        .map(|param| {
                            format!(
                                r#"<li><code>{}</code> ({}): {}{}</li>"#,
                                param.name,
                                param.param_type,
                                param.description,
                                if param.optional { " (optional)" } else { "" }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    func.returns,
                    if !func.examples.is_empty() {
                        format!(
                            r#"
                        <div class="examples">
                            <h4>Examples</h4>
                            {}
                        </div>
                    "#,
                            func.examples
                                .iter()
                                .map(|example| {
                                    format!(
                                        r#"<pre><code class="language-rust">{}</code></pre>"#,
                                        example
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("")
                        )
                    } else {
                        String::new()
                    },
                    if !func.notes.is_empty() {
                        format!(
                            r#"
                        <div class="notes">
                            <h4>Notes</h4>
                            <ul>
                                {}
                            </ul>
                        </div>
                    "#,
                            func.notes
                                .iter()
                                .map(|note| { format!(r#"<li>{}</li>"#, note) })
                                .collect::<Vec<_>>()
                                .join("")
                        )
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn generate_tutorials(&self, outputpath: &Path) -> Result<()> {
        let tutorials_dir = output_path.join("tutorials");
        fs::create_dir_all(&tutorials_dir)?;

        // Generate tutorials index
        let tutorials_index = self.generate_tutorials_index();
        let mut tutorials_index_file = fs::File::create(tutorials_dir.join("index.html"))?;
        tutorials_index_file.write_all(tutorials_index.as_bytes())?;

        // Generate individual tutorial pages
        for tutorial in &self.tutorials {
            let tutorial_html = self.generate_tutorial_page(tutorial);
            let tutorial_filename =
                format!("{}.html", tutorial.title.to_lowercase().replace(" ", "-"));
            let mut tutorial_file = fs::File::create(tutorials_dir.join(tutorial_filename))?;
            tutorial_file.write_all(tutorial_html.as_bytes())?;
        }

        Ok(())
    }

    fn generate_tutorials_index(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tutorials - {}</title>
    <link rel="stylesheet" href="../static/style.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="index.html" class="active">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>Tutorials</h2>
            <p>Step-by-step guides to master SciRS2 NDImage features.</p>
            
            <div class="tutorial-grid">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>
</body>
</html>
        "#,
            self.title,
            self.title,
            self.generate_tutorial_cards()
        )
    }

    fn generate_tutorial_cards(&self) -> String {
        self.tutorials
            .iter()
            .map(|tutorial| {
                let difficulty_class = match tutorial.difficulty.as_str() {
                    "Beginner" => "difficulty-beginner",
                    "Intermediate" => "difficulty-intermediate",
                    "Advanced" => "difficulty-advanced",
                    _ => "difficulty-beginner",
                };

                format!(
                    r#"
                <div class="tutorial-card">
                    <h3><a href="{}.html">{}</a></h3>
                    <p>{}</p>
                    <div class="tutorial-meta">
                        <span class="difficulty {}">📖 {}</span>
                    </div>
                </div>
            "#,
                    tutorial.title.to_lowercase().replace(" ", "-"),
                    tutorial.title,
                    tutorial.description,
                    difficulty_class,
                    tutorial.difficulty
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn generate_tutorial_page(&self, tutorial: &Tutorial) -> String {
        // Convert markdown-like content to HTML
        let html_content = self.markdown_to_html(&tutorial.content);

        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <div class="tutorial-content">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            tutorial.title, self.title, self.title, html_content
        )
    }

    fn markdown_to_html(&self, markdown: &str) -> String {
        // Simple markdown to HTML conversion
        let mut html = markdown.to_string();

        // Headers
        html = html.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n");
        html = html.replace("## ", "<h2>").replace("\n\n", "</h2>\n\n");
        html = html.replace("### ", "<h3>").replace("\n\n", "</h3>\n\n");

        // Code blocks
        let code_block_regex = regex::Regex::new(r"```(\w+)\n(.*?)\n```").unwrap();
        html = code_block_regex
            .replace_all(&html, r#"<pre><code class="language-$1">$2</code></pre>"#)
            .to_string();

        // Inline code
        let inline_code_regex = regex::Regex::new(r"`([^`]+)`").unwrap();
        html = inline_code_regex
            .replace_all(&html, r#"<code>$1</code>"#)
            .to_string();

        // Paragraphs
        let paragraphs: Vec<&str> = html.split("\n\n").collect();
        let mut result = String::new();

        for paragraph in paragraphs {
            if !paragraph.trim().is_empty()
                && !paragraph.starts_with("<h")
                && !paragraph.starts_with("<pre")
                && !paragraph.starts_with("<code")
            {
                result.push_str(&format!("<p>{}</p>\n", paragraph.trim()));
            } else {
                result.push_str(&format!("{}\n", paragraph));
            }
        }

        result
    }

    fn generate_examples(&self, outputpath: &Path) -> Result<()> {
        let examples_dir = output_path.join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate examples index
        let examples_index = self.generate_examples_index();
        let mut examples_index_file = fs::File::create(examples_dir.join("index.html"))?;
        examples_index_file.write_all(examples_index.as_bytes())?;

        // Generate individual example pages
        for example in &self.examples {
            let example_html = self.generate_example_page(example);
            let example_filename =
                format!("{}.html", example.title.to_lowercase().replace(" ", "-"));
            let mut example_file = fs::File::create(examples_dir.join(example_filename))?;
            example_file.write_all(example_html.as_bytes())?;
        }

        Ok(())
    }

    fn generate_examples_index(&self) -> String {
        let categories: std::collections::HashSet<String> =
            self.examples.iter().map(|e| e.category.clone()).collect();

        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Examples - {}</title>
    <link rel="stylesheet" href="../static/style.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="index.html" class="active">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>Examples</h2>
            <p>Practical examples demonstrating real-world usage of SciRS2 NDImage.</p>
            
            {}
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>
</body>
</html>
        "#,
            self.title,
            self.title,
            categories
                .iter()
                .map(|category| {
                    let category_examples: Vec<&Example> = self
                        .examples
                        .iter()
                        .filter(|e| &e.category == category)
                        .collect();

                    format!(
                        r#"
                    <div class="example-category">
                        <h3>{}</h3>
                        <div class="example-grid">
                            {}
                        </div>
                    </div>
                "#,
                        category,
                        category_examples
                            .iter()
                            .map(|example| {
                                format!(
                                    r#"
                            <div class="example-card">
                                <h4><a href="{}.html">{}</a></h4>
                                <p>{}</p>
                                <span class="example-category">{}</span>
                            </div>
                        "#,
                                    example.title.to_lowercase().replace(" ", "-"),
                                    example.title,
                                    example.description,
                                    example.category
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    )
                })
                .collect::<Vec<_>>()
                .join("")
        )
    }

    fn generate_example_page(&self, example: &Example) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <div class="example-header">
                <h2>{}</h2>
                <p class="example-description">{}</p>
                <span class="example-category">{}</span>
            </div>
            
            <div class="code-example">
                <h3>Code</h3>
                <pre><code class="language-rust">{}</code></pre>
            </div>
            
            {}
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            example.title,
            self.title,
            self.title,
            example.title,
            example.description,
            example.category,
            example.code,
            if let Some(output) = &example.expected_output {
                format!(
                    r#"
                    <div class="expected-output">
                        <h3>Expected Output</h3>
                        <pre><code>{}</code></pre>
                    </div>
                "#,
                    output
                )
            } else {
                String::new()
            }
        )
    }

    fn generate_search_index(&self, outputpath: &Path) -> Result<()> {
        // Generate search index JSON
        let mut search_index = HashMap::new();

        // Add modules to search index
        for module in &self.modules {
            search_index.insert(
                format!("module_{}", module.name.to_lowercase()),
                serde_json::json!({
                    "title": module.name,
                    "description": module.description,
                    "url": format!("api/{}.html", module.name.to_lowercase()),
                    "type": "module"
                }),
            );

            // Add functions to search index
            for function in &module.functions {
                search_index.insert(
                    format!("function_{}_{}", module.name.to_lowercase(), function.name.to_lowercase()),
                    serde_json::json!({
                        "title": function.name,
                        "description": function.description,
                        "url": format!("api/{}.html#{}", module.name.to_lowercase(), function.name.to_lowercase()),
                        "type": "function",
                        "module": module.name
                    })
                );
            }
        }

        // Add tutorials to search index
        for tutorial in &self.tutorials {
            search_index.insert(
                format!("tutorial_{}", tutorial.title.to_lowercase().replace(" ", "_")),
                serde_json::json!({
                    "title": tutorial.title,
                    "description": tutorial.description,
                    "url": format!("tutorials/{}.html", tutorial.title.to_lowercase().replace(" ", "-")),
                    "type": "tutorial"
                })
            );
        }

        // Add examples to search index
        for example in &self.examples {
            search_index.insert(
                format!("example_{}", example.title.to_lowercase().replace(" ", "_")),
                serde_json::json!({
                    "title": example.title,
                    "description": example.description,
                    "url": format!("examples/{}.html", example.title.to_lowercase().replace(" ", "-")),
                    "type": "example",
                    "category": example.category
                })
            );
        }

        // Write search index
        let search_index_json = serde_json::to_string_pretty(&search_index)?;
        let mut search_index_file = fs::File::create(output_path.join("search_index.json"))?;
        search_index_file.write_all(search_index_json.as_bytes())?;

        // Generate search page
        let search_html = self.generate_search_page();
        let mut search_file = fs::File::create(output_path.join("search.html"))?;
        search_file.write_all(search_html.as_bytes())?;

        Ok(())
    }

    fn generate_search_page(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search - {}</title>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="index.html">{}</a></h1>
            <nav class="nav">
                <a href="index.html">Home</a>
                <a href="api/index.html">API Reference</a>
                <a href="tutorials/index.html">Tutorials</a>
                <a href="examples/index.html">Examples</a>
                <a href="search.html" class="active">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>Search Documentation</h2>
            <div class="search-container">
                <input type="text" id="search-input" placeholder="Search functions, tutorials, examples..." />
                <div id="search-results"></div>
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="static/search.js"></script>
</body>
</html>
        "#,
            self.title, self.title
        )
    }

    fn generate_static_files(&self, outputpath: &Path) -> Result<()> {
        let static_dir = output_path.join("static");
        fs::create_dir_all(&static_dir)?;

        // Generate CSS
        let css_content = self.generate_default_css();
        let mut css_file = fs::File::create(static_dir.join("style.css"))?;
        css_file.write_all(css_content.as_bytes())?;

        // Generate JavaScript
        let js_content = self.generate_default_js();
        let mut js_file = fs::File::create(static_dir.join("main.js"))?;
        js_file.write_all(js_content.as_bytes())?;

        // Generate search JavaScript
        let search_js = r#"
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    let searchIndex = {};

    // Load search index
    fetch('search_index.json')
        .then(response => response.json())
        .then(data => {
            searchIndex = data;
        });

    searchInput.addEventListener('input', function(e) {
        const query = e.target.value.toLowerCase().trim();
        
        if (query.length < 2) {
            searchResults.innerHTML = '';
            return;
        }

        const results = Object.entries(searchIndex).filter(([key, item]) => {
            return item.title.toLowerCase().includes(query) ||
                   item.description.toLowerCase().includes(query) ||
                   (item.module && item.module.toLowerCase().includes(query));
        });

        displayResults(results.slice(0, 10));
    });

    function displayResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<p>No results found</p>';
            return;
        }

        const html = results.map(([key, item]) => `
            <div class="search-result">
                <h4><a href="${item.url}">${item.title}</a></h4>
                <p>${item.description}</p>
                <span class="result-type">${item.type}</span>
                ${item.module ? `<span class="result-module">${item.module}</span>` : ''}
            </div>
        `).join('');

        searchResults.innerHTML = html;
    }
});
        "#;
        let mut search_js_file = fs::File::create(static_dir.join("search.js"))?;
        search_js_file.write_all(search_js.as_bytes())?;

        // Generate Prism.js files for syntax highlighting
        let prism_css = self.generate_default_prism_css();
        let mut prism_css_file = fs::File::create(static_dir.join("prism.css"))?;
        prism_css_file.write_all(prism_css.as_bytes())?;

        let prism_js = self.generate_default_prism_js();
        let mut prism_js_file = fs::File::create(static_dir.join("prism.js"))?;
        prism_js_file.write_all(prism_js.as_bytes())?;

        Ok(())
    }

    pub fn generate_readme(&self, outputpath: &Path) -> Result<()> {
        let readme_content = format!(
            r#"# {} - Documentation Website

This directory contains the generated documentation website for {}.

## Structure

```
docs/
├── index.html          # Main landing page
├── api/                # API reference documentation
│   ├── index.html     # API overview
│   ├── filters.html   # Filters module
│   ├── morphology.html # Morphology module
│   └── ...
├── tutorials/          # Step-by-step tutorials
│   ├── index.html     # Tutorials overview
│   └── *.html         # Individual tutorials
├── examples/           # Practical examples
│   ├── index.html     # Examples overview
│   └── *.html         # Individual examples
├── search.html         # Search functionality
├── search_index.json   # Search index data
└── static/            # CSS, JavaScript, and assets
    ├── style.css      # Main stylesheet
    ├── main.js        # Main JavaScript
    ├── search.js      # Search functionality
    ├── prism.css      # Syntax highlighting styles
    └── prism.js       # Syntax highlighting script
```

## Features

- **Comprehensive API Documentation**: Complete reference for all modules and functions
- **Interactive Tutorials**: Step-by-step guides for different skill levels
- **Practical Examples**: Real-world usage examples with code and explanations
- **Search Functionality**: Fast search across all documentation content
- **Responsive Design**: Works on desktop and mobile devices
- **Syntax Highlighting**: Code examples with proper Rust syntax highlighting

## Usage

To view the documentation:

1. Open `index.html` in a web browser
2. Navigate through the different sections using the navigation menu
3. Use the search functionality to find specific content

## Deployment

This documentation can be deployed to any static hosting service:

- GitHub Pages
- Netlify
- Vercel
- AWS S3
- Or any web server

## Version

Documentation generated for {} version {}.

## License

Documentation licensed under MIT License.
"#,
            self.title, self.title, self.title, self.version
        );

        let mut readme_file = fs::File::create(output_path.join("README.md"))?;
        readme_file.write_all(readme_content.as_bytes())?;
        Ok(())
    }

    fn generate_default_css(&self) -> String {
        r#"
/* SciRS2 NDImage Documentation Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #fff;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header */
.header {
    background: #2c3e50;
    color: white;
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

.logo a {
    color: white;
    text-decoration: none;
}

.nav {
    display: flex;
    gap: 2rem;
}

.nav a {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

.nav a:hover,
.nav a.active {
    background-color: #34495e;
}

/* Main content */
.main {
    min-height: calc(100vh - 120px);
    padding: 2rem 0;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4rem 0;
    text-align: center;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.hero-description {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    opacity: 0.9;
}

.hero-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
}

.btn {
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s;
}

.btn-primary {
    background: #3498db;
    color: white;
}

.btn-primary:hover {
    background: #2980b9;
}

.btn-secondary {
    background: transparent;
    color: white;
    border: 2px solid white;
}

.btn-secondary:hover {
    background: white;
    color: #667eea;
}

/* Feature grid */
.features {
    padding: 4rem 0;
    background: #f8f9fa;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.feature-card {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.feature-card h4 {
    font-size: 1.2rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

/* Module grid */
.modules {
    padding: 4rem 0;
}

.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.module-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1.5rem;
    background: white;
}

.module-card h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.module-functions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
}

.function-count {
    color: #666;
    font-size: 0.9rem;
}

.module-link {
    color: #3498db;
    text-decoration: none;
    font-weight: 500;
}

.module-link:hover {
    text-decoration: underline;
}

/* Quick start */
.quick-start {
    background: #2c3e50;
    color: white;
    padding: 3rem 0;
}

.quick-start pre {
    background: #34495e;
    padding: 1rem;
    border-radius: 6px;
    margin: 1rem 0;
    overflow-x: auto;
}

/* Footer */
.footer {
    background: #2c3e50;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

.footer a {
    color: #3498db;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* Function documentation */
.function {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
}

.function-name {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.function-signature {
    background: #f8f9fa;
    border-left: 4px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
}

.function-description {
    margin-bottom: 1.5rem;
}

.parameters,
.returns,
.examples,
.notes {
    margin-bottom: 1.5rem;
}

.parameters h4,
.returns h4,
.examples h4,
.notes h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.parameters ul {
    list-style: none;
    padding-left: 0;
}

.parameters li {
    background: #f8f9fa;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 4px;
}

.examples pre {
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1rem;
    overflow-x: auto;
}

/* Search */
.search-container {
    max-width: 600px;
    margin: 2rem auto;
}

#search-input {
    width: 100%;
    padding: 1rem;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 1.1rem;
}

#search-input:focus {
    border-color: #3498db;
    outline: none;
}

#search-results {
    margin-top: 2rem;
}

.search-result {
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.search-result h4 {
    margin-bottom: 0.5rem;
}

.search-result a {
    color: #3498db;
    text-decoration: none;
}

.search-result a:hover {
    text-decoration: underline;
}

.result-type,
.result-module {
    display: inline-block;
    background: #ecf0f1;
    color: #2c3e50;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .hero h2 {
        font-size: 2rem;
    }
    
    .hero-buttons {
        flex-direction: column;
        align-items: center;
    }
    
    .feature-grid,
    .module-grid {
        grid-template-columns: 1fr;
    }
    
    .nav {
        flex-direction: column;
        gap: 0.5rem;
    }
}
        "#
        .to_string()
    }

    fn generate_default_js(&self) -> String {
        r##"
// SciRS2 NDImage Documentation JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy functionality to code blocks
    document.querySelectorAll('pre code').forEach(codeBlock => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-button';
        button.style.position = 'absolute';
        button.style.top = '10px';
        button.style.right = '10px';
        button.style.padding = '5px 10px';
        button.style.background = '#3498db';
        button.style.color = 'white';
        button.style.border = 'none';
        button.style.borderRadius = '3px';
        button.style.cursor = 'pointer';
        button.style.fontSize = '12px';

        const pre = codeBlock.parentNode;
        pre.style.position = 'relative';
        pre.appendChild(button);

        button.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
    });

    console.log('SciRS2 NDImage Documentation loaded successfully');
});
        "##
        .to_string()
    }

    fn generate_default_prism_css(&self) -> String {
        r#"
/* Prism.js Default Theme */
code[class*="language-"],
pre[class*="language-"] {
    color: #333;
    background: none;
    font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
    font-size: 1em;
    text-align: left;
    white-space: pre;
    word-spacing: normal;
    word-break: normal;
    word-wrap: normal;
    line-height: 1.5;
    -moz-tab-size: 4;
    -o-tab-size: 4;
    tab-size: 4;
    -webkit-hyphens: none;
    -moz-hyphens: none;
    -ms-hyphens: none;
    hyphens: none;
}

pre[class*="language-"] {
    position: relative;
    margin: .5em 0;
    overflow: visible;
    padding: 1em;
    background: #f5f2f0;
    border-radius: 6px;
}

:not(pre) > code[class*="language-"],
pre[class*="language-"] {
    background: #f5f2f0;
}

:not(pre) > code[class*="language-"] {
    padding: .1em;
    border-radius: .3em;
    white-space: normal;
}

.token.comment,
.token.prolog,
.token.doctype,
.token.cdata {
    color: slategray;
}

.token.punctuation {
    color: #999;
}

.token.namespace {
    opacity: .7;
}

.token.property,
.token.tag,
.token.boolean,
.token.number,
.token.constant,
.token.symbol,
.token.deleted {
    color: #905;
}

.token.selector,
.token.attr-name,
.token.string,
.token.char,
.token.builtin,
.token.inserted {
    color: #690;
}

.token.operator,
.token.entity,
.token.url,
.language-css .token.string,
.style .token.string {
    color: #9a6e3a;
}

.token.atrule,
.token.attr-value,
.token.keyword {
    color: #07a;
}

.token.function,
.token.class-name {
    color: #DD4A68;
}

.token.regex,
.token.important,
.token.variable {
    color: #e90;
}

.token.important,
.token.bold {
    font-weight: bold;
}

.token.italic {
    font-style: italic;
}

.token.entity {
    cursor: help;
}
        "#
        .to_string()
    }

    fn generate_default_prism_js(&self) -> String {
        r#"
/* Prism.js Core - Minimal Implementation */
(function() {
    if (typeof window === 'undefined') return;

    var Prism = {
        util: {
            encode: function(tokens) {
                if (tokens instanceof Array) {
                    return tokens.map(function(token) {
                        return Prism.util.encode(token);
                    });
                } else if (typeof tokens === 'object' && tokens.content) {
                    return tokens.type + ' ' + Prism.util.encode(tokens.content);
                } else {
                    return tokens.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                }
            }
        },
        
        languages: {
            rust: {
                'comment': {
                    pattern: /(^|[^\\])\/\*[\s\S]*?\*\/|(^|[^\\:])\/\/.*/,
                    lookbehind: true
                },
                'string': /"(?:[^"\\]|\\.)*"/,
                'keyword': /\b(?:as|break|const|continue|crate|else|enum|extern|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|type|unsafe|use|where|while)\b/,
                'function': /\b[a-z_][a-z0-9_]*(?=\s*\()/i,
                'macro': /\b[a-z_][a-z0-9_]*!/i,
                'number': /\b(?:0x[a-f0-9]+|0o[0-7]+|0b[01]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/i,
                'punctuation': /[{}[\];(),.:]/
            },
            
            toml: {
                'comment': /#.*/,
                'string': /"(?:[^"\\]|\\.)*"/,
                'section': /^\[[^\]]+\]/m,
                'key': /^[^=\n]+(?==)/m,
                'number': /\b\d+(?:\.\d+)?\b/,
                'boolean': /\b(?:true|false)\b/,
                'punctuation': /[=\[\]]/
            }
        },
        
        highlight: function(text, grammar) {
            // Simple regex-based highlighting
            var tokens = text;
            
            for (var key in grammar) {
                var pattern = grammar[key];
                if (pattern && pattern.pattern) {
                    tokens = tokens.replace(pattern.pattern, function(match) {
                        return '<span class="token ' + key + '">' + match + '</span>';
                    });
                } else if (pattern) {
                    tokens = tokens.replace(pattern, function(match) {
                        return '<span class="token ' + key + '">' + match + '</span>';
                    });
                }
            }
            
            return tokens;
        },
        
        highlightAll: function() {
            var elements = document.querySelectorAll('code[class*="language-"], pre[class*="language-"]');
            
            elements.forEach(function(element) {
                var language = element.className.match(/language-(\w+)/);
                if (language && Prism.languages[language[1]]) {
                    var grammar = Prism.languages[language[1]];
                    element.innerHTML = Prism.highlight(element.textContent, grammar);
                }
            });
        }
    };
    
    window.Prism = Prism;
    
    // Auto-highlight on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', Prism.highlightAll);
    } else {
        Prism.highlightAll();
    }
})();
        "#.to_string()
    }
}

impl Default for DocumentationSite {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
pub fn generate_documentation_website() -> Result<()> {
    let mut doc_site = DocumentationSite::new();
    doc_site.build_comprehensive_documentation()?;

    let output_dir = "docs";
    doc_site.generate_html_documentation(output_dir)?;
    doc_site.generate_readme(Path::new(output_dir))?;

    println!(
        "Documentation website generated in '{}' directory",
        output_dir
    );
    println!("Open 'docs/index.html' in a web browser to view the documentation");

    Ok(())
}

#[allow(dead_code)]
pub fn export_documentation_to_formats() -> Result<()> {
    let doc_site = DocumentationSite::new();

    // Export to different formats
    export_to_markdown(&doc_site)?;
    export_to_json(&doc_site)?;

    Ok(())
}

#[allow(dead_code)]
fn export_to_markdown(_docsite: &DocumentationSite) -> Result<()> {
    let mut markdown = String::new();

    markdown.push_str(&format!("# {}\n\n", doc_site.title));
    markdown.push_str(&format!("{}\n\n", doc_site.description));

    for module in &doc_site.modules {
        markdown.push_str(&format!("## {}\n\n", module.name));
        markdown.push_str(&format!("{}\n\n", module.description));

        for function in &module.functions {
            markdown.push_str(&format!("### {}\n\n", function.name));
            markdown.push_str(&format!("```rust\n{}\n```\n\n", function.signature));
            markdown.push_str(&format!("{}\n\n", function.description));
        }
    }

    let mut file = fs::File::create("docs/API_REFERENCE.md")?;
    file.write_all(markdown.as_bytes())?;

    Ok(())
}

#[allow(dead_code)]
fn export_to_json(_docsite: &DocumentationSite) -> Result<()> {
    let json_data = serde_json::to_string_pretty(_doc_site)?;
    let mut file = fs::File::create("docs/documentation.json")?;
    file.write_all(json_data.as_bytes())?;

    Ok(())
}
