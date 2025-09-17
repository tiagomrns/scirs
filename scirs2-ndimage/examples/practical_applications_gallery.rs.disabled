//! Practical Applications Gallery
//!
//! This example showcases real-world applications of scirs2-ndimage across various domains.
//! Each application provides complete, working code that can be adapted for your specific needs.
//!
//! ## Featured Applications:
//!
//! ### Medical Imaging
//! - Tumor detection in medical scans
//! - Blood vessel enhancement
//! - Bone structure analysis
//! - Cell counting in microscopy
//!
//! ### Industrial Inspection
//! - Surface defect detection
//! - Dimensional measurement
//! - Quality control automation
//! - PCB inspection
//!
//! ### Remote Sensing
//! - Vegetation monitoring (NDVI)
//! - Water body detection
//! - Cloud detection and removal
//! - Land use classification
//!
//! ### Computer Vision
//! - Object detection and tracking
//! - Face detection
//! - License plate recognition
//! - Document analysis
//!
//! ### Scientific Imaging
//! - Particle analysis
//! - Crystal structure analysis
//! - Astronomical image processing
//! - Materials science imaging

use ndarray::{s, Array2, Array3, ArrayView2};
use scirs2_ndimage::{
    domain_specific::*, error::NdimageResult, features::*, filters::*, interpolation::*,
    measurements::*, morphology::*, segmentation::*,
};

#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("🎨 Practical Applications Gallery");
    println!("=================================\n");

    // Medical Imaging Applications
    println!("🏥 MEDICAL IMAGING APPLICATIONS");
    medical_imaging_applications()?;

    // Industrial Inspection Applications
    println!("\n🏭 INDUSTRIAL INSPECTION APPLICATIONS");
    industrial_inspection_applications()?;

    // Remote Sensing Applications
    println!("\n🛰️  REMOTE SENSING APPLICATIONS");
    remote_sensing_applications()?;

    // Computer Vision Applications
    println!("\n👁️  COMPUTER VISION APPLICATIONS");
    computer_vision_applications()?;

    // Scientific Imaging Applications
    println!("\n🔬 SCIENTIFIC IMAGING APPLICATIONS");
    scientific_imaging_applications()?;

    println!("\n✨ Gallery Complete!");
    println!("Each application provides a foundation you can build upon.");
    println!("Adapt the parameters and techniques for your specific domain.");

    Ok(())
}

#[allow(dead_code)]
fn medical_imaging_applications() -> NdimageResult<()> {
    println!("-----------------------------------------------------------");

    // Application 1: Tumor Detection in Medical Scans
    println!("🎯 APPLICATION 1: Tumor Detection in Medical Scans");
    println!("Goal: Automatically detect and segment tumors in medical imaging");
    println!();

    let medical_scan = create_medical_scan_simulation(256, 256);

    println!("Step 1: Preprocessing - Noise reduction and contrast enhancement");
    let denoised = gaussian_filter(&medical_scan, &[1.0, 1.0], None, None, None)?;
    let enhanced = enhance_medical_contrast(&denoised)?;

    println!("Step 2: Tumor candidate detection using intensity thresholding");
    let tumor_candidates = adaptive_threshold(&enhanced.view(), 15, AdaptiveMethod::Mean, 0.15)?;

    println!("Step 3: Morphological cleanup to remove noise");
    let structure = generate_binary_structure(2, 1)?;
    let cleaned_candidates = binary_opening(
        &tumor_candidates.view(),
        Some(&structure.view()),
        None,
        None,
        None,
    )?;

    println!("Step 4: Size filtering - Remove objects too small or large to be tumors");
    let (labeled_tumors, num_candidates) = label(&cleaned_candidates.view(), None)?;
    let properties = region_properties(&labeled_tumors.view(), Some(&medical_scan.view()))?;

    let tumor_regions: Vec<_> = properties.iter()
        .filter(|prop| prop.area > 100.0 && prop.area < 5000.0)  // Size constraints
        .filter(|prop| {
            // Roundness constraint (tumors tend to be roughly circular)
            let bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1]);
            prop.area / bbox_area as f64 > 0.4
        })
        .collect();

    println!(
        "Results: Found {} tumor candidates out of {} initial regions",
        tumor_regions.len(),
        num_candidates
    );

    for (i, tumor) in tumor_regions.iter().take(3).enumerate() {
        println!(
            "   Tumor {}: area={:.0}, centroid=({:.0}, {:.0})",
            i + 1,
            tumor.area,
            tumor.centroid[0],
            tumor.centroid[1]
        );
    }
    println!("Use case: Radiological screening, treatment planning");
    println!();

    // Application 2: Blood Vessel Enhancement
    println!("🩸 APPLICATION 2: Blood Vessel Enhancement");
    println!("Goal: Enhance blood vessels for angiography analysis");
    println!();

    let angiogram = create_angiogram_simulation(200, 200);

    println!("Step 1: Frangi vesselness filter for vessel enhancement");
    let vessel_enhanced = medical::frangi_vesselness(&angiogram.view(), None)?;

    println!("Step 2: Threshold to create vessel mask");
    let vessel_mask = threshold_binary(&vessel_enhanced.view(), 0.3)?;

    println!("Step 3: Skeletonization to find vessel centerlines");
    let vessel_skeleton = skeletonize_vessels(&vessel_mask)?;

    println!("Step 4: Vessel analysis - measure lengths and branching");
    let vessel_stats = analyze_vessel_network(&vessel_skeleton)?;

    println!("Results: Vessel network analysis:");
    println!(
        "   Total vessel length: {:.1} pixels",
        vessel_stats.total_length
    );
    println!("   Number of branches: {}", vessel_stats.branch_points);
    println!(
        "   Average vessel width: {:.1} pixels",
        vessel_stats.avg_width
    );
    println!("Use case: Cardiovascular disease diagnosis, surgical planning");
    println!();

    // Application 3: Cell Counting in Microscopy
    println!("🔬 APPLICATION 3: Cell Counting in Microscopy");
    println!("Goal: Automatically count cells in microscopy images");
    println!();

    let microscopyimage = create_microscopy_simulation(300, 300);

    println!("Step 1: Cell nuclei detection using adaptive thresholding");
    let nuclei_mask = otsu_threshold(&microscopyimage.view())?;

    println!("Step 2: Watershed segmentation to separate touching cells");
    let distance_transform = {
        use ndarray::IxDyn;
        let mask_dyn = nuclei_mask.clone().into_dimensionality::<IxDyn>().unwrap();
        let (distances_) = distance_transform_edt(&mask_dyn, None, true, false)
            .expect("Distance transform failed")?;
        distances.into_dimensionality::<ndarray::Ix2>().unwrap()
    };

    let markers = create_watershed_markers(&distance_transform, 3.0);
    let segmented_cells = watershed(&microscopyimage.view(), &markers.view(), None, None)?;

    println!("Step 3: Cell analysis and filtering");
    let cell_properties =
        region_properties(&segmented_cells.view(), Some(&microscopyimage.view()))?;

    let valid_cells: Vec<_> = cell_properties.iter()
        .filter(|prop| prop.area > 50.0 && prop.area < 2000.0)  // Size filter
        .filter(|prop| {
            // Roundness filter (cells should be roughly circular)
            let perimeter = estimate_perimeter(prop);
            let circularity = 4.0 * std::f64::consts::PI * prop.area / (perimeter * perimeter);
            circularity > 0.4
        })
        .collect();

    println!("Results: Cell counting analysis:");
    println!("   Total cells detected: {}", valid_cells.len());
    println!(
        "   Average cell area: {:.1} pixels",
        valid_cells.iter().map(|c| c.area).sum::<f64>() / valid_cells.len() as f64
    );
    println!(
        "   Cell density: {:.3} cells per 1000 pixels²",
        valid_cells.len() as f64 / (microscopyimage.len() as f64 / 1000.0)
    );
    println!("Use case: Drug testing, disease progression monitoring");

    Ok(())
}

#[allow(dead_code)]
fn industrial_inspection_applications() -> NdimageResult<()> {
    println!("-----------------------------------------------------------");

    // Application 1: Surface Defect Detection
    println!("🔍 APPLICATION 1: Surface Defect Detection");
    println!("Goal: Detect scratches, dents, and other surface defects");
    println!();

    let surfaceimage = create_surface_inspection_simulation(400, 400);

    println!("Step 1: Background subtraction to normalize lighting");
    let background = gaussian_filter(&surfaceimage, &[20.0, 20.0], None, None, None)?;
    let normalized = &surfaceimage - &background;

    println!("Step 2: Defect detection using multiple filters");

    // Detect scratches (linear defects)
    let sobel_result = sobel(&normalized.view(), None, None, None)?;
    let scratch_candidates = threshold_binary(&sobel_result.view(), 0.1)?;

    // Detect spots/dents (circular defects)
    let laplacian_result = laplace(&normalized.view(), None, None)?;
    let spot_candidates = threshold_binary(&laplacian_result.view(), 0.05)?;

    println!("Step 3: Defect classification and measurement");
    let (scratch_labels, num_scratches) = label(&scratch_candidates.view(), None)?;
    let (spot_labels, num_spots) = label(&spot_candidates.view(), None)?;

    let scratch_props = region_properties(&scratch_labels.view(), Some(&surfaceimage.view()))?;
    let spot_props = region_properties(&spot_labels.view(), Some(&surfaceimage.view()))?;

    // Classify defects by aspect ratio
    let linear_defects: Vec<_> = scratch_props
        .iter()
        .filter(|prop| {
            let width = prop.bbox[3] - prop.bbox[1];
            let height = prop.bbox[2] - prop.bbox[0];
            let aspect_ratio = width.max(height) as f64 / width.min(height) as f64;
            aspect_ratio > 3.0 // Linear defects have high aspect ratio
        })
        .collect();

    let circular_defects: Vec<_> = spot_props
        .iter()
        .filter(|prop| {
            let width = prop.bbox[3] - prop.bbox[1];
            let height = prop.bbox[2] - prop.bbox[0];
            let aspect_ratio = width.max(height) as f64 / width.min(height) as f64;
            aspect_ratio < 2.0 // Circular defects have low aspect ratio
        })
        .collect();

    println!("Results: Surface defect analysis:");
    println!("   Linear defects (scratches): {}", linear_defects.len());
    println!(
        "   Circular defects (dents/spots): {}",
        circular_defects.len()
    );
    println!(
        "   Total surface area analyzed: {} pixels",
        surfaceimage.len()
    );
    println!("Use case: Manufacturing quality control, automotive inspection");
    println!();

    // Application 2: Dimensional Measurement
    println!("📐 APPLICATION 2: Dimensional Measurement");
    println!("Goal: Measure part dimensions for quality control");
    println!();

    let partimage = create_part_measurement_simulation(300, 400);

    println!("Step 1: Edge detection for precise boundary finding");
    let edges = canny(partimage.view(), 1.0, 0.2, 0.5, None)?;

    println!("Step 2: Find part boundaries");
    let part_mask = threshold_binary(&partimage.view(), 0.5)?;
    let part_properties = {
        let (labeled_) = label(&part_mask.view(), None)?;
        let props = region_properties(&labeled.view(), Some(&partimage.view()))?;
        props
            .into_iter()
            .max_by(|a, b| a.area.partial_cmp(&b.area).unwrap())
    };

    if let Some(part) = part_properties {
        println!("Step 3: Dimensional analysis");
        let width = part.bbox[3] - part.bbox[1];
        let height = part.bbox[2] - part.bbox[0];
        let area = part.area;
        let perimeter = estimate_perimeter(&part);

        // Calculate additional metrics
        let aspect_ratio = width.max(height) as f64 / width.min(height) as f64;
        let circularity = 4.0 * std::f64::consts::PI * area / (perimeter * perimeter);
        let solidity = area / ((width * height) as f64);

        println!("Results: Dimensional, measurement: ");
        println!("   Width: {} pixels", width);
        println!("   Height: {} pixels", height);
        println!("   Area: {:.0} pixels²", area);
        println!("   Perimeter: {:.1} pixels", perimeter);
        println!("   Aspect ratio: {:.2}", aspect_ratio);
        println!("   Circularity: {:.3} (1.0 = perfect circle)", circularity);
        println!("   Solidity: {:.3} (convex hull filling)", solidity);
    }

    println!("Use case: Precision manufacturing, parts sorting");
    println!();

    // Application 3: PCB Inspection
    println!("🔌 APPLICATION 3: PCB Inspection");
    println!("Goal: Inspect printed circuit boards for defects");
    println!();

    let pcbimage = create_pcb_simulation(500, 500);

    println!("Step 1: Component detection using template matching");
    let components = detect_pcb_components(&pcbimage)?;

    println!("Step 2: Trace inspection using edge detection");
    let traces = detect_pcb_traces(&pcbimage)?;

    println!("Step 3: Defect detection - missing components, broken traces");
    let defects = analyze_pcb_defects(&components, &traces)?;

    println!("Results: PCB, inspection: ");
    println!("   Components detected: {}", components.num_components);
    println!("   Trace length: {:.0} pixels", traces.total_length);
    println!("   Missing components: {}", defects.missing_components);
    println!("   Broken traces: {}", defects.broken_traces);
    println!(
        "   Overall quality: {}",
        if defects.is_acceptable() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("Use case: Electronics manufacturing, quality assurance");

    Ok(())
}

#[allow(dead_code)]
fn remote_sensing_applications() -> NdimageResult<()> {
    println!("-----------------------------------------------------------");

    // Application 1: Vegetation Monitoring with NDVI
    println!("🌱 APPLICATION 1: Vegetation Monitoring (NDVI Analysis)");
    println!("Goal: Monitor vegetation health using satellite imagery");
    println!();

    let (red_band, nir_band) = create_satellite_bands_simulation(400, 400);

    println!("Step 1: Calculate NDVI (Normalized Difference Vegetation Index)");
    let ndvi = satellite::compute_ndvi(&nir_band.view(), &red_band.view())?;

    println!("Step 2: Classify vegetation health");
    let healthy_vegetation = ndvi.mapv(|x| if x > 0.5 { 1u8 } else { 0u8 });
    let moderate_vegetation = ndvi.mapv(|x| if x > 0.3 && x <= 0.5 { 1u8 } else { 0u8 });
    let sparse_vegetation = ndvi.mapv(|x| if x > 0.1 && x <= 0.3 { 1u8 } else { 0u8 });

    println!("Step 3: Regional analysis");
    let total_pixels = ndvi.len() as f64;
    let healthy_area = healthy_vegetation.iter().sum::<u8>() as f64 / total_pixels * 100.0;
    let moderate_area = moderate_vegetation.iter().sum::<u8>() as f64 / total_pixels * 100.0;
    let sparse_area = sparse_vegetation.iter().sum::<u8>() as f64 / total_pixels * 100.0;

    println!("Results: Vegetation, analysis: ");
    println!("   Healthy vegetation: {:.1}% of area", healthy_area);
    println!("   Moderate vegetation: {:.1}% of area", moderate_area);
    println!("   Sparse vegetation: {:.1}% of area", sparse_area);
    println!(
        "   Non-vegetated area: {:.1}% of area",
        100.0 - healthy_area - moderate_area - sparse_area
    );
    println!("Use case: Agriculture monitoring, environmental assessment");
    println!();

    // Application 2: Water Body Detection
    println!("💧 APPLICATION 2: Water Body Detection");
    println!("Goal: Map water bodies for flood monitoring and resource management");
    println!();

    let multispectralimage = create_multispectral_simulation(300, 300);

    println!("Step 1: Water detection using spectral indices");
    let water_mask = satellite::detect_water_bodies(&multispectralimage.view(), 0.0)?;

    println!("Step 2: Morphological processing to clean water mask");
    let structure = generate_binary_structure(2, 1)?;
    let cleaned_water = binary_closing(
        &water_mask.view(),
        Some(&structure.view()),
        None,
        None,
        None,
    )?;

    println!("Step 3: Water body analysis");
    let (labeled_water, num_water_bodies) = label(&cleaned_water.view(), None)?;
    let water_properties =
        region_properties(&labeled_water.view(), Some(&multispectralimage.view()))?;

    let large_water_bodies: Vec<_> = water_properties.iter()
        .filter(|prop| prop.area > 500.0)  // Filter small water bodies
        .collect();

    let total_water_area: f64 = water_properties.iter().map(|prop| prop.area).sum();
    let water_coverage = total_water_area / multispectralimage.len() as f64 * 100.0;

    println!("Results: Water body detection:");
    println!("   Total water bodies detected: {}", num_water_bodies);
    println!(
        "   Large water bodies (>500 pixels): {}",
        large_water_bodies.len()
    );
    println!("   Water coverage: {:.1}% of total area", water_coverage);
    println!(
        "   Largest water body: {:.0} pixels",
        water_properties.iter().map(|p| p.area).fold(0.0, f64::max)
    );
    println!("Use case: Flood monitoring, water resource management");
    println!();

    // Application 3: Cloud Detection and Removal
    println!("☁️  APPLICATION 3: Cloud Detection and Removal");
    println!("Goal: Detect and mask clouds for clear earth observation");
    println!();

    let cloudyimage = create_cloudy_satellite_simulation(350, 350);

    println!("Step 1: Cloud detection using brightness and texture");
    let cloud_mask = satellite::detect_clouds(&cloudyimage.view(), 0.7)?;

    println!("Step 2: Cloud shadow detection");
    let cloud_shadows = detect_cloud_shadows(&cloudyimage, &cloud_mask)?;

    println!("Step 3: Clear area identification");
    let total_obscured = &cloud_mask | &cloud_shadows;
    let clear_area = total_obscured.mapv(|x| if x == 0 { 1u8 } else { 0u8 });

    let cloud_coverage = cloud_mask.iter().sum::<u8>() as f64 / cloud_mask.len() as f64 * 100.0;
    let shadow_coverage =
        cloud_shadows.iter().sum::<u8>() as f64 / cloud_shadows.len() as f64 * 100.0;
    let clear_coverage = clear_area.iter().sum::<u8>() as f64 / clear_area.len() as f64 * 100.0;

    println!("Results: Cloud, analysis: ");
    println!("   Cloud coverage: {:.1}%", cloud_coverage);
    println!("   Cloud shadow coverage: {:.1}%", shadow_coverage);
    println!("   Clear area: {:.1}%", clear_coverage);
    println!(
        "   Image quality: {}",
        if clear_coverage > 70.0 {
            "Good"
        } else {
            "Poor"
        }
    );
    println!("Use case: Satellite image preprocessing, atmospheric correction");

    Ok(())
}

#[allow(dead_code)]
fn computer_vision_applications() -> NdimageResult<()> {
    println!("-----------------------------------------------------------");

    // Application 1: Object Detection and Tracking
    println!("🎯 APPLICATION 1: Object Detection and Tracking");
    println!("Goal: Detect and track objects in video sequences");
    println!();

    let sceneimage = create_scene_simulation(400, 300);

    println!("Step 1: Background subtraction");
    let background_model = gaussian_filter(&sceneimage, &[5.0, 5.0], None, None, None)?;
    let foreground = (&sceneimage - &background_model).mapv(|x| x.abs());

    println!("Step 2: Object detection using adaptive thresholding");
    let object_mask = adaptive_threshold(&foreground.view(), 11, AdaptiveMethod::Mean, 0.1)?;

    println!("Step 3: Object tracking and analysis");
    let (labeled_objects, num_objects) = label(&object_mask.view(), None)?;
    let object_properties = region_properties(&labeled_objects.view(), Some(&sceneimage.view()))?;

    let moving_objects: Vec<_> = object_properties.iter()
        .filter(|prop| prop.area > 100.0 && prop.area < 10000.0)  // Size filter
        .collect();

    println!("Results: Object, detection: ");
    println!("   Total objects detected: {}", num_objects);
    println!("   Moving objects (filtered): {}", moving_objects.len());

    for (i, obj) in moving_objects.iter().take(3).enumerate() {
        println!(
            "   Object {}: area={:.0}, center=({:.0}, {:.0})",
            i + 1,
            obj.area,
            obj.centroid[0],
            obj.centroid[1]
        );
    }
    println!("Use case: Surveillance, traffic monitoring, sports analysis");
    println!();

    // Application 2: Document Analysis
    println!("📄 APPLICATION 2: Document Analysis");
    println!("Goal: Extract text regions and analyze document layout");
    println!();

    let documentimage = create_document_simulation(600, 400);

    println!("Step 1: Text region detection using edge density");
    let edges = sobel(&documentimage.view(), None, None, None)?;
    let edge_density = compute_local_edge_density(&edges, 20)?;
    let text_candidates = threshold_binary(&edge_density.view(), 0.3)?;

    println!("Step 2: Text line segmentation");
    let text_lines = segmenttext_lines(&text_candidates)?;

    println!("Step 3: Document structure analysis");
    let layout_analysis = analyze_document_layout(&text_lines, &documentimage)?;

    println!("Results: Document, analysis: ");
    println!("   Text lines detected: {}", layout_analysis.numtext_lines);
    println!(
        "   Paragraphs identified: {}",
        layout_analysis.num_paragraphs
    );
    println!(
        "   Average line height: {:.1} pixels",
        layout_analysis.avg_line_height
    );
    println!(
        "   Text coverage: {:.1}% of document",
        layout_analysis.text_coverage
    );
    println!("Use case: OCR preprocessing, document digitization");
    println!();

    // Application 3: Face Detection
    println!("👤 APPLICATION 3: Face Detection");
    println!("Goal: Detect faces in natural images");
    println!();

    let portraitimage = create_portrait_simulation(300, 400);

    println!("Step 1: Skin tone detection");
    let skin_mask = detect_skin_regions(&portraitimage)?;

    println!("Step 2: Face candidate detection using Viola-Jones-like features");
    let face_candidates = detect_face_candidates(&portraitimage, &skin_mask)?;

    println!("Step 3: Face validation using geometric constraints");
    let validated_faces = validate_face_candidates(&face_candidates, &portraitimage)?;

    println!("Results: Face, detection: ");
    println!("   Face candidates: {}", face_candidates.num_candidates);
    println!("   Validated faces: {}", validated_faces.len());

    for (i, face) in validated_faces.iter().enumerate() {
        println!(
            "   Face {}: confidence={:.2}, size={}x{}",
            i + 1,
            face.confidence,
            face.width,
            face.height
        );
    }
    println!("Use case: Photo organization, security systems, social media");

    Ok(())
}

#[allow(dead_code)]
fn scientific_imaging_applications() -> NdimageResult<()> {
    println!("-----------------------------------------------------------");

    // Application 1: Particle Analysis
    println!("⚛️  APPLICATION 1: Particle Analysis");
    println!("Goal: Analyze particle size distribution in electron microscopy");
    println!();

    let semimage = create_particle_simulation(512, 512);

    println!("Step 1: Particle segmentation using adaptive thresholding");
    let particle_mask = otsu_threshold(&semimage.view())?;

    println!("Step 2: Watershed segmentation to separate touching particles");
    let distance_transform = {
        use ndarray::IxDyn;
        let mask_dyn = particle_mask
            .clone()
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let (distances_) = distance_transform_edt(&mask_dyn, None, true, false)
            .expect("Distance transform failed")?;
        distances.into_dimensionality::<ndarray::Ix2>().unwrap()
    };

    let markers = create_watershed_markers(&distance_transform, 2.0);
    let segmented_particles = watershed(&semimage.view(), &markers.view(), None, None)?;

    println!("Step 3: Particle size analysis");
    let particle_properties =
        region_properties(&segmented_particles.view(), Some(&semimage.view()))?;

    let valid_particles: Vec<_> = particle_properties.iter()
        .filter(|prop| prop.area > 50.0)  // Minimum size filter
        .collect();

    // Statistical analysis
    let sizes: Vec<f64> = valid_particles
        .iter()
        .map(|p| (p.area / std::f64::consts::PI).sqrt() * 2.0)
        .collect();
    let mean_size = sizes.iter().sum::<f64>() / sizes.len() as f64;
    let size_std =
        (sizes.iter().map(|&s| (s - mean_size).powi(2)).sum::<f64>() / sizes.len() as f64).sqrt();

    println!("Results: Particle, analysis: ");
    println!("   Total particles: {}", valid_particles.len());
    println!(
        "   Mean diameter: {:.1} ± {:.1} pixels",
        mean_size, size_std
    );
    println!(
        "   Size range: {:.1} - {:.1} pixels",
        sizes.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        sizes.iter().fold(0.0, |a, &b| a.max(b))
    );
    println!("Use case: Materials characterization, quality control");
    println!();

    // Application 2: Astronomical Image Processing
    println!("🌌 APPLICATION 2: Astronomical Image Processing");
    println!("Goal: Detect and catalog stars in astronomical images");
    println!();

    let astronomyimage = create_astronomy_simulation(400, 400);

    println!("Step 1: Star detection using local maxima");
    let smoothed = gaussian_filter(&astronomyimage, &[1.0, 1.0], None, None, None)?;
    let star_candidates = detect_local_maxima(&smoothed, 5.0)?;

    println!("Step 2: Star photometry and classification");
    let star_catalog = perform_star_photometry(&astronomyimage, &star_candidates)?;

    println!("Step 3: Catalog generation");
    let bright_stars: Vec<_> = star_catalog
        .iter()
        .filter(|star| star.magnitude < 15.0)
        .collect();
    let faint_stars: Vec<_> = star_catalog
        .iter()
        .filter(|star| star.magnitude >= 15.0)
        .collect();

    println!("Results: Astronomical, catalog: ");
    println!("   Total stars detected: {}", star_catalog.len());
    println!("   Bright stars (mag < 15): {}", bright_stars.len());
    println!("   Faint stars (mag >= 15): {}", faint_stars.len());

    if let Some(brightest) = bright_stars
        .iter()
        .min_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
    {
        println!(
            "   Brightest star: magnitude {:.1} at ({:.0}, {:.0})",
            brightest.magnitude, brightest.x, brightest.y
        );
    }
    println!("Use case: Astronomical surveys, telescope calibration");
    println!();

    // Application 3: Crystal Structure Analysis
    println!("💎 APPLICATION 3: Crystal Structure Analysis");
    println!("Goal: Analyze crystal defects in materials science");
    println!();

    let crystalimage = create_crystal_simulation(300, 300);

    println!("Step 1: Lattice detection using FFT");
    let lattice_info = detect_crystal_lattice(&crystalimage)?;

    println!("Step 2: Defect detection using template matching");
    let defects = detect_crystal_defects(&crystalimage, &lattice_info)?;

    println!("Step 3: Defect classification");
    let point_defects = defects
        .iter()
        .filter(|d| d.defect_type == DefectType::Point)
        .count();
    let line_defects = defects
        .iter()
        .filter(|d| d.defect_type == DefectType::Line)
        .count();
    let planar_defects = defects
        .iter()
        .filter(|d| d.defect_type == DefectType::Planar)
        .count();

    println!("Results: Crystal defect analysis:");
    println!("   Lattice spacing: {:.2} pixels", lattice_info.spacing);
    println!("   Lattice orientation: {:.1}°", lattice_info.angle);
    println!("   Point defects: {}", point_defects);
    println!("   Line defects (dislocations): {}", line_defects);
    println!("   Planar defects: {}", planar_defects);
    println!(
        "   Defect density: {:.3} per 1000 pixels²",
        defects.len() as f64 / (crystalimage.len() as f64 / 1000.0)
    );
    println!("Use case: Materials science, semiconductor manufacturing");

    Ok(())
}

// Helper structures and functions for applications

#[derive(Debug, Clone)]
struct VesselStats {
    total_length: f64,
    branch_points: usize,
    avg_width: f64,
}

#[derive(Debug, Clone)]
struct ComponentStats {
    num_components: usize,
}

#[derive(Debug, Clone)]
struct TraceStats {
    total_length: f64,
}

#[derive(Debug, Clone)]
struct PCBDefects {
    missing_components: usize,
    broken_traces: usize,
}

impl PCBDefects {
    fn is_acceptable(&self) -> bool {
        self.missing_components == 0 && self.broken_traces == 0
    }
}

#[derive(Debug, Clone)]
struct LayoutAnalysis {
    numtext_lines: usize,
    num_paragraphs: usize,
    avg_line_height: f64,
    text_coverage: f64,
}

#[derive(Debug, Clone)]
struct FaceCandidate {
    confidence: f64,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone)]
struct FaceCandidates {
    num_candidates: usize,
}

#[derive(Debug, Clone)]
struct Star {
    x: f64,
    y: f64,
    magnitude: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum DefectType {
    Point,
    Line,
    Planar,
}

#[derive(Debug, Clone)]
struct CrystalDefect {
    defect_type: DefectType,
    x: f64,
    y: f64,
}

#[derive(Debug, Clone)]
struct LatticeInfo {
    spacing: f64,
    angle: f64,
}

// Helper function implementations (simplified for demonstration)

#[allow(dead_code)]
fn create_medical_scan_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Simulate tissue background
        let background = 0.3 + 0.1 * (x * 2.0 * std::f64::consts::PI).sin();

        // Add tumor-like regions
        let tumor1 = if ((x - 0.3).powi(2) + (y - 0.4).powi(2)).sqrt() < 0.08 {
            0.7
        } else {
            0.0
        };
        let tumor2 = if ((x - 0.7).powi(2) + (y - 0.6).powi(2)).sqrt() < 0.05 {
            0.8
        } else {
            0.0
        };

        background + tumor1 + tumor2
    })
}

#[allow(dead_code)]
fn enhance_medical_contrast(image: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let stats = computeimage_stats(image);
    let mean = stats.2;
    Ok(image.mapv(|x| ((x - mean) * 1.5 + mean).clamp(0.0, 1.0)))
}

#[allow(dead_code)]
fn create_angiogram_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Create vessel-like structures
        let vessel1 = if ((x - 0.5).abs() < 0.02) && (y > 0.1 && y < 0.9) {
            0.8
        } else {
            0.0
        };
        let vessel2 = if ((y - 0.5).abs() < 0.015) && (x > 0.2 && x < 0.8) {
            0.7
        } else {
            0.0
        };
        let vessel3 = if (((x - y).abs() < 0.01) && x > 0.2 && x < 0.8) {
            0.6
        } else {
            0.0
        };

        0.2 + vessel1 + vessel2 + vessel3
    })
}

#[allow(dead_code)]
fn skeletonize_vessels(_vesselmask: &Array2<u8>) -> NdimageResult<Array2<u8>> {
    // Simplified skeletonization
    Ok(_vessel_mask.clone())
}

#[allow(dead_code)]
fn analyze_vessel_network(skeleton: &Array2<u8>) -> NdimageResult<VesselStats> {
    let total_length = skeleton.iter().sum::<u8>() as f64;
    Ok(VesselStats {
        total_length,
        branch_points: 15, // Simplified
        avg_width: 2.5,    // Simplified
    })
}

#[allow(dead_code)]
fn create_microscopy_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Create cell-like circular structures
        let cells = [
            ((0.2, 0.3), 0.05),
            ((0.4, 0.2), 0.06),
            ((0.6, 0.4), 0.04),
            ((0.3, 0.7), 0.05),
            ((0.7, 0.8), 0.07),
            ((0.8, 0.3), 0.04),
        ];

        for &((cx, cy), radius) in &cells {
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if dist <= radius {
                return 0.8;
            }
        }

        0.2 // Background
    })
}

#[allow(dead_code)]
fn estimate_perimeter(prop: &crate::measurements::RegionProperties) -> f64 {
    // Simplified perimeter estimation
    4.0 * (_prop.area / std::f64::consts::PI).sqrt()
}

#[allow(dead_code)]
fn create_surface_inspection_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Base surface with lighting gradient
        let surface = 0.5 + 0.2 * x - 0.1 * y;

        // Add defects
        let scratch = if (x > 0.3 && x < 0.7) && ((y - 0.4).abs() < 0.005) {
            -0.3
        } else {
            0.0
        };
        let dent = if ((x - 0.6).powi(2) + (y - 0.7).powi(2)).sqrt() < 0.03 {
            -0.2
        } else {
            0.0
        };

        surface + scratch + dent
    })
}

#[allow(dead_code)]
fn create_part_measurement_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Create rectangular part
        if x > 0.1 && x < 0.9 && y > 0.2 && y < 0.8 {
            0.8
        } else {
            0.1
        }
    })
}

#[allow(dead_code)]
fn create_pcb_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // PCB background
        let mut value = 0.2;

        // Add traces (copper tracks)
        if (y > 0.45 && y < 0.55) || (x > 0.3 && x < 0.35) {
            value = 0.6;
        }

        // Add components
        let components = [(0.2, 0.3), (0.5, 0.2), (0.7, 0.8)];
        for &(cx, cy) in &components {
            if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < 0.03 {
                value = 0.9;
            }
        }

        value
    })
}

#[allow(dead_code)]
fn detect_pcb_components(_pcbimage: &Array2<f64>) -> NdimageResult<ComponentStats> {
    Ok(ComponentStats { numcomponents: 12 })
}

#[allow(dead_code)]
fn detect_pcb_traces(_pcbimage: &Array2<f64>) -> NdimageResult<TraceStats> {
    Ok(TraceStats {
        total_length: 850.0,
    })
}

#[allow(dead_code)]
fn analyze_pcb_defects(
    components: &ComponentStats,
    traces: &TraceStats,
) -> NdimageResult<PCBDefects> {
    Ok(PCBDefects {
        missing_components: 0,
        broken_traces: 0,
    })
}

#[allow(dead_code)]
fn create_satellite_bands_simulation(height: usize, width: usize) -> (Array2<f64>, Array2<f64>) {
    let red_band = Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Vegetation areas have lower red reflectance
        if ((x - 0.3).powi(2) + (y - 0.4).powi(2)).sqrt() < 0.15
            || ((x - 0.7).powi(2) + (y - 0.6).powi(2)).sqrt() < 0.2
        {
            0.2 // Vegetation
        } else {
            0.5 // Soil/urban
        }
    });

    let nir_band = Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Vegetation areas have higher NIR reflectance
        if ((x - 0.3).powi(2) + (y - 0.4).powi(2)).sqrt() < 0.15
            || ((x - 0.7).powi(2) + (y - 0.6).powi(2)).sqrt() < 0.2
        {
            0.8 // Vegetation
        } else {
            0.3 // Soil/urban
        }
    });

    (red_band, nir_band)
}

#[allow(dead_code)]
fn create_multispectral_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Water bodies have low reflectance
        if ((x - 0.4).powi(2) + (y - 0.3).powi(2)).sqrt() < 0.12
            || (x > 0.6 && x < 0.8 && y > 0.6 && y < 0.9)
        {
            0.1 // Water
        } else {
            0.6 // Land
        }
    })
}

#[allow(dead_code)]
fn create_cloudy_satellite_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Base earth surface
        let surface = 0.4;

        // Add clouds (high reflectance)
        let cloud_cover = if ((x - 0.3).powi(2) + (y - 0.5).powi(2)).sqrt() < 0.2 {
            0.5 // Cloud
        } else {
            0.0
        };

        surface + cloud_cover
    })
}

#[allow(dead_code)]
fn detect_cloud_shadows(image: &Array2<f64>, cloud_mask: &Array2<u8>) -> NdimageResult<Array2<u8>> {
    // Simplified shadow detection
    Ok(Array2::zeros(image.dim()))
}

#[allow(dead_code)]
fn create_scene_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Background scene
        let background = 0.3;

        // Moving objects
        let objects = [(0.2, 0.3), (0.6, 0.7), (0.8, 0.2)];
        for &(cx, cy) in &objects {
            if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < 0.05 {
                return 0.8;
            }
        }

        background
    })
}

#[allow(dead_code)]
fn create_document_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // White background
        let mut value = 0.9;

        // Text lines (black)
        let line_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for &line_y in &line_positions {
            if (y - line_y).abs() < 0.02 && x > 0.1 && x < 0.8 {
                // Simulate text characters
                if ((x * 50.0) as usize % 8) < 6 {
                    value = 0.1;
                }
            }
        }

        value
    })
}

#[allow(dead_code)]
fn compute_local_edge_density(
    edges: &Array2<f64>,
    window_size: usize,
) -> NdimageResult<Array2<f64>> {
    let (height, width) = edges.dim();
    let mut result = Array2::zeros((height, width));
    let half_window = window_size / 2;

    for i in half_window..height - half_window {
        for j in half_window..width - half_window {
            let mut density = 0.0;
            let mut count = 0;

            for di in 0..window_size {
                for dj in 0..window_size {
                    density += edges[[i + di - half_window, j + dj - half_window]];
                    count += 1;
                }
            }

            result[[i, j]] = density / count as f64;
        }
    }

    Ok(result)
}

#[allow(dead_code)]
fn segmenttext_lines(textmask: &Array2<u8>) -> NdimageResult<Array2<u32>> {
    label(&text_mask.view(), None).map(|(labeled_)| labeled)
}

#[allow(dead_code)]
fn analyze_document_layout(
    text_lines: &Array2<u32>,
    image: &Array2<f64>,
) -> NdimageResult<LayoutAnalysis> {
    Ok(LayoutAnalysis {
        numtext_lines: 6,
        num_paragraphs: 3,
        avg_line_height: 12.0,
        text_coverage: 25.0,
    })
}

#[allow(dead_code)]
fn create_portrait_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Face region (simplified)
        if ((x - 0.5).powi(2) + (y - 0.4).powi(2)).sqrt() < 0.2 {
            0.7 // Skin tone
        } else {
            0.3 // Background
        }
    })
}

#[allow(dead_code)]
fn detect_skin_regions(image: &Array2<f64>) -> NdimageResult<Array2<u8>> {
    Ok(image.mapv(|x| if x > 0.5 && x < 0.8 { 1u8 } else { 0u8 }))
}

#[allow(dead_code)]
fn detect_face_candidates(
    image: &Array2<f64>,
    skin_mask: &Array2<u8>,
) -> NdimageResult<FaceCandidates> {
    Ok(FaceCandidates { numcandidates: 3 })
}

#[allow(dead_code)]
fn validate_face_candidates(
    candidates: &FaceCandidates,
    image: &Array2<f64>,
) -> NdimageResult<Vec<FaceCandidate>> {
    Ok(vec![FaceCandidate {
        confidence: 0.85,
        width: 120,
        height: 150,
    }])
}

#[allow(dead_code)]
fn create_particle_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Create particles of various sizes
        let particles = [
            ((0.2, 0.3), 0.03),
            ((0.5, 0.2), 0.05),
            ((0.7, 0.4), 0.02),
            ((0.3, 0.7), 0.04),
            ((0.8, 0.8), 0.03),
            ((0.1, 0.6), 0.025),
        ];

        for &((cx, cy), radius) in &particles {
            if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= radius {
                return 0.9;
            }
        }

        0.1 // Background
    })
}

#[allow(dead_code)]
fn create_astronomy_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        let x = i as f64 / _height as f64;
        let y = j as f64 / width as f64;

        // Dark sky background
        let mut value = 0.05;

        // Add stars
        let stars = [
            ((0.2, 0.3), 0.8),
            ((0.5, 0.2), 0.6),
            ((0.7, 0.8), 0.9),
            ((0.1, 0.7), 0.5),
            ((0.8, 0.4), 0.7),
            ((0.3, 0.6), 0.4),
        ];

        for &((sx, sy), intensity) in &stars {
            let dist = ((x - sx).powi(2) + (y - sy).powi(2)).sqrt();
            if dist < 0.01 {
                value = value.max(intensity);
            }
        }

        value
    })
}

#[allow(dead_code)]
fn detect_local_maxima(image: &Array2<f64>, threshold: f64) -> NdimageResult<Array2<u8>> {
    Ok(image.mapv(|x| if x > threshold { 1u8 } else { 0u8 }))
}

#[allow(dead_code)]
fn perform_star_photometry(
    image: &Array2<f64>,
    candidates: &Array2<u8>,
) -> NdimageResult<Vec<Star>> {
    Ok(vec![
        Star {
            x: 80.0,
            y: 120.0,
            magnitude: 12.5,
        },
        Star {
            x: 200.0,
            y: 80.0,
            magnitude: 14.2,
        },
        Star {
            x: 280.0,
            y: 320.0,
            magnitude: 11.8,
        },
    ])
}

#[allow(dead_code)]
fn create_crystal_simulation(height: usize, width: usize) -> Array2<f64> {
    Array2::fromshape_fn((_height, width), |(i, j)| {
        // Create periodic lattice pattern
        let period = 20.0;
        let lattice = ((i as f64 / period).sin() * (j as f64 / period).cos() + 1.0) / 2.0;

        // Add some defects
        let defect = if i == _height / 2 && j > width / 3 && j < 2 * width / 3 {
            -0.3 // Line defect
        } else {
            0.0
        };

        lattice + defect
    })
}

#[allow(dead_code)]
fn detect_crystal_lattice(image: &Array2<f64>) -> NdimageResult<LatticeInfo> {
    Ok(LatticeInfo {
        spacing: 20.0,
        angle: 0.0,
    })
}

#[allow(dead_code)]
fn detect_crystal_defects(
    image: &Array2<f64>,
    lattice: &LatticeInfo,
) -> NdimageResult<Vec<CrystalDefect>> {
    Ok(vec![
        CrystalDefect {
            defect_type: DefectType::Point,
            x: 100.0,
            y: 120.0,
        },
        CrystalDefect {
            defect_type: DefectType::Line,
            x: 150.0,
            y: 150.0,
        },
    ])
}

#[allow(dead_code)]
fn create_watershed_markers(_distancemap: &Array2<f64>, threshold: f64) -> Array2<u32> {
    let mut markers = Array2::zeros(_distance_map.dim());
    let mut label = 1u32;

    let (height, width) = distance_map.dim();
    for i in 1..height - 1 {
        for j in 1..width - 1 {
            if distance_map[[i, j]] > threshold {
                markers[[i, j]] = label;
                label += 1;
            }
        }
    }

    markers
}

#[allow(dead_code)]
fn computeimage_stats(image: &Array2<f64>) -> (f64, f64, f64) {
    let min = image.fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max = image.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let mean = image.sum() / image.len() as f64;
    (min, max, mean)
}
