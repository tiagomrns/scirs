//! Domain-specific file format support
//!
//! This module provides specialized file format support for various scientific domains:
//! - Bioinformatics: FASTA, FASTQ, SAM/BAM, VCF
//! - Geospatial: GeoTIFF, Shapefile, GeoJSON, KML
//! - Astronomical: FITS, VOTable
//!
//! These formats are commonly used in their respective fields and require
//! specialized handling for efficient processing and metadata preservation.

#![allow(dead_code)]
#![allow(missing_docs)]

/// Bioinformatics file formats
///
/// Provides support for common bioinformatics file formats:
/// - FASTA: Nucleotide and protein sequences
/// - FASTQ: Sequences with quality scores
/// - SAM/BAM: Sequence alignment data
/// - VCF: Variant Call Format for genomic variations
pub mod bioinformatics;

/// Geospatial file formats
///
/// Provides support for geographic and spatial data formats:
/// - GeoTIFF: Georeferenced raster images
/// - Shapefile: Vector geographic data
/// - GeoJSON: Geographic data in JSON format
/// - KML/KMZ: Keyhole Markup Language for geographic annotation
pub mod geospatial;

/// Astronomical file formats
///
/// Provides support for astronomy and astrophysics data formats:
/// - FITS: Flexible Image Transport System
/// - VOTable: Virtual Observatory Table format
/// - HDF5-based formats used in astronomy
pub mod astronomical;
