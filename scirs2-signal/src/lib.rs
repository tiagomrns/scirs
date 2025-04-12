//! Signal processing module
//!
//! This module provides implementations of various signal processing algorithms
//! for filtering, convolution, and spectral analysis.
//!
//! ## Overview
//!
//! * Filtering: FIR and IIR filters, filter design
//! * Convolution and correlation
//! * Spectral analysis and periodograms
//! * Peak finding and signal measurements
//! * Waveform generation and processing
//!
//! ## Examples
//!
//! ```
//! use scirs2_signal::filter::butter;
//! use scirs2_signal::filter::filtfilt;
//!
//! // Generate a simple signal and apply a Butterworth filter
//! // let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! // let fs = 100.0;  // Sample rate in Hz
//! // let cutoff = 10.0;  // Cutoff frequency in Hz
//! //
//! // let (b, a) = butter(4, cutoff / (fs / 2.0), "lowpass").unwrap();
//! // let filtered = filtfilt(&b, &a, &signal).unwrap();
//! ```

// Export error types
pub mod error;
pub use error::{SignalError, SignalResult};

// Signal processing module structure
pub mod convolve;
pub mod filter;
pub mod peak;
pub mod resample;
pub mod spectral;
pub mod waveforms;

// Re-export commonly used functions
pub use convolve::{convolve, correlate, deconvolve};
pub use filter::{bessel, butter, cheby1, cheby2, ellip, filtfilt, firwin, lfilter};
pub use peak::{find_peaks, peak_prominences, peak_widths};
pub use spectral::{periodogram, spectrogram, stft, welch};
pub use waveforms::{chirp, gausspulse, sawtooth, square};

// Signal measurement functions
pub mod measurements;
pub use measurements::{peak_to_peak, peak_to_rms, rms, snr, thd};

// Utility functions for signal processing
pub mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
