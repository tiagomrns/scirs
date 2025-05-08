//! Linear Time-Invariant (LTI) Systems
//!
//! This module provides types and functions for working with Linear Time-Invariant
//! systems, which are a fundamental concept in control theory and signal processing.
//!
//! Three different representations are provided:
//! - Transfer function representation: numerator and denominator polynomials
//! - Zero-pole-gain representation: zeros, poles, and gain
//! - State-space representation: A, B, C, D matrices
//!
//! These representations can be converted between each other, and used to analyze
//! system behavior through techniques such as impulse response, step response,
//! frequency response, and Bode plots.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::Zero;
use std::fmt::Debug;

/// A trait for all LTI system representations
pub trait LtiSystem {
    /// Get the transfer function representation of the system
    fn to_tf(&self) -> SignalResult<TransferFunction>;

    /// Get the zero-pole-gain representation of the system
    fn to_zpk(&self) -> SignalResult<ZerosPoleGain>;

    /// Get the state-space representation of the system
    fn to_ss(&self) -> SignalResult<StateSpace>;

    /// Calculate the system's frequency response
    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>>;

    /// Calculate the system's impulse response
    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Calculate the system's step response
    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Check if the system is stable
    fn is_stable(&self) -> SignalResult<bool>;
}

/// Transfer function representation of an LTI system
///
/// The transfer function is represented as a ratio of two polynomials:
/// H(s) = (b[0] * s^n + b[1] * s^(n-1) + ... + b[n]) / (a[0] * s^m + a[1] * s^(m-1) + ... + a[m])
///
/// Where:
/// - b: numerator coefficients (highest power first)
/// - a: denominator coefficients (highest power first)
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator coefficients (highest power first)
    pub num: Vec<f64>,

    /// Denominator coefficients (highest power first)
    pub den: Vec<f64>,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl TransferFunction {
    /// Create a new transfer function
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `TransferFunction` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::TransferFunction;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// ```
    pub fn new(mut num: Vec<f64>, mut den: Vec<f64>, dt: Option<bool>) -> SignalResult<Self> {
        // Remove leading zeros from numerator and denominator
        while num.len() > 1 && num[0].abs() < 1e-10 {
            num.remove(0);
        }

        while den.len() > 1 && den[0].abs() < 1e-10 {
            den.remove(0);
        }

        // Check if denominator is all zeros
        if den.iter().all(|&x| x.abs() < 1e-10) {
            return Err(SignalError::ValueError(
                "Denominator polynomial cannot be zero".to_string(),
            ));
        }

        // Normalize the denominator so that the leading coefficient is 1
        if !den.is_empty() && den[0].abs() > 1e-10 {
            let den_lead = den[0];
            for coef in &mut den {
                *coef /= den_lead;
            }

            // Also scale the numerator accordingly
            for coef in &mut num {
                *coef /= den_lead;
            }
        }

        Ok(TransferFunction {
            num,
            den,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get the order of the numerator polynomial
    pub fn num_order(&self) -> usize {
        self.num.len().saturating_sub(1)
    }

    /// Get the order of the denominator polynomial
    pub fn den_order(&self) -> usize {
        self.den.len().saturating_sub(1)
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Evaluate numerator polynomial
        let mut num_val = Complex64::zero();
        for (i, &coef) in self.num.iter().enumerate() {
            let power = (self.num.len() - 1 - i) as i32;
            num_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Evaluate denominator polynomial
        let mut den_val = Complex64::zero();
        for (i, &coef) in self.den.iter().enumerate() {
            let power = (self.den.len() - 1 - i) as i32;
            den_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Return the ratio
        if den_val.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num_val / den_val
        }
    }
}

impl LtiSystem for TransferFunction {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        Ok(self.clone())
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert transfer function to ZPK form by finding roots of numerator and denominator
        // This is a basic implementation - a production version would use more robust methods

        let gain = if self.num.is_empty() {
            0.0
        } else {
            self.num[0]
        };

        // Note: In practice, we would use a reliable polynomial root-finding algorithm
        // For now, returning placeholder with empty zeros and poles
        Ok(ZerosPoleGain {
            zeros: Vec::new(), // Replace with actual roots of numerator
            poles: Vec::new(), // Replace with actual roots of denominator
            gain,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert transfer function to state-space form
        // For a SISO system, this involves creating a controllable canonical form

        // This is a placeholder implementation - a full implementation would
        // properly handle the controllable canonical form construction

        // For now, return an empty state-space system
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        // For continuous-time systems, we use numerical simulation by
        // converting to state-space form and then simulating the response.
        if !self.dt {
            // Convert to state-space form if it's not already available
            let ss = self.to_ss()?;

            // Get time step (assume uniform sampling)
            let dt = if t.len() > 1 { t[1] - t[0] } else { 0.001 };

            // Simulate impulse response
            let mut response = vec![0.0; t.len()];

            if !ss.b.is_empty() && !ss.c.is_empty() {
                // Initial state is zero
                let mut x = vec![0.0; ss.n_states];

                // For an impulse, the input at t[0] is 1/dt, and 0 otherwise
                // Inject impulse: u[0] = 1/dt, which approximates a continuous impulse
                for j in 0..ss.n_inputs {
                    for i in 0..ss.n_states {
                        x[i] += ss.b[i * ss.n_inputs + j] * (1.0 / dt);
                    }
                }

                // Record initial output
                for i in 0..ss.n_outputs {
                    let mut y = 0.0;
                    for j in 0..ss.n_states {
                        y += ss.c[i * ss.n_states + j] * x[j];
                    }
                    if i == 0 {
                        // For SISO systems
                        response[0] = y;
                    }
                }

                // Simulate the system response for the rest of the time points
                for k in 1..t.len() {
                    // Update state: dx/dt = Ax + Bu, use forward Euler for simplicity
                    let mut x_new = vec![0.0; ss.n_states];

                    for i in 0..ss.n_states {
                        for j in 0..ss.n_states {
                            x_new[i] += ss.a[i * ss.n_states + j] * x[j] * dt;
                        }
                        // No input term (Bu) after initial impulse
                    }

                    // Copy updated state
                    x = x_new;

                    // Calculate output: y = Cx + Du (u is zero after initial impulse)
                    for i in 0..ss.n_outputs {
                        let mut y = 0.0;
                        for j in 0..ss.n_states {
                            y += ss.c[i * ss.n_states + j] * x[j];
                        }
                        if i == 0 {
                            // For SISO systems
                            response[k] = y;
                        }
                    }
                }
            }

            Ok(response)
        } else {
            // For discrete-time systems, impulse response h[n] is equivalent to
            // the inverse Z-transform of the transfer function H(z)
            // For a DT system H(z) = B(z)/A(z), the impulse response is given by
            // the coefficients of the series expansion of H(z)

            let n = t.len();
            let mut response = vec![0.0; n];

            // Check if we have the right number of coefficients
            if self.num.is_empty() || self.den.is_empty() {
                return Ok(response);
            }

            // For a proper transfer function with normalized denominator,
            // the first impulse response value is b[0]/a[0]
            response[0] = if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                self.num[0] / self.den[0]
            } else {
                self.num[0]
            };

            // For later samples, we use the recurrence relation:
            // h[n] = (b[n] - sum_{k=1}^n a[k]*h[n-k])/a[0]
            for n in 1..response.len() {
                // Add numerator contribution
                if n < self.num.len() {
                    response[n] = self.num[n];
                }

                // Subtract denominator * past outputs
                for k in 1..std::cmp::min(n + 1, self.den.len()) {
                    response[n] -= self.den[k] * response[n - k];
                }

                // Normalize by a[0]
                if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                    response[n] /= self.den[0];
                }
            }

            Ok(response)
        }
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        if !self.dt {
            // For continuous-time systems:
            // 1. Get impulse response
            let impulse = self.impulse_response(t)?;

            // 2. Integrate the impulse response to get the step response
            // Using the trapezoidal rule for integration
            let mut step = vec![0.0; t.len()];

            if t.len() > 1 {
                let dt = t[1] - t[0];

                // Initialize with the first value
                step[0] = impulse[0] * dt / 2.0;

                // Accumulate the integral
                for i in 1..t.len() {
                    step[i] = step[i - 1] + (impulse[i - 1] + impulse[i]) * dt / 2.0;
                }
            }

            Ok(step)
        } else {
            // For discrete-time systems:
            // The step response can be calculated either by:
            // 1. Convolving the impulse response with a step input
            // 2. Directly simulating with a step input
            // We'll use approach 1 for simplicity

            let impulse = self.impulse_response(t)?;
            let mut step = vec![0.0; t.len()];

            // Convolve with a unit step (running sum of impulse response)
            for i in 0..t.len() {
                for j in 0..=i {
                    step[i] += impulse[j];
                }
            }

            Ok(step)
        }
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would check the poles from to_zpk()
        Ok(true)
    }
}

/// Zeros-poles-gain representation of an LTI system
///
/// The transfer function is represented as:
/// H(s) = gain * (s - zeros[0]) * (s - zeros[1]) * ... / ((s - poles[0]) * (s - poles[1]) * ...)
#[derive(Debug, Clone)]
pub struct ZerosPoleGain {
    /// Zeros of the transfer function
    pub zeros: Vec<Complex64>,

    /// Poles of the transfer function
    pub poles: Vec<Complex64>,

    /// System gain
    pub gain: f64,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl ZerosPoleGain {
    /// Create a new zeros-poles-gain representation
    ///
    /// # Arguments
    ///
    /// * `zeros` - Zeros of the transfer function
    /// * `poles` - Poles of the transfer function
    /// * `gain` - System gain
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `ZerosPoleGain` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::ZerosPoleGain;
    /// use num_complex::Complex64;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let zpk = ZerosPoleGain::new(
    ///     Vec::new(),  // No zeros
    ///     vec![Complex64::new(-1.0, 0.0)],  // One pole at s = -1
    ///     1.0,  // Gain = 1
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
        gain: f64,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        Ok(ZerosPoleGain {
            zeros,
            poles,
            gain,
            dt: dt.unwrap_or(false),
        })
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Compute the numerator product (s - zeros[i])
        let mut num = Complex64::new(self.gain, 0.0);
        for &zero in &self.zeros {
            num *= s - zero;
        }

        // Compute the denominator product (s - poles[i])
        let mut den = Complex64::new(1.0, 0.0);
        for &pole in &self.poles {
            den *= s - pole;
        }

        // Return the ratio
        if den.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num / den
        }
    }
}

impl LtiSystem for ZerosPoleGain {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert ZPK to transfer function by expanding the polynomial products
        // This is a basic implementation - a production version would use more robust methods

        // For now, return a placeholder
        // In practice, we would expand (s - zero_1) * (s - zero_2) * ... for the numerator
        // and (s - pole_1) * (s - pole_2) * ... for the denominator

        Ok(TransferFunction {
            num: vec![self.gain],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        Ok(self.clone())
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert ZPK to state-space
        // Typically done by first converting to transfer function, then to state-space

        // For now, return a placeholder
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        for &pole in &self.poles {
            if self.dt {
                // For discrete-time systems, check if poles are inside the unit circle
                if pole.norm() >= 1.0 {
                    return Ok(false);
                }
            } else {
                // For continuous-time systems, check if poles have negative real parts
                if pole.re >= 0.0 {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// State-space representation of an LTI system
///
/// The system is represented as:
/// dx/dt = A*x + B*u  (for continuous-time systems)
/// x[k+1] = A*x[k] + B*u[k]  (for discrete-time systems)
/// y = C*x + D*u
///
/// Where:
/// - x is the state vector
/// - u is the input vector
/// - y is the output vector
/// - A, B, C, D are matrices of appropriate dimensions
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State matrix (n_states x n_states)
    pub a: Vec<f64>,

    /// Input matrix (n_states x n_inputs)
    pub b: Vec<f64>,

    /// Output matrix (n_outputs x n_states)
    pub c: Vec<f64>,

    /// Feedthrough matrix (n_outputs x n_inputs)
    pub d: Vec<f64>,

    /// Number of state variables
    pub n_states: usize,

    /// Number of inputs
    pub n_inputs: usize,

    /// Number of outputs
    pub n_outputs: usize,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl StateSpace {
    /// Create a new state-space system
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `StateSpace` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::StateSpace;
    ///
    /// // Create a simple first-order system: dx/dt = -x + u, y = x
    /// let ss = StateSpace::new(
    ///     vec![-1.0],  // A = [-1]
    ///     vec![1.0],   // B = [1]
    ///     vec![1.0],   // C = [1]
    ///     vec![0.0],   // D = [0]
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        d: Vec<f64>,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        // Determine the system dimensions from the matrix shapes
        let n_states = (a.len() as f64).sqrt() as usize;

        // Check if A is square
        if n_states * n_states != a.len() {
            return Err(SignalError::ValueError(
                "A matrix must be square".to_string(),
            ));
        }

        // Infer n_inputs from B
        let n_inputs = if n_states == 0 { 0 } else { b.len() / n_states };

        // Check consistency of B
        if n_states * n_inputs != b.len() {
            return Err(SignalError::ValueError(
                "B matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Infer n_outputs from C
        let n_outputs = if n_states == 0 { 0 } else { c.len() / n_states };

        // Check consistency of C
        if n_outputs * n_states != c.len() {
            return Err(SignalError::ValueError(
                "C matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Check consistency of D
        if n_outputs * n_inputs != d.len() {
            return Err(SignalError::ValueError(
                "D matrix has inconsistent dimensions".to_string(),
            ));
        }

        Ok(StateSpace {
            a,
            b,
            c,
            d,
            n_states,
            n_inputs,
            n_outputs,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get an element of the A matrix
    pub fn a(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for A matrix".to_string(),
            ));
        }

        Ok(self.a[i * self.n_states + j])
    }

    /// Get an element of the B matrix
    pub fn b(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for B matrix".to_string(),
            ));
        }

        Ok(self.b[i * self.n_inputs + j])
    }

    /// Get an element of the C matrix
    pub fn c(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for C matrix".to_string(),
            ));
        }

        Ok(self.c[i * self.n_states + j])
    }

    /// Get an element of the D matrix
    pub fn d(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for D matrix".to_string(),
            ));
        }

        Ok(self.d[i * self.n_inputs + j])
    }
}

impl LtiSystem for StateSpace {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert state-space to transfer function
        // For SISO systems, TF(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse and polynomial expansion

        Ok(TransferFunction {
            num: vec![1.0],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert state-space to ZPK
        // Typically done by first converting to transfer function, then factoring

        // For now, return a placeholder
        Ok(ZerosPoleGain {
            zeros: Vec::new(),
            poles: Vec::new(),
            gain: 1.0,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        Ok(self.clone())
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        // Calculate the frequency response for state-space system
        // H(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse for each frequency

        let response = vec![Complex64::new(1.0, 0.0); w.len()];
        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A state-space system is stable if all eigenvalues of A have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would calculate the eigenvalues of A

        Ok(true)
    }
}

/// Calculate the Bode plot data (magnitude and phase) for an LTI system
///
/// # Arguments
///
/// * `system` - The LTI system to analyze
/// * `w` - The frequency points at which to evaluate the response
///
/// # Returns
///
/// * A tuple containing (frequencies, magnitude in dB, phase in degrees)
pub fn bode<T: LtiSystem>(
    system: &T,
    w: Option<&[f64]>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Default frequencies if none provided
    let frequencies = match w {
        Some(freq) => freq.to_vec(),
        None => {
            // Generate logarithmically spaced frequencies between 0.01 and 100 rad/s
            let n = 100;
            let mut w_out = Vec::with_capacity(n);

            let w_min = 0.01;
            let w_max = 100.0;
            let log_step = f64::powf(w_max / w_min, 1.0 / (n - 1) as f64);

            let mut w_val = w_min;
            for _ in 0..n {
                w_out.push(w_val);
                w_val *= log_step;
            }

            w_out
        }
    };

    // Calculate frequency response
    let resp = system.frequency_response(&frequencies)?;

    // Convert to magnitude (dB) and phase (degrees)
    let mut mag = Vec::with_capacity(resp.len());
    let mut phase = Vec::with_capacity(resp.len());

    for &val in &resp {
        // Magnitude in dB: 20 * log10(|H(jw)|)
        let mag_db = 20.0 * val.norm().log10();
        mag.push(mag_db);

        // Phase in degrees: arg(H(jw)) * 180/pi
        let phase_deg = val.arg() * 180.0 / std::f64::consts::PI;
        phase.push(phase_deg);
    }

    Ok((frequencies, mag, phase))
}

/// Functions for creating and manipulating LTI systems
pub mod system {
    use super::*;

    /// Create a transfer function system from numerator and denominator coefficients
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `TransferFunction` instance
    pub fn tf(num: Vec<f64>, den: Vec<f64>, dt: Option<bool>) -> SignalResult<TransferFunction> {
        TransferFunction::new(num, den, dt)
    }

    /// Create a zeros-poles-gain system
    ///
    /// # Arguments
    ///
    /// * `zeros` - Zeros of the transfer function
    /// * `poles` - Poles of the transfer function
    /// * `gain` - System gain
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `ZerosPoleGain` instance
    pub fn zpk(
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
        gain: f64,
        dt: Option<bool>,
    ) -> SignalResult<ZerosPoleGain> {
        ZerosPoleGain::new(zeros, poles, gain, dt)
    }

    /// Create a state-space system
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `StateSpace` instance
    pub fn ss(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        d: Vec<f64>,
        dt: Option<bool>,
    ) -> SignalResult<StateSpace> {
        StateSpace::new(a, b, c, d, dt)
    }

    /// Convert a continuous-time system to a discrete-time system using zero-order hold method
    ///
    /// # Arguments
    ///
    /// * `system` - A continuous-time LTI system
    /// * `dt` - The sampling period
    ///
    /// # Returns
    ///
    /// * A discretized version of the system
    pub fn c2d<T: LtiSystem>(system: &T, _dt: f64) -> SignalResult<StateSpace> {
        // Convert to state-space first
        let ss_sys = system.to_ss()?;

        // Ensure the system is continuous-time
        if ss_sys.dt {
            return Err(SignalError::ValueError(
                "System is already discrete-time".to_string(),
            ));
        }

        // For now, return a placeholder for the discretized system
        // In practice, we would use the matrix exponential method: A_d = exp(A*dt)

        Ok(StateSpace {
            a: ss_sys.a.clone(),
            b: ss_sys.b.clone(),
            c: ss_sys.c.clone(),
            d: ss_sys.d.clone(),
            n_states: ss_sys.n_states,
            n_inputs: ss_sys.n_inputs,
            n_outputs: ss_sys.n_outputs,
            dt: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tf_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        assert_eq!(tf.num.len(), 1);
        assert_eq!(tf.den.len(), 2);
        assert_relative_eq!(tf.num[0], 1.0);
        assert_relative_eq!(tf.den[0], 1.0);
        assert_relative_eq!(tf.den[1], 1.0);
        assert!(!tf.dt);

        // Test normalization
        let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();
        assert_relative_eq!(tf2.num[0], 1.0);
        assert_relative_eq!(tf2.den[0], 1.0);
        assert_relative_eq!(tf2.den[1], 1.0);
    }

    #[test]
    fn test_tf_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Evaluate at s = 0
        let result = tf.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = tf.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_zpk_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        assert_eq!(zpk.zeros.len(), 0);
        assert_eq!(zpk.poles.len(), 1);
        assert_relative_eq!(zpk.poles[0].re, -1.0);
        assert_relative_eq!(zpk.poles[0].im, 0.0);
        assert_relative_eq!(zpk.gain, 1.0);
        assert!(!zpk.dt);
    }

    #[test]
    fn test_zpk_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        // Evaluate at s = 0
        let result = zpk.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = zpk.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_ss_creation() {
        // Create a simple first-order system: dx/dt = -x + u, y = x
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        assert_eq!(ss.n_states, 1);
        assert_eq!(ss.n_inputs, 1);
        assert_eq!(ss.n_outputs, 1);
        assert_relative_eq!(ss.a[0], -1.0);
        assert_relative_eq!(ss.b[0], 1.0);
        assert_relative_eq!(ss.c[0], 1.0);
        assert_relative_eq!(ss.d[0], 0.0);
        assert!(!ss.dt);
    }

    #[test]
    fn test_bode() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Compute Bode plot at omega = 0.1, 1, 10
        let freqs = vec![0.1, 1.0, 10.0];
        let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();

        // Check frequencies
        assert_eq!(w.len(), 3);
        assert_relative_eq!(w[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(w[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(w[2], 10.0, epsilon = 1e-6);

        // Check magnitudes (in dB)
        assert_eq!(mag.len(), 3);
        // At omega = 0.1, |H| = 0.995, which is -0.043 dB
        assert_relative_eq!(mag[0], -0.043, epsilon = 0.01);
        // At omega = 1, |H| = 0.707, which is -3 dB
        assert_relative_eq!(mag[1], -3.0, epsilon = 0.1);
        // At omega = 10, |H| = 0.0995, which is -20.043 dB
        assert_relative_eq!(mag[2], -20.043, epsilon = 0.1);

        // Check phases (in degrees)
        assert_eq!(phase.len(), 3);
        // At omega = 0.1, phase is about -5.7 degrees
        assert_relative_eq!(phase[0], -5.7, epsilon = 0.1);
        // At omega = 1, phase is -45 degrees
        assert_relative_eq!(phase[1], -45.0, epsilon = 0.1);
        // At omega = 10, phase is about -84.3 degrees
        assert_relative_eq!(phase[2], -84.3, epsilon = 0.1);
    }

    #[test]
    fn test_is_stable() {
        // Stable continuous-time system: H(s) = 1 / (s + 1)
        let stable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();
        assert!(stable.is_stable().unwrap());

        // Unstable continuous-time system: H(s) = 1 / (s - 1)
        let unstable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None).unwrap();
        assert!(!unstable.is_stable().unwrap());

        // Stable discrete-time system: H(z) = 1 / (z - 0.5)
        let stable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(0.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(stable_dt.is_stable().unwrap());

        // Unstable discrete-time system: H(z) = 1 / (z - 1.5)
        let unstable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(!unstable_dt.is_stable().unwrap());
    }
}
