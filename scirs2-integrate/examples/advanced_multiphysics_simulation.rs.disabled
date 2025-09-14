//! Advanced Multi-Physics Simulation Example
//!
//! This example demonstrates the integration of multiple specialized solvers
//! from scirs2-integrate, including quantum mechanics, fluid dynamics, and
//! financial modeling with advanced analysis capabilities.

use ndarray::Array1;
use scirs2_integrate::{
    // Analysis tools
    BifurcationAnalyzer,
    BifurcationPoint,
    FinanceMethod,
    FinancialOption,
    FluidBoundaryCondition,
    FluidState,
    HarmonicOscillator,
    // ODE solving - removed unused imports
    // Visualization
    //visualization::{VisualizationEngine, PhaseSpacePlot},
    // Error handling
    IntegrateResult as Result,
    NavierStokesParams,
    // Fluid dynamics
    NavierStokesSolver,
    OptionStyle,
    OptionType,
    QuantumState,
    SchrodingerMethod,
    // Quantum mechanics
    SchrodingerSolver,
    StabilityAnalyzer,
    StabilityResult,
    // Financial modeling
    StochasticPDESolver,
    VolatilityModel,
};

/// Multi-physics simulation coordinator
pub struct MultiPhysicsSimulation {
    /// Quantum system parameters
    pub quantum_params: QuantumSystemParams,
    /// Fluid system parameters
    pub fluid_params: FluidSystemParams,
    /// Financial system parameters
    pub financial_params: FinancialSystemParams,
    /// Coupling parameters between systems
    pub coupling_params: CouplingParams,
}

/// Quantum system configuration
#[derive(Debug, Clone)]
pub struct QuantumSystemParams {
    /// Number of grid points
    pub n_points: usize,
    /// Time step
    pub dt: f64,
    /// Mass of particle
    pub mass: f64,
    /// Potential parameters
    pub potential_strength: f64,
}

/// Fluid system configuration
#[derive(Debug, Clone)]
pub struct FluidSystemParams {
    /// Grid size
    pub grid_size: (usize, usize),
    /// Kinematic viscosity
    pub viscosity: f64,
    /// Reynolds number
    pub reynolds_number: f64,
    /// Lid velocity for cavity flow
    pub lid_velocity: f64,
}

/// Financial system configuration
#[derive(Debug, Clone)]
pub struct FinancialSystemParams {
    /// Initial asset price
    pub spot_price: f64,
    /// Strike price
    pub strike_price: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Volatility
    pub volatility: f64,
    /// Time to maturity
    pub maturity: f64,
}

/// Coupling parameters between different physics
#[derive(Debug, Clone)]
pub struct CouplingParams {
    /// Quantum-fluid coupling strength
    pub quantum_fluid_coupling: f64,
    /// Fluid-financial coupling strength
    pub fluid_financial_coupling: f64,
    /// Financial-quantum coupling strength
    pub financial_quantum_coupling: f64,
}

impl Default for QuantumSystemParams {
    fn default() -> Self {
        Self {
            n_points: 200,
            dt: 0.001,
            mass: 1.0,
            potential_strength: 1.0,
        }
    }
}

impl Default for FluidSystemParams {
    fn default() -> Self {
        Self {
            grid_size: (64, 64),
            viscosity: 0.01,
            reynolds_number: 1000.0,
            lid_velocity: 1.0,
        }
    }
}

impl Default for FinancialSystemParams {
    fn default() -> Self {
        Self {
            spot_price: 100.0,
            strike_price: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            maturity: 1.0,
        }
    }
}

impl Default for CouplingParams {
    fn default() -> Self {
        Self {
            quantum_fluid_coupling: 0.1,
            fluid_financial_coupling: 0.05,
            financial_quantum_coupling: 0.02,
        }
    }
}

impl Default for MultiPhysicsSimulation {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiPhysicsSimulation {
    /// Create a new multi-physics simulation
    pub fn new() -> Self {
        Self {
            quantum_params: QuantumSystemParams::default(),
            fluid_params: FluidSystemParams::default(),
            financial_params: FinancialSystemParams::default(),
            coupling_params: CouplingParams::default(),
        }
    }

    /// Run the complete multi-physics simulation
    pub fn run_simulation(&self, simulationtime: f64) -> Result<SimulationResults> {
        println!("Starting multi-physics simulation...");

        // 1. Quantum mechanics simulation
        println!("Solving quantum system...");
        let quantum_results = self.solve_quantum_system(simulationtime)?;

        // 2. Fluid dynamics simulation
        println!("Solving fluid system...");
        let fluid_results = self.solve_fluid_system(simulationtime)?;

        // 3. Financial modeling
        println!("Solving financial system...");
        let financial_results = self.solve_financial_system()?;

        // 4. Coupled system analysis
        println!("Analyzing coupled system...");
        let coupled_analysis =
            self.analyze_coupled_system(&quantum_results, &fluid_results, &financial_results)?;

        // 5. Bifurcation analysis on the coupled system
        println!("Performing bifurcation analysis...");
        let bifurcation_results = self.perform_bifurcation_analysis()?;

        // 6. Stability analysis
        println!("Performing stability analysis...");
        let stability_results = self.perform_stability_analysis()?;

        Ok(SimulationResults {
            quantum_results,
            fluid_results,
            financial_results,
            coupled_analysis,
            bifurcation_results,
            stability_results,
        })
    }

    /// Solve the quantum mechanical system
    fn solve_quantum_system(&self, simulationtime: f64) -> Result<QuantumResults> {
        // Create quantum potential (harmonic oscillator with coupling effects)
        let potential = Box::new(HarmonicOscillator {
            k: self.quantum_params.potential_strength,
            x0: 0.0,
        });

        // Create Schrödinger solver
        let solver = SchrodingerSolver::new(
            self.quantum_params.n_points,
            self.quantum_params.dt,
            potential,
            SchrodingerMethod::SplitOperator,
        );

        // Create spatial grid
        let x = Array1::linspace(-10.0, 10.0, self.quantum_params.n_points);

        // Create initial Gaussian wave packet
        let initial_state = SchrodingerSolver::gaussian_wave_packet(
            &x,
            -2.0, // Initial position
            1.0,  // Width
            5.0,  // Initial momentum
            self.quantum_params.mass,
        );

        // Solve _time evolution
        let states = solver.solve_time_dependent(&initial_state, simulationtime)?;

        // Calculate observables over _time
        let mut position_expectation = Vec::new();
        let mut momentum_expectation = Vec::new();
        let mut energy = Vec::new();

        for state in &states {
            position_expectation.push(state.expectation_position());
            momentum_expectation.push(state.expectation_momentum());

            // Calculate kinetic + potential energy
            let ke = state.expectation_momentum().powi(2) / (2.0 * self.quantum_params.mass);
            let pe =
                self.quantum_params.potential_strength * state.expectation_position().powi(2) / 2.0;
            energy.push(ke + pe);
        }

        Ok(QuantumResults {
            states,
            position_expectation,
            momentum_expectation,
            energy,
        })
    }

    /// Solve the fluid dynamics system
    fn solve_fluid_system(&self, simulationtime: f64) -> Result<FluidResults> {
        // Create Navier-Stokes parameters
        let ns_params = NavierStokesParams {
            nu: self.fluid_params.viscosity,
            rho: 1.0,
            dt: 0.01,
            max_pressure_iter: 100,
            pressure_tol: 1e-6,
            semi_lagrangian: false,
        };

        // Create solver with boundary conditions (lid-driven cavity)
        let solver = NavierStokesSolver::new(
            ns_params,
            (
                FluidBoundaryCondition::NoSlip,
                FluidBoundaryCondition::NoSlip,
            ),
            (
                FluidBoundaryCondition::NoSlip,
                FluidBoundaryCondition::NoSlip,
            ),
        );

        // Create initial state (lid-driven cavity)
        let initial_state = NavierStokesSolver::lid_driven_cavity(
            self.fluid_params.grid_size.0,
            self.fluid_params.grid_size.1,
            self.fluid_params.lid_velocity,
        );

        // Solve fluid system
        let save_interval = 10;
        let states = solver.solve_2d(initial_state, simulationtime, save_interval)?;

        // Calculate flow properties
        let mut kinetic_energy = Vec::new();
        let mut enstrophy = Vec::new();
        let mut max_velocity = Vec::new();

        for state in &states {
            let u = &state.velocity[0];
            let v = &state.velocity[1];

            // Kinetic energy
            let ke: f64 = u
                .iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| 0.5 * (ui * ui + vi * vi))
                .sum::<f64>()
                * state.dx
                * state.dy;
            kinetic_energy.push(ke);

            // Enstrophy (vorticity squared)
            let mut enstrophy_val = 0.0;
            let (ny, nx) = u.dim();
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let dvdx = (v[[j, i + 1]] - v[[j, i - 1]]) / (2.0 * state.dx);
                    let dudy = (u[[j + 1, i]] - u[[j - 1, i]]) / (2.0 * state.dy);
                    let vorticity = dvdx - dudy;
                    enstrophy_val += vorticity * vorticity;
                }
            }
            enstrophy.push(enstrophy_val);

            // Maximum velocity magnitude
            let max_vel = u
                .iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| (ui * ui + vi * vi).sqrt())
                .fold(0.0, f64::max);
            max_velocity.push(max_vel);
        }

        Ok(FluidResults {
            states,
            kinetic_energy,
            enstrophy,
            max_velocity,
        })
    }

    /// Solve the financial system
    fn solve_financial_system(&self) -> Result<FinancialResults> {
        // Create financial option
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: self.financial_params.strike_price,
            maturity: self.financial_params.maturity,
            spot: self.financial_params.spot_price,
            risk_free_rate: self.financial_params.risk_free_rate,
            dividend_yield: 0.0,
        };

        // Create volatility model
        let volatility_model = VolatilityModel::Constant(self.financial_params.volatility);

        // Create PDE solver
        let solver = StochasticPDESolver::new(
            100, // n_asset
            50,  // n_time
            volatility_model,
            FinanceMethod::FiniteDifference,
        );

        // Price the option
        let option_price = solver.price_option(&option)?;

        // Calculate Greeks
        let greeks = solver.calculate_greeks(&option)?;

        // Monte Carlo pricing for comparison
        let mc_solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(self.financial_params.volatility),
            FinanceMethod::MonteCarlo {
                n_paths: 10000,
                antithetic: true,
            },
        );
        let mc_price = mc_solver.price_option(&option)?;

        // Sensitivity analysis - price vs volatility
        let mut volatilities = Vec::new();
        let mut prices = Vec::new();

        for vol in (10..=50).step_by(5) {
            let vol_val = vol as f64 / 100.0;
            volatilities.push(vol_val);

            let vol_model = VolatilityModel::Constant(vol_val);
            let vol_solver =
                StochasticPDESolver::new(50, 30, vol_model, FinanceMethod::FiniteDifference);
            let price = vol_solver.price_option(&option)?;
            prices.push(price);
        }

        Ok(FinancialResults {
            option_price,
            monte_carlo_price: mc_price,
            greeks,
            volatility_sensitivity: (volatilities, prices),
        })
    }

    /// Analyze the coupled multi-physics system
    fn analyze_coupled_system(
        &self,
        quantum: &QuantumResults,
        fluid: &FluidResults,
        financial: &FinancialResults,
    ) -> Result<CoupledAnalysis> {
        // Calculate cross-correlations between different physics
        let q_f_correlation = self.calculate_correlation(&quantum.energy, &fluid.kinetic_energy);

        let f_fin_correlation =
            self.calculate_correlation(&fluid.max_velocity, &financial.volatility_sensitivity.1);

        let fin_q_correlation =
            if quantum.position_expectation.len() >= financial.volatility_sensitivity.1.len() {
                self.calculate_correlation(
                    &quantum.position_expectation[..financial.volatility_sensitivity.1.len()],
                    &financial.volatility_sensitivity.1,
                )
            } else {
                0.0
            };

        // Energy transfer analysis
        let total_quantum_energy: f64 = quantum.energy.iter().sum();
        let total_fluid_energy: f64 = fluid.kinetic_energy.iter().sum();
        let total_financial_value = financial.option_price;

        let energy_distribution = EnergyDistribution {
            quantum_fraction: total_quantum_energy
                / (total_quantum_energy + total_fluid_energy + total_financial_value),
            fluid_fraction: total_fluid_energy
                / (total_quantum_energy + total_fluid_energy + total_financial_value),
            financial_fraction: total_financial_value
                / (total_quantum_energy + total_fluid_energy + total_financial_value),
        };

        // Coupling strength analysis
        let coupling_effectiveness = CouplingEffectiveness {
            quantum_fluid: q_f_correlation.abs() * self.coupling_params.quantum_fluid_coupling,
            fluid_financial: f_fin_correlation.abs()
                * self.coupling_params.fluid_financial_coupling,
            financial_quantum: fin_q_correlation.abs()
                * self.coupling_params.financial_quantum_coupling,
        };

        Ok(CoupledAnalysis {
            cross_correlations: CrossCorrelations {
                quantum_fluid: q_f_correlation,
                fluid_financial: f_fin_correlation,
                financial_quantum: fin_q_correlation,
            },
            energy_distribution,
            coupling_effectiveness,
        })
    }

    /// Calculate correlation coefficient between two time series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }

        let x_slice = &x[..n];
        let y_slice = &y[..n];

        let x_mean: f64 = x_slice.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_slice.iter().sum::<f64>() / n as f64;

        let numerator: f64 = x_slice
            .iter()
            .zip(y_slice.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_var: f64 = x_slice.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let y_var: f64 = y_slice.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let denominator = (x_var * y_var).sqrt();

        if denominator > 1e-12 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Perform bifurcation analysis on a simplified coupled system
    fn perform_bifurcation_analysis(&self) -> Result<Vec<BifurcationPoint>> {
        // Extract coupling parameters to avoid lifetime issues
        let quantum_fluid_coupling = self.coupling_params.quantum_fluid_coupling;
        let fluid_financial_coupling = self.coupling_params.fluid_financial_coupling;

        // Define a simplified coupled system for bifurcation analysis
        // This represents the interaction between quantum and fluid systems
        let coupled_system = move |x: &Array1<f64>, param: f64| -> Array1<f64> {
            let quantum_state = x[0];
            let fluid_velocity = x[1];

            // Coupled equations with parameter-dependent interaction
            let dq_dt = -quantum_state + param * fluid_velocity * quantum_fluid_coupling;
            let df_dt = -fluid_velocity + param * quantum_state * fluid_financial_coupling;

            Array1::from_vec(vec![dq_dt, df_dt])
        };

        let analyzer = BifurcationAnalyzer::new(
            2,           // 2D system
            (-2.0, 2.0), // Parameter range
            20,          // Number of parameter samples
        );

        let initial_guess = Array1::from_vec(vec![0.1, 0.1]);
        analyzer.continuation_analysis(coupled_system, &initial_guess)
    }

    /// Perform stability analysis on the coupled system
    fn perform_stability_analysis(&self) -> Result<StabilityResult> {
        // Extract coupling parameters to avoid lifetime issues
        let quantum_fluid_coupling = self.coupling_params.quantum_fluid_coupling;
        let fluid_financial_coupling = self.coupling_params.fluid_financial_coupling;

        // Define the autonomous version of the coupled system
        let coupled_autonomous = move |x: &Array1<f64>| -> Array1<f64> {
            let quantum_state = x[0];
            let fluid_velocity = x[1];

            // Fixed parameter value for stability analysis
            let param = 1.0;

            let dq_dt = -quantum_state + param * fluid_velocity * quantum_fluid_coupling;
            let df_dt = -fluid_velocity + param * quantum_state * fluid_financial_coupling;

            Array1::from_vec(vec![dq_dt, df_dt])
        };

        let analyzer = StabilityAnalyzer::new(2);
        let domain = vec![(-5.0, 5.0), (-5.0, 5.0)];

        analyzer.analyze_stability(coupled_autonomous, &domain)
    }
}

/// Results from quantum mechanics simulation
#[derive(Debug, Clone)]
pub struct QuantumResults {
    pub states: Vec<QuantumState>,
    pub position_expectation: Vec<f64>,
    pub momentum_expectation: Vec<f64>,
    pub energy: Vec<f64>,
}

/// Results from fluid dynamics simulation
#[derive(Debug, Clone)]
pub struct FluidResults {
    pub states: Vec<FluidState>,
    pub kinetic_energy: Vec<f64>,
    pub enstrophy: Vec<f64>,
    pub max_velocity: Vec<f64>,
}

/// Results from financial modeling
#[derive(Debug, Clone)]
pub struct FinancialResults {
    pub option_price: f64,
    pub monte_carlo_price: f64,
    pub greeks: scirs2_integrate::Greeks,
    pub volatility_sensitivity: (Vec<f64>, Vec<f64>), // (volatilities, prices)
}

/// Analysis of coupled system interactions
#[derive(Debug, Clone)]
pub struct CoupledAnalysis {
    pub cross_correlations: CrossCorrelations,
    pub energy_distribution: EnergyDistribution,
    pub coupling_effectiveness: CouplingEffectiveness,
}

/// Cross-correlation coefficients between subsystems
#[derive(Debug, Clone)]
pub struct CrossCorrelations {
    pub quantum_fluid: f64,
    pub fluid_financial: f64,
    pub financial_quantum: f64,
}

/// Energy distribution across subsystems
#[derive(Debug, Clone)]
pub struct EnergyDistribution {
    pub quantum_fraction: f64,
    pub fluid_fraction: f64,
    pub financial_fraction: f64,
}

/// Effectiveness of coupling between subsystems
#[derive(Debug, Clone)]
pub struct CouplingEffectiveness {
    pub quantum_fluid: f64,
    pub fluid_financial: f64,
    pub financial_quantum: f64,
}

/// Complete simulation results
#[derive(Debug, Clone)]
pub struct SimulationResults {
    pub quantum_results: QuantumResults,
    pub fluid_results: FluidResults,
    pub financial_results: FinancialResults,
    pub coupled_analysis: CoupledAnalysis,
    pub bifurcation_results: Vec<BifurcationPoint>,
    pub stability_results: StabilityResult,
}

impl SimulationResults {
    /// Generate a comprehensive summary report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== MULTI-PHYSICS SIMULATION REPORT ===\n\n");

        // Quantum system summary
        report.push_str("QUANTUM MECHANICS RESULTS:\n");
        report.push_str(&format!(
            "  - Number of time steps: {}\n",
            self.quantum_results.states.len()
        ));
        report.push_str(&format!(
            "  - Final position expectation: {:.6}\n",
            self.quantum_results
                .position_expectation
                .last()
                .unwrap_or(&0.0)
        ));
        report.push_str(&format!(
            "  - Final momentum expectation: {:.6}\n",
            self.quantum_results
                .momentum_expectation
                .last()
                .unwrap_or(&0.0)
        ));
        report.push_str(&format!(
            "  - Average energy: {:.6}\n\n",
            self.quantum_results.energy.iter().sum::<f64>()
                / self.quantum_results.energy.len() as f64
        ));

        // Fluid system summary
        report.push_str("FLUID DYNAMICS RESULTS:\n");
        report.push_str(&format!(
            "  - Number of time steps: {}\n",
            self.fluid_results.states.len()
        ));
        report.push_str(&format!(
            "  - Final kinetic energy: {:.6}\n",
            self.fluid_results.kinetic_energy.last().unwrap_or(&0.0)
        ));
        report.push_str(&format!(
            "  - Maximum velocity: {:.6}\n",
            self.fluid_results
                .max_velocity
                .iter()
                .fold(0.0f64, |a, &b| a.max(b))
        ));
        report.push_str(&format!(
            "  - Final enstrophy: {:.6}\n\n",
            self.fluid_results.enstrophy.last().unwrap_or(&0.0)
        ));

        // Financial system summary
        report.push_str("FINANCIAL MODELING RESULTS:\n");
        report.push_str(&format!(
            "  - Option price (PDE): {:.6}\n",
            self.financial_results.option_price
        ));
        report.push_str(&format!(
            "  - Option price (Monte Carlo): {:.6}\n",
            self.financial_results.monte_carlo_price
        ));
        report.push_str(&format!(
            "  - Delta: {:.6}\n",
            self.financial_results.greeks.delta
        ));
        report.push_str(&format!(
            "  - Gamma: {:.6}\n",
            self.financial_results.greeks.gamma
        ));
        report.push_str(&format!(
            "  - Vega: {:.6}\n\n",
            self.financial_results.greeks.vega
        ));

        // Coupled analysis summary
        report.push_str("COUPLED SYSTEM ANALYSIS:\n");
        report.push_str(&format!(
            "  - Quantum-Fluid correlation: {:.6}\n",
            self.coupled_analysis.cross_correlations.quantum_fluid
        ));
        report.push_str(&format!(
            "  - Fluid-Financial correlation: {:.6}\n",
            self.coupled_analysis.cross_correlations.fluid_financial
        ));
        report.push_str(&format!(
            "  - Financial-Quantum correlation: {:.6}\n",
            self.coupled_analysis.cross_correlations.financial_quantum
        ));
        report.push_str(&format!(
            "  - Energy distribution - Quantum: {:.3}%, Fluid: {:.3}%, Financial: {:.3}%\n\n",
            self.coupled_analysis.energy_distribution.quantum_fraction * 100.0,
            self.coupled_analysis.energy_distribution.fluid_fraction * 100.0,
            self.coupled_analysis.energy_distribution.financial_fraction * 100.0
        ));

        // Bifurcation analysis summary
        report.push_str("BIFURCATION ANALYSIS:\n");
        report.push_str(&format!(
            "  - Number of bifurcation points found: {}\n",
            self.bifurcation_results.len()
        ));
        for (i, bif_point) in self.bifurcation_results.iter().enumerate() {
            report.push_str(&format!(
                "    Bifurcation {}: {:?} at parameter = {:.6}\n",
                i + 1,
                bif_point.bifurcation_type,
                bif_point.parameter_value
            ));
        }
        report.push('\n');

        // Stability analysis summary
        report.push_str("STABILITY ANALYSIS:\n");
        report.push_str(&format!(
            "  - Number of fixed points: {}\n",
            self.stability_results.fixed_points.len()
        ));
        for (i, fp) in self.stability_results.fixed_points.iter().enumerate() {
            report.push_str(&format!(
                "    Fixed point {}: {:?} at ({:.6}, {:.6})\n",
                i + 1,
                fp.stability,
                fp.location[0],
                fp.location[1]
            ));
        }

        report.push_str("\n=== END OF REPORT ===\n");
        report
    }
}

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Advanced Multi-Physics Simulation Example");
    println!("==========================================");

    // Create and configure the simulation
    let mut simulation = MultiPhysicsSimulation::new();

    // Customize parameters
    simulation.quantum_params.potential_strength = 0.5;
    simulation.fluid_params.reynolds_number = 100.0;
    simulation.financial_params.volatility = 0.25;
    simulation.coupling_params.quantum_fluid_coupling = 0.2;

    // Run the simulation
    let simulationtime = 2.0;
    let results = simulation.run_simulation(simulationtime)?;

    // Generate and display report
    let report = results.generate_report();
    println!("{report}");

    // Additional analysis
    println!("=== ADDITIONAL INSIGHTS ===");

    // Check for strong coupling
    let strong_coupling_threshold = 0.5;
    let correlations = &results.coupled_analysis.cross_correlations;

    if correlations.quantum_fluid.abs() > strong_coupling_threshold {
        println!("⚠️  Strong quantum-fluid coupling detected!");
    }

    if correlations.fluid_financial.abs() > strong_coupling_threshold {
        println!("⚠️  Strong fluid-financial coupling detected!");
    }

    if correlations.financial_quantum.abs() > strong_coupling_threshold {
        println!("⚠️  Strong financial-quantum coupling detected!");
    }

    // Stability assessment
    let stable_points = results
        .stability_results
        .fixed_points
        .iter()
        .filter(|fp| fp.stability == scirs2_integrate::StabilityType::Stable)
        .count();

    println!("System has {stable_points} stable fixed points");

    if !results.bifurcation_results.is_empty() {
        println!("System exhibits bifurcations - parameter sensitivity detected");
    }

    println!("\nSimulation completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiphysics_simulation() {
        let simulation = MultiPhysicsSimulation::new();

        // Test that we can create the simulation without errors
        assert_eq!(simulation.quantum_params.n_points, 200);
        assert_eq!(simulation.fluid_params.grid_size, (64, 64));
        assert_eq!(simulation.financial_params.spot_price, 100.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let simulation = MultiPhysicsSimulation::new();

        // Test correlation with identical series
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let correlation = simulation.calculate_correlation(&x, &x);
        assert!((correlation - 1.0).abs() < 1e-10);

        // Test correlation with anti-correlated series
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation = simulation.calculate_correlation(&x, &y);
        assert!((correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_components() {
        let simulation = MultiPhysicsSimulation::new();

        // Test individual components
        let quantum_result = simulation.solve_quantum_system(0.1);
        assert!(quantum_result.is_ok());

        let financial_result = simulation.solve_financial_system();
        assert!(financial_result.is_ok());
    }
}
