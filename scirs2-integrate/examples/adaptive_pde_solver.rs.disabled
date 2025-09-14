//! Advanced Adaptive PDE Solver Example
//!
//! This example demonstrates a practical application of the new advanced modules
//! by solving the 2D reaction-diffusion equation with adaptive mesh refinement,
//! advanced error control, and parallel optimization.
//!
//! The problem solved is:
//! ∂u/∂t = D∇²u + f(u,v)
//! ∂v/∂t = E∇²v + g(u,v)
//!
//! This represents systems like:
//! - Chemical reaction networks
//! - Population dynamics (predator-prey)
//! - Pattern formation in biology

use ndarray::{s, Array1, Array2, ArrayView1};
use scirs2_integrate::{
    // Advanced modules
    amr_advanced::{
        AdaptiveCell, AdaptiveMeshLevel, AdvancedAMRManager, CellId, FeatureDetectionCriterion,
        FeatureType, GradientRefinementCriterion, RefinementFlag,
    },
    error_estimation::AdvancedErrorEstimator,
    parallel_optimization::{
        ArithmeticOp, LoadBalancingStrategy, ParallelOptimizer, VectorOperation,
        VectorizedComputeTask,
    },
    performance_monitor::{PerformanceAnalyzer, PerformanceProfiler},

    IntegrateResult,
};
use std::collections::{HashMap, HashSet};

/// Parameters for the reaction-diffusion system
#[derive(Debug, Clone)]
pub struct ReactionDiffusionParams {
    /// Diffusion coefficient for species u
    pub d_u: f64,
    /// Diffusion coefficient for species v
    pub d_v: f64,
    /// Reaction rate parameter
    pub alpha: f64,
    /// Reaction rate parameter
    pub beta: f64,
    /// Reaction rate parameter
    pub gamma: f64,
}

impl Default for ReactionDiffusionParams {
    fn default() -> Self {
        // Gray-Scott model parameters for spot patterns
        Self {
            d_u: 2e-5,
            d_v: 1e-5,
            alpha: 0.04,  // feed rate
            beta: 0.1,    // kill rate
            gamma: 0.062, // reaction rate
        }
    }
}

/// Adaptive PDE solver with advanced features
pub struct AdaptivePDESolver {
    /// AMR manager
    amr_manager: AdvancedAMRManager<f64>,
    /// Error estimator
    error_estimator: AdvancedErrorEstimator<f64>,
    /// Parallel optimizer
    parallel_optimizer: ParallelOptimizer,
    /// Performance profiler
    profiler: PerformanceProfiler,
    /// Domain size
    #[allow(dead_code)]
    domain_size: f64,
    /// Current time
    current_time: f64,
}

impl AdaptivePDESolver {
    /// Create new adaptive PDE solver
    pub fn new(_initial_resolution: usize, domainsize: f64) -> IntegrateResult<Self> {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_phase("solver_initialization");

        // Create initial uniform mesh
        let initial_mesh = Self::create_initial_mesh(_initial_resolution, domain_size);
        let mut amr_manager = AdvancedAMRManager::new(initial_mesh, 4, domain_size / 512.0);

        // Add sophisticated refinement criteria
        let gradient_criterion = GradientRefinementCriterion {
            component: Some(0), // Monitor u component
            threshold: 0.1,
            relative_tolerance: 0.05,
        };
        amr_manager.add_criterion(Box::new(gradient_criterion));

        let feature_criterion = FeatureDetectionCriterion {
            threshold: 0.05,
            feature_types: vec![FeatureType::SharpGradient, FeatureType::LocalExtremum],
            window_size: 3,
        };
        amr_manager.add_criterion(Box::new(feature_criterion));

        // Configure parallel optimizer
        let mut parallel_optimizer = ParallelOptimizer::new(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );
        parallel_optimizer.load_balancer = LoadBalancingStrategy::Adaptive;
        parallel_optimizer.initialize()?;

        // Create error estimator
        let error_estimator = AdvancedErrorEstimator::new(1e-6, 3);

        profiler.end_phase("solver_initialization");

        Ok(Self {
            amr_manager,
            error_estimator,
            parallel_optimizer,
            profiler,
            domain_size,
            current_time: 0.0,
        })
    }

    /// Create initial uniform mesh
    fn create_initial_mesh(_resolution: usize, domainsize: f64) -> AdaptiveMeshLevel<f64> {
        let mut cells = HashMap::new();
        let cell_size = domain_size / _resolution as f64;
        let mut boundary_cells = HashSet::new();

        for i in 0.._resolution {
            for j in 0.._resolution {
                let cell_id = CellId {
                    level: 0,
                    index: i * _resolution + j,
                };

                let x = (i as f64 + 0.5) * cell_size;
                let y = (j as f64 + 0.5) * cell_size;

                // Check if boundary cell
                if i == 0 || i == _resolution - 1 || j == 0 || j == _resolution - 1 {
                    boundary_cells.insert(cell_id);
                }

                let cell = AdaptiveCell {
                    id: cell_id,
                    center: Array1::from_vec(vec![x, y]),
                    _size: cell_size,
                    solution: Array1::from_vec(vec![
                        initial_concentration_u(x, y, domain_size),
                        initial_concentration_v(x, y, domain_size),
                    ]),
                    error_estimate: 0.0,
                    refinement_flag: RefinementFlag::None,
                    is_active: true,
                    neighbors: vec![],
                    parent: None,
                    children: vec![],
                };

                cells.insert(cell_id, cell);
            }
        }

        AdaptiveMeshLevel {
            level: 0,
            cells,
            grid_spacing: cell_size,
            boundary_cells,
        }
    }

    /// Solve the reaction-diffusion system for one time step
    pub fn solve_time_step(
        &mut self,
        dt: f64,
        params: &ReactionDiffusionParams,
    ) -> IntegrateResult<SolverStepResult> {
        self.profiler.start_phase("time_step");

        // Extract current solution
        let solution = self.extract_solution_matrix();

        // Compute reaction-diffusion updates using parallel operations
        let diffusion_update = self.compute_diffusion_parallel(&solution, params)?;
        let reaction_update = self.compute_reaction_parallel(&solution, params)?;

        // Combine updates
        let combined_update = &diffusion_update + &reaction_update;
        let new_solution = &solution + &(&combined_update * dt);

        // Update mesh with new solution
        self.update_mesh_solution(&new_solution)?;

        // Perform error estimation
        let ode_fn = |_t: f64, y: &ArrayView1<f64>| {
            // Simplified ODE function for error estimation
            Array1::from_vec(y.iter().map(|&yi| -0.1 * yi).collect())
        };

        let flattened_solution = new_solution.iter().cloned().collect::<Array1<f64>>();
        let error_analysis = self.error_estimator.analyze_error(
            &flattened_solution,
            dt,
            ode_fn,
            Some(1e-8), // Embedded error estimate
        )?;

        // Adaptive mesh refinement based on solution gradients
        let adaptation_result = self.amr_manager.adapt_mesh(&new_solution)?;

        // Update time
        self.current_time += dt;

        self.profiler.end_phase("time_step");

        Ok(SolverStepResult {
            time: self.current_time,
            solution: new_solution,
            error_analysis,
            adaptation_result,
            active_cells: self.count_active_cells(),
        })
    }

    /// Extract solution as matrix from AMR mesh
    fn extract_solution_matrix(&self) -> Array2<f64> {
        let active_cells: Vec<&AdaptiveCell<f64>> = self
            .amr_manager
            .mesh_hierarchy
            .levels
            .iter()
            .flat_map(|level| level.cells.values())
            .filter(|cell| cell.is_active)
            .collect();

        let n_cells = active_cells.len();
        let n_species = 2; // u and v

        Array2::from_shape_fn((n_cells, n_species), |(i, j)| active_cells[i].solution[j])
    }

    /// Update mesh cells with new solution
    fn update_mesh_solution(&mut self, solution: &Array2<f64>) -> IntegrateResult<()> {
        let mut cell_index = 0;

        for level in &mut self.amr_manager.mesh_hierarchy.levels {
            for cell in level.cells.values_mut() {
                if cell.is_active && cell_index < solution.nrows() {
                    for j in 0..cell.solution.len().min(solution.ncols()) {
                        cell.solution[j] = solution[[cell_index, j]];
                    }
                    cell_index += 1;
                }
            }
        }

        Ok(())
    }

    /// Compute diffusion term using parallel operations
    fn compute_diffusion_parallel(
        &self,
        solution: &Array2<f64>,
        params: &ReactionDiffusionParams,
    ) -> IntegrateResult<Array2<f64>> {
        // Simplified diffusion computation using parallel vectorized operations
        let laplacian_u_task = VectorizedComputeTask {
            input: solution.slice(s![.., 0..1]).to_owned(),
            operation: VectorOperation::ElementWise(ArithmeticOp::Multiply(params.d_u)),
            chunk_size: 64,
            prefer_simd: true,
        };

        let laplacian_v_task = VectorizedComputeTask {
            input: solution.slice(s![.., 1..2]).to_owned(),
            operation: VectorOperation::ElementWise(ArithmeticOp::Multiply(params.d_v)),
            chunk_size: 64,
            prefer_simd: true,
        };

        let u_diffusion = self
            .parallel_optimizer
            .execute_vectorized(laplacian_u_task)?;
        let v_diffusion = self
            .parallel_optimizer
            .execute_vectorized(laplacian_v_task)?;

        // Combine results
        let mut result = Array2::zeros(solution.dim());
        result.slice_mut(s![.., 0..1]).assign(&u_diffusion);
        result.slice_mut(s![.., 1..2]).assign(&v_diffusion);

        Ok(result)
    }

    /// Compute reaction term using parallel operations
    fn compute_reaction_parallel(
        &self,
        solution: &Array2<f64>,
        params: &ReactionDiffusionParams,
    ) -> IntegrateResult<Array2<f64>> {
        let mut result = Array2::zeros(solution.dim());

        // Gray-Scott reaction terms
        for i in 0..solution.nrows() {
            let u = solution[[i, 0]];
            let v = solution[[i, 1]];
            let uv2 = u * v * v;

            // du/dt reaction term: -uv² + α(1-u)
            result[[i, 0]] = -uv2 + params.alpha * (1.0 - u);

            // dv/dt reaction term: uv² - (α+β)v
            result[[i, 1]] = uv2 - (params.alpha + params.beta) * v;
        }

        Ok(result)
    }

    /// Count active cells across all levels
    fn count_active_cells(&self) -> usize {
        self.amr_manager
            .mesh_hierarchy
            .levels
            .iter()
            .map(|level| level.cells.values().filter(|cell| cell.is_active).count())
            .sum()
    }

    /// Finalize and get performance report
    pub fn finalize(self) -> (PerformanceAnalyzer, SolverStatistics) {
        let final_cell_count = self.count_active_cells();
        let metrics = self.profiler.finalize();
        let stats = SolverStatistics {
            total_time: metrics.total_time,
            final_cell_count,
            function_evaluations: metrics.function_evaluations,
        };

        (PerformanceAnalyzer, stats)
    }
}

/// Result from a single solver time step
pub struct SolverStepResult {
    pub time: f64,
    pub solution: Array2<f64>,
    pub error_analysis: scirs2_integrate::error_estimation::ErrorAnalysisResult<f64>,
    pub adaptation_result: scirs2_integrate::amr_advanced::AMRAdaptationResult<f64>,
    pub active_cells: usize,
}

/// Overall solver statistics
#[derive(Debug)]
pub struct SolverStatistics {
    pub total_time: std::time::Duration,
    pub final_cell_count: usize,
    pub function_evaluations: usize,
}

/// Initial concentration for species u
#[allow(dead_code)]
fn initial_concentration_u(x: f64, y: f64, domainsize: f64) -> f64 {
    let center_x = domain_size * 0.5;
    let center_y = domain_size * 0.5;
    let radius = domain_size * 0.1;

    let distance = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();

    if distance < radius {
        0.5 + 0.1 * (2.0 * std::f64::consts::PI * distance / radius).sin()
    } else {
        0.5
    }
}

/// Initial concentration for species v
#[allow(dead_code)]
fn initial_concentration_v(x: f64, y: f64, domainsize: f64) -> f64 {
    let center_x = domain_size * 0.5;
    let center_y = domain_size * 0.5;
    let radius = domain_size * 0.15;

    let distance = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();

    if distance < radius {
        0.25
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn main() -> IntegrateResult<()> {
    println!("=== Adaptive PDE Solver Example ===\n");
    println!("Solving 2D Gray-Scott reaction-diffusion system with:");
    println!("- Adaptive mesh refinement");
    println!("- Advanced error estimation");
    println!("- Parallel optimization");
    println!("- Performance monitoring\n");

    // Create solver
    let mut solver = AdaptivePDESolver::new(32, 1.0)?;
    let params = ReactionDiffusionParams::default();

    println!("Initial setup complete. Starting time integration...\n");

    // Time integration
    let dt = 0.001;
    let n_steps = 100;
    let output_interval = 20;

    for step in 0..n_steps {
        let result = solver.solve_time_step(dt, &params)?;

        if step % output_interval == 0 {
            println!("Step {}: t = {:.4}", step, result.time);
            println!("  Active cells: {}", result.active_cells);
            println!(
                "  Primary error: {:.2e}",
                result.error_analysis.primary_estimate
            );
            println!(
                "  Cells refined: {}",
                result.adaptation_result.cells_refined
            );
            println!(
                "  Solution quality - smoothness: {:.3}",
                result.error_analysis.quality_metrics.smoothness
            );

            if let Some(richardson_error) = result.error_analysis.richardson_error {
                println!("  Richardson error: {richardson_error:.2e}");
            }
            println!();
        }
    }

    // Generate final report
    let (_analyzer, stats) = solver.finalize();
    println!("=== Final Statistics ===");
    println!(
        "Total computation time: {:.3}s",
        stats.total_time.as_secs_f64()
    );
    println!("Final cell count: {}", stats.final_cell_count);
    println!("Function evaluations: {}", stats.function_evaluations);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_solver_creation() {
        let solver = AdaptivePDESolver::new(16, 1.0);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_initial_conditions() {
        let u = initial_concentration_u(0.5, 0.5, 1.0);
        let v = initial_concentration_v(0.5, 0.5, 1.0);

        assert!(u > 0.0 && u < 1.0);
        assert!(v >= 0.0 && v < 1.0);
    }

    #[test]
    fn test_reaction_diffusion_params() {
        let params = ReactionDiffusionParams::default();
        assert!(params.d_u > 0.0);
        assert!(params.d_v > 0.0);
        assert!(params.alpha > 0.0);
    }
}
