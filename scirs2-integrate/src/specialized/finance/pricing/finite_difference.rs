//! Finite difference methods for option pricing

use crate::error::IntegrateResult;
use crate::specialized::finance::models::VolatilityModel;
use crate::specialized::finance::solvers::StochasticPDESolver;
use crate::specialized::finance::types::{FinancialOption, OptionStyle, OptionType};
use ndarray::Array2;

/// Main finite difference pricing function
pub fn price_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
) -> IntegrateResult<f64> {
    match &solver.volatility_model {
        VolatilityModel::Constant(sigma) => black_scholes_finite_difference(solver, option, *sigma),
        VolatilityModel::Heston {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => heston_finite_difference(solver, option, *v0, *theta, *kappa, *sigma, *rho),
        // Add other models as needed
        _ => Err(crate::error::IntegrateError::ValueError(
            "Finite difference not implemented for this volatility model".to_string(),
        )),
    }
}

/// Black-Scholes finite difference solver
pub fn black_scholes_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    sigma: f64,
) -> IntegrateResult<f64> {
    let dt = option.maturity / (solver.n_time - 1) as f64;
    let s_max = option.spot * 3.0;
    let ds = s_max / (solver.n_asset - 1) as f64;

    // Initialize grid
    let mut v = Array2::zeros((solver.n_time, solver.n_asset));

    // Terminal condition
    for i in 0..solver.n_asset {
        let s = i as f64 * ds;
        v[[solver.n_time - 1, i]] = solver.payoff(option, s);
    }

    // Boundary conditions
    match option.option_type {
        OptionType::Call => {
            for t_idx in 0..solver.n_time {
                let t = (solver.n_time - 1 - t_idx) as f64 * dt;
                v[[t_idx, 0]] = 0.0;
                v[[t_idx, solver.n_asset - 1]] =
                    s_max - option.strike * (-option.risk_free_rate * t).exp();
            }
        }
        OptionType::Put => {
            for t_idx in 0..solver.n_time {
                let t = (solver.n_time - 1 - t_idx) as f64 * dt;
                v[[t_idx, 0]] = option.strike * (-option.risk_free_rate * t).exp();
                v[[t_idx, solver.n_asset - 1]] = 0.0;
            }
        }
    }

    // Backward induction
    for t_idx in (0..solver.n_time - 1).rev() {
        for i in 1..solver.n_asset - 1 {
            let s = i as f64 * ds;

            // Finite difference coefficients
            let a = 0.5
                * dt
                * (sigma * sigma * (i as f64) * (i as f64) - option.risk_free_rate * i as f64);
            let b = 1.0 - dt * (sigma * sigma * (i as f64) * (i as f64) + option.risk_free_rate);
            let c = 0.5
                * dt
                * (sigma * sigma * (i as f64) * (i as f64) + option.risk_free_rate * i as f64);

            // Explicit scheme
            let v_new =
                a * v[[t_idx + 1, i - 1]] + b * v[[t_idx + 1, i]] + c * v[[t_idx + 1, i + 1]];

            // For American options, apply early exercise condition
            if option.option_style == OptionStyle::American {
                v[[t_idx, i]] = v_new.max(solver.payoff(option, s));
            } else {
                v[[t_idx, i]] = v_new;
            }
        }
    }

    // Interpolate to get option value at initial spot price
    let spot_idx = ((option.spot / ds) as usize).min(solver.n_asset - 2);
    let weight = (option.spot - spot_idx as f64 * ds) / ds;

    Ok(v[[0, spot_idx]] * (1.0 - weight) + v[[0, spot_idx + 1]] * weight)
}

/// Heston model finite difference solver
pub fn heston_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
) -> IntegrateResult<f64> {
    // TODO: Implement Heston finite difference
    // This is a placeholder implementation
    Err(crate::error::IntegrateError::NotImplementedError(
        "Heston finite difference implementation in progress".to_string(),
    ))
}
