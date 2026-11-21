//! The MSL's examples of `Modelica.Media.Air.MoistAir` against psychrometric
//! values computed independently (IAPWS saturation pressure over water,
//! Wagner et al. 2011 sublimation pressure over ice, ASHRAE enthalpy
//! `h1+x = 1.006 t + x (2501 + 1.86 t)` kJ/kg dry air).
//!
//! Both examples declare `Medium.BaseProperties` through a non-replaceable
//! `package Medium = ...MoistAir`; `Xi(start = reference_X[1:nXi])` then has
//! to read MoistAir's `reference_X`, not PartialMedium's `fill(1/nX, nX)`
//! (which failed DAE construction in about half of the compiles).

use rumoca_sim::{SimOptions, SimResult, simulate_dae_with_diagnostics};

use super::msl_sim_regression::require_msl_compiler;

fn simulate(example: &str) -> SimResult {
    let source = format!("model Example\n  extends {example};\nend Example;");
    let compiled = require_msl_compiler()
        .model("Example")
        .compile_str(&source, "Example.mo")
        .unwrap_or_else(|error| panic!("{example} compiles: {error}"));
    simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            dt: Some(0.1),
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("{example} simulates: {error}"))
}

fn at(result: &SimResult, name: &str, time: f64) -> f64 {
    let column = result
        .names
        .iter()
        .position(|candidate| candidate == name)
        .unwrap_or_else(|| panic!("{name} in {:?}", result.names));
    let row = result
        .times
        .iter()
        .position(|t| (t - time).abs() < 1e-9)
        .unwrap_or_else(|| panic!("t = {time} in {:?}", result.times));
    result.data[column][row]
}

fn assert_close(value: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (value - expected).abs() <= tolerance,
        "{what} = {value}, expected {expected} +- {tolerance}"
    );
}

/// 5 % water at 1 bar heated from 274 K at 90 K/s: fog at the start
/// (phi = pw/psat = 7802 Pa / 650.0 Pa), dry at 364 K (psat 72.48 kPa).
#[test]
fn moist_air_example_gives_the_relative_humidity_of_its_composition() {
    for _ in 0..3 {
        let result = simulate("Modelica.Media.Examples.MoistAir");
        assert_close(at(&result, "medium.Xi[1]", 0.0), 0.05, 1e-12, "Xi[1](0)");
        assert_close(at(&result, "medium.phi", 0.0), 12.0028, 1e-3, "phi(0)");
        assert!(at(&result, "medium.X_liquid", 0.0) > 0.046, "fog at 274 K");
        assert_close(at(&result, "medium.phi", 1.0), 0.107_64, 1e-5, "phi(1)");
    }
}

/// The isotherms at 20 degC and -10 degC and the line of 50 % relative
/// humidity at x = 0.012 kg/kg dry air (t = 0.4). Below 0 degC the water
/// beyond saturation is ice: saturation over ice (259.90 Pa, over water it
/// would be 286.5 Pa) and its enthalpy 2050 t - 333000 J/kg.
#[test]
fn psychrometric_data_matches_psychrometric_values() {
    let result = simulate("Modelica.Media.Examples.PsychrometricData");
    assert_close(at(&result, "x", 0.4), 0.012, 1e-12, "x(0.4)");
    assert_close(
        at(&result, "medium_T[5].phi", 0.4),
        0.809_19,
        1e-5,
        "phi(20 degC)",
    );
    assert_close(
        at(&result, "hx_T[5]", 0.4),
        50_578.4,
        100.0,
        "h1+x(20 degC)",
    );
    assert_close(
        at(&result, "medium_T[2].p_steam_sat", 0.4),
        259.904,
        1e-3,
        "psat(-10 degC)",
    );
    assert_close(
        at(&result, "medium_T[2].phi", 0.4),
        7.2829,
        1e-3,
        "phi(-10 degC)",
    );
    assert_close(
        at(&result, "hx_T[2]", 0.4),
        -9_705.8,
        100.0,
        "h1+x(-10 degC)",
    );
    assert_close(
        at(&result, "medium_phi[5].phi", 0.4),
        0.5,
        1e-9,
        "phi line 0.5",
    );
    assert_close(
        at(&result, "medium_phi[5].T", 0.4),
        273.15 + 28.012,
        1e-3,
        "T(phi = 0.5)",
    );
}
