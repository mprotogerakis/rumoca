//! `Modelica.Media.R134a.R134a_ph`, the MSL's two-phase refrigerant: its
//! Helmholtz coefficient records give their values in extends clauses with a
//! field sized by a sibling (`extends EOSIdealCoeff(nc = 5, a = {...})`),
//! `FindInterval` sizes its search by `scalar(size(breaks))`, its saturation
//! record is redeclared by extension and set through local modifiers, and
//! its unrolled Newton iterations lower to blocks beyond an AArch64
//! conditional branch's range. Heated at 2.93 bar from 200 kJ/kg, the
//! refrigerant boils at 0 degC (IIR reference: h' = 200 kJ/kg at 0 degC).

use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

use super::msl_sim_regression::require_msl_compiler;

const SOURCE: &str = "model R134aHeated
  package M = Modelica.Media.R134a.R134a_ph;
  parameter Real p = 2.93e5;
  Real h(start = 2.0e5, fixed = true);
  M.ThermodynamicState st = M.setState_ph(p, h);
  Real T = M.temperature(st);
  Real x = M.vapourQuality(st);
  Real Tsat = M.saturationTemperature(p);
equation
  der(h) = 1e3;
end R134aHeated;
";

#[test]
fn r134a_boils_at_its_saturation_temperature() {
    let compiled = require_msl_compiler()
        .model("R134aHeated")
        .compile_str(SOURCE, "R134aHeated.mo")
        .unwrap_or_else(|error| panic!("{error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 250.0,
            dt: Some(25.0),
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let at = |name: &str, time: f64| {
        let column = result
            .names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or_else(|| panic!("{name} in {:?}", result.names));
        let row = result
            .times
            .iter()
            .position(|t| (t - time).abs() < 1e-6)
            .unwrap_or_else(|| panic!("t = {time} in {:?}", result.times));
        result.data[column][row]
    };
    let tsat = at("Tsat", 100.0);
    assert!((tsat - 273.17).abs() < 0.01, "Tsat = {tsat}");
    // Saturated liquid at the start, wet steam at constant T while it boils,
    // the quality linear in h, superheated vapour at 450 kJ/kg.
    assert!((at("T", 0.0) - 273.15).abs() < 0.02);
    assert!((at("T", 100.0) - tsat).abs() < 1e-9);
    assert!(
        (at("x", 100.0) - 0.5034).abs() < 1e-3,
        "x = {}",
        at("x", 100.0)
    );
    assert!((at("x", 175.0) - 0.8811).abs() < 1e-3);
    assert!(
        (at("T", 250.0) - 330.31).abs() < 0.05,
        "T = {}",
        at("T", 250.0)
    );
}
