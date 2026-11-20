//! What a Modelica.Fluid volume needs to start from its medium's start values:
//! the initial equations of a pipe's component array (`der(mediums.T) =
//! zeros(n)`, `mediums.Xi = fill(X_start[1:nXi], n)`).

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

fn final_values(model: &str, source: &str, names: &[&str]) -> Vec<f64> {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("{model} compiles: {error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("{model} simulates: {error}"));
    names
        .iter()
        .map(|name| {
            let index = result.names.iter().position(|n| n == name).expect(name);
            *result.data[index].last().expect("a sample")
        })
        .collect()
}

/// `der(b.T)` over a component array is the array of the elements'
/// derivatives, and `fill` of a slice spells copies of the slice.
#[test]
fn component_array_initial_equations_differentiate_and_fill_elementwise() {
    let source = "
model ArrayInitial
  model B
    Real T(start = 0);
    Real Xi[2](each start = 1);
  equation
    der(T) = 1 - T;
    der(Xi) = {0.5, 0.25} - Xi;
  end B;
  parameter Real X_start[3] = {0.1, 0.2, 0.7};
  B b[3];
initial equation
  der(b.T) = zeros(3);
  b.Xi = fill(X_start[1:2], 3);
end ArrayInitial;";
    let values = final_values("ArrayInitial", source, &["b[2].T", "b[2].Xi[1]"]);
    assert!((values[0] - 1.0).abs() < 1e-6, "steady T = {}", values[0]);
    let expected = 0.5 - 0.4 * (-1.0f64).exp();
    assert!((values[1] - expected).abs() < 1e-5, "Xi = {}", values[1]);
}

/// A medium function's `h_T = smooth(0, if T < Tlimit then ... else ...)`
/// (DryAirNasa): past its certificate the guard, inside a function, cannot
/// own an event, so the conditional differentiates branch-wise without one.
/// `U = 4*p` while `p < 5`, so the preferred state decays as `exp(-t)` from
/// its start; a basis withheld for `U` would start at zero instead.
#[test]
fn a_preferred_state_is_reached_through_smooth_inside_a_function() {
    let source = "
model PreferThroughFunction
  function h
    input Real x;
    output Real y;
  algorithm
    y := smooth(0, if x < 5 then x else 2*x - 5);
  end h;
  Real p(stateSelect = StateSelect.prefer, start = 1);
  Real U;
equation
  U = 3*p + h(p);
  der(U) = -U;
end PreferThroughFunction;";
    let p = final_values("PreferThroughFunction", source, &["p"])[0];
    assert!((p - (-1.0f64).exp()).abs() < 1e-5, "p(1) = {p}");
}
