//! What a Modelica.Fluid volume needs to start from its medium's start values:
//! the medium's `StateSelect.prefer` states reached through `min` and
//! `smooth(0, if ...)`, translation-time selections that read an `outer`
//! element's modified value, and the initial equations of a pipe's component
//! array (`der(mediums.T) = zeros(n)`, `mediums.Xi = fill(X_start[1:nXi], n)`).

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

/// `U = 4*p` while `p < 2`, so the preferred state decays as `exp(-t)` from
/// its start; a basis withheld for `U` would start at zero instead.
#[test]
fn a_preferred_state_is_reached_through_min_and_a_smooth_conditional() {
    let source = "
model PreferThroughMin
  Real p(stateSelect = StateSelect.prefer, start = 1);
  Real U;
equation
  U = min(p, 2)*3 + smooth(0, if p < 5 then p else 2*p - 5);
  der(U) = -U;
end PreferThroughMin;";
    let p = final_values("PreferThroughMin", source, &["p"])[0];
    assert!((p - (-1.0f64).exp()).abs() < 1e-5, "p(1) = {p}");
}

/// An initial if-equation whose branches differ in size is selected at
/// translation; an `outer` element's value is its `inner`'s modification.
#[test]
fn an_initial_selection_reads_the_inner_value_of_an_outer_element() {
    let source = "
model OuterSelection
  model System
    parameter Boolean fixInitial = false;
    parameter Integer mode = 1;
  end System;
  inner System system(fixInitial = true, mode = 2);
  model Part
    outer System system;
    parameter Boolean fix = system.fixInitial;
    Real x(start = 0);
    Real y(start = 0);
  equation
    der(x) = -x;
    der(y) = -y;
  initial equation
    if fix then
      x = 5;
    end if;
    if system.mode == 2 then
      y = 5;
    end if;
  end Part;
  Part c;
end OuterSelection;";
    for value in final_values("OuterSelection", source, &["c.x", "c.y"]) {
        assert!((value - 5.0 * (-1.0f64).exp()).abs() < 1e-5, "{value}");
    }
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
