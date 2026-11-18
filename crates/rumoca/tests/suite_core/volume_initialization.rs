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
