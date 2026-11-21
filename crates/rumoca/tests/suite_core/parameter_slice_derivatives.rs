//! A slice of a parameter is constant, so an equation reading one can be
//! differentiated, as a Modelica.Fluid pipe's `pathLengths = cat(1, ...,
//! 0.5*(lengths[1:n-1] + lengths[2:n]), ...)` is when its momentum starts
//! steady (`der(m_flows) = zeros(m)` against `Is = m_flows .* pathLengths`).

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

#[test]
fn equation_over_parameter_slices_is_differentiated() {
    let source = "
model SliceLengths
  parameter Real lengths[2] = {1, 2};
  Real pathLengths[3];
  Real Is[3];
  Real mf[3];
equation
  pathLengths = cat(1, {0.5*lengths[1]}, 0.5*(lengths[1:1] + lengths[2:2]), {0.5*lengths[2]});
  der(Is) = ones(3) - mf;
  Is = {mf[i]*pathLengths[i] for i in 1:3};
initial equation
  der(mf) = zeros(3);
end SliceLengths;";
    let compiled = Compiler::new()
        .model("SliceLengths")
        .compile_str(source, "SliceLengths.mo")
        .unwrap_or_else(|error| panic!("SliceLengths compiles: {error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("SliceLengths simulates: {error}"));
    let last = |name: &str| {
        let index = result.names.iter().position(|n| n == name).expect(name);
        *result.data[index].last().expect("a sample")
    };
    // Steady from the start: mf = 1, Is = pathLengths.
    for (name, expected) in [
        ("mf[2]", 1.0),
        ("Is[1]", 0.5),
        ("Is[2]", 1.5),
        ("Is[3]", 1.0),
    ] {
        let value = last(name);
        assert!((value - expected).abs() < 1e-6, "{name} = {value}");
    }
}
