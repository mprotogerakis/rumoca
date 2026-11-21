//! An array of components sized by a parameter of a sibling component, as in
//! `FixedHeatFlow heat[pipe.nNodes]` (Modelica.Fluid.Examples.BranchingDynamicPipes).

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

/// `a[p.nNodes]` reads p's modifier, `b[p.n]` p's default `n = nNodes` with
/// p's modified `nNodes`, `c[q.n]` q's plain default. Each array is expanded
/// into its elements, each with its own equation.
#[test]
fn component_array_sized_by_sibling_member_expands_into_elements() {
    let source = "
model MemberDimensions
  model C
    parameter Real k = 1;
    Real x(start = 1, fixed = true);
  equation
    der(x) = -k*x*(1 + 0.5*(x - 1));
  end C;
  model P
    parameter Integer nNodes = 2;
    final parameter Integer n = nNodes;
  end P;
  P p(nNodes = 3);
  P q;
  C a[p.nNodes](k = {1, 2, 3});
  C b[p.n];
  C c[q.n];
end MemberDimensions;";
    let compiled = Compiler::new()
        .model("MemberDimensions")
        .compile_str(source, "MemberDimensions.mo")
        .unwrap_or_else(|error| panic!("MemberDimensions compiles: {error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("MemberDimensions simulates: {error}"));
    let last = |name: &str| {
        let index = result
            .names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{name} in {:?}", result.names));
        *result.data[index].last().expect("a sample")
    };
    for name in ["a[3].x", "b[3].x", "c[2].x"] {
        last(name);
    }
    for name in ["a[4].x", "b[4].x", "c[3].x"] {
        assert!(!result.names.iter().any(|n| n == name), "{name} exists");
    }
    // der(x) = -k*x*(1 + (x - 1)/2) = -k*x*(1 + x)/2: x = 1/(2*exp(k*t/2) - 1).
    let exact = |k: f64| 1.0 / (2.0 * (k / 2.0).exp() - 1.0);
    for (name, k) in [
        ("a[1].x", 1.0),
        ("a[3].x", 3.0),
        ("b[2].x", 1.0),
        ("c[2].x", 1.0),
    ] {
        let value = last(name);
        assert!(
            (value - exact(k)).abs() < 1e-4,
            "{name} = {value}, not {}",
            exact(k)
        );
    }
}
