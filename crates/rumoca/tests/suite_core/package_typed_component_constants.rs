//! A component whose type is named through a package (`Medium.B b` with
//! `package Medium = M1`, or `M1.B b`) reads the package constants of the
//! class it inherits from that package (MLS 3.7 §5.3.2, §7.1).
//!
//! This is the shape of `Modelica.Media.Examples.MoistAir`: `Xi(start =
//! reference_X[1:nXi])` is declared in `PartialMedium.BaseProperties`, which
//! `MoistAir.BaseProperties` extends, and MoistAir sets `reference_X` in its
//! extends clause. With two media of the same base each component must take
//! its own medium's value, not the base's default or another medium's.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

const SOURCE: &str = r#"
package PM
  constant String names[:] = {"unusable"};
  final constant Integer nS = size(names, 1);
  constant Integer n = nS;
  constant Real ref[n] = fill(1/n, n);
  replaceable partial model B
    Real x[n](start = ref[1:n], each fixed = true);
  end B;
end PM;
partial package Mixture
  extends PM;
end Mixture;
package M1
  extends Mixture(names = {"a", "b"}, ref = {0.25, 0.75});
  redeclare model extends B
  equation
    der(x) = 0*x;
  end B;
end M1;
package M2
  extends Mixture(names = {"a", "b"}, ref = {0.125, 0.875});
  redeclare model extends B
  equation
    der(x) = 0*x;
  end B;
end M2;
model TwoMedia
  package Medium1 = M1;
  package Medium2 = M2;
  Medium1.B b1;
  Medium2.B b2;
  M1.B b3;
  Real r = Medium1.ref[1] + Medium2.ref[1];
end TwoMedia;
"#;

#[test]
fn a_component_typed_through_a_package_reads_that_package_constants() {
    let compiled = Compiler::new()
        .model("TwoMedia")
        .compile_str(SOURCE, "TwoMedia.mo")
        .unwrap_or_else(|error| panic!("TwoMedia compiles: {error:?}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            ..SimOptions::default()
        },
    )
    .expect("TwoMedia simulates");
    let start = |name: &str| {
        let column = result
            .names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or_else(|| panic!("{name} in {:?}", result.names));
        result.data[column][0]
    };
    for (name, expected) in [
        ("b1.x[1]", 0.25),
        ("b1.x[2]", 0.75),
        ("b2.x[1]", 0.125),
        ("b2.x[2]", 0.875),
        ("b3.x[1]", 0.25),
        ("b3.x[2]", 0.75),
    ] {
        assert_eq!(start(name), expected, "{name}(0)");
    }
}
