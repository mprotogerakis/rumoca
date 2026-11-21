//! MLS §7.2.5: a non-`each` modifier on an array of components built from
//! `fill`, `ones` or `zeros` gives each element its own element, as
//! `alpha = -1e-2*ones(pipe2.n)` on Modelica.Fluid.Examples.
//! BranchingDynamicPipes' heat sources.
use rumoca::Compiler;
use rumoca_sim::{SimOptions, eval_dae_at};

#[test]
fn constructor_modifiers_distribute_over_component_array() {
    let source = "
model Probe
  model E
    parameter Real a = 0;
    parameter Real b = 0;
    Real x(start = 1, fixed = true);
  equation
    der(x) = a + b*x;
  end E;
  parameter Integer n = 2;
  E e[n](a = -2*ones(n), b = 3*fill(2, n), x(start = 0.5*fill(4, n)));
  E f[n](a = 2*ones(n) - zeros(n));
end Probe;";
    let compiled = Compiler::new()
        .model("Probe")
        .compile_str(source, "Probe.mo")
        .unwrap_or_else(|error| panic!("Probe compiles: {error}"));
    let probe = eval_dae_at(&compiled.dae, &SimOptions::default(), &[], 0.0)
        .expect("evaluate initial derivatives");
    assert!(probe.report.error.is_none(), "{:?}", probe.report.error);
    let derivative = |name: &str| {
        let slot = format!("der({name})");
        probe
            .report
            .derivatives
            .iter()
            .find(|entry| entry.name == slot)
            .unwrap_or_else(|| panic!("{slot} in {:?}", probe.report.derivatives))
            .value
    };
    // e: -2 + 6*2; f: 2.
    for (name, expected) in [("e[1].x", 10.0), ("e[2].x", 10.0), ("f[2].x", 2.0)] {
        assert_eq!(derivative(name), expected, "der({name})");
    }
}
