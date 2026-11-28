//! An initial equation that reads a discrete a `when initial()` sets.
//!
//! MLS 3.7 §8.6: the initialization problem is every equation of the model
//! and its initial equations at once; a `when initial()` holds there. A
//! `fixed = false` parameter (or a free state) that an initial equation
//! defines from a value reading such a discrete (`p0 = v`, `v = u0 + ..`,
//! `when initial() then u0 = level`) was projected once, before the initial
//! event ran the `when`, from the discrete's declared start, and kept that:
//! `p0 = 1.5` here where the consistent start is `p0 = v = 3`.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

fn first(source: &str, model: &str, names: &[&str]) -> Vec<f64> {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, "initial_when.mo")
        .unwrap_or_else(|error| panic!("{model}: {error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            dt: Some(0.5),
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{model}: {error}"));
    names
        .iter()
        .map(|name| {
            let column = result
                .names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("no {name} in {:?}", result.names));
            result.data[column][0]
        })
        .collect()
}

/// `u0 = level = 3` at the start, so `v = 3` and `p0 = v = 3`.
#[test]
fn a_parameter_s_initial_equation_reads_what_an_initial_when_sets() {
    let source = "
model Held
  parameter Real p0(fixed = false);
  Real level(start = 3, fixed = true);
  discrete Real u0(start = 0);
  Real v = u0 + (level - u0)/2;
  Real p = p0;
initial equation
  p0 = v;
equation
  der(level) = 1;
  when initial() then
    u0 = level;
  end when;
end Held;";
    let values = first(source, "Held", &["u0", "v", "p"]);
    assert_eq!(values, vec![3.0, 3.0, 3.0], "u0, v, p at the start");
}

/// The same for a state without a fixed start: `x = v = 3`.
#[test]
fn a_state_s_initial_equation_reads_what_an_initial_when_sets() {
    let source = "
model HeldState
  Real x;
  Real level(start = 3, fixed = true);
  discrete Real u0(start = 0);
  Real v = u0 + (level - u0)/2;
initial equation
  x = v;
equation
  der(level) = 1;
  der(x) = 0;
  when initial() then
    u0 = level;
  end when;
end HeldState;";
    let values = first(source, "HeldState", &["u0", "v", "x"]);
    assert_eq!(values, vec![3.0, 3.0, 3.0], "u0, v, x at the start");
}
