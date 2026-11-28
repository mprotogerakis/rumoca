//! A source that steps exactly at a sample instant, read by that sample.
//!
//! MLS 3.7 §8.5: the equations active at an event instant hold together. At
//! 4 s `when time >= 4 then u = 1` and `when sample(0, 2) then y = u` are both
//! active, so `y = 1` there. The tick's rows executed first, over the values
//! their pass entered with, and `y` took the `u` of before the instant: the
//! sampled value was one period late (true from 6 s). A step written as a
//! relation in an equation (`u = if time >= 4 then 1 else 0`) was right.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

/// The values of `names` in the last row recorded at or before `time`.
fn at(source: &str, model: &str, time: f64, names: &[&str]) -> Vec<f64> {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, "step_at_sample.mo")
        .unwrap_or_else(|error| panic!("{model}: {error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 8.0,
            dt: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{model}: {error}"));
    let row = result
        .times
        .iter()
        .rposition(|sample| *sample <= time)
        .expect("a row at or before the time");
    names
        .iter()
        .map(|name| {
            let column = result
                .names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("no {name} in {:?}", result.names));
            result.data[column][row]
        })
        .collect()
}

const WHEN: &str = "
model Step
  discrete Real u(start = 0, fixed = true);
  discrete Real n(start = 0, fixed = true);
  discrete Real y(start = -1, fixed = true);
  discrete Real yp(start = -1, fixed = true);
  discrete Real k(start = 0, fixed = true);
equation
  when time >= 4 then
    u = 1;
    n = pre(n) + 1;
  end when;
  when sample(0, 2) then
    y = u;
    yp = pre(u);
    k = pre(k) + 1;
  end when;
end Step;";

/// The tick at 4 s reads the stepped `u`; its `pre(u)` is the value before
/// the instant; the step's body ran once and the tick counted once.
#[test]
fn a_tick_reads_the_value_a_when_on_time_sets_at_its_instant() {
    assert_eq!(at(WHEN, "Step", 3.5, &["u", "y", "k"]), vec![0.0, 0.0, 2.0]);
    assert_eq!(
        at(WHEN, "Step", 4.5, &["u", "n", "y", "yp", "k"]),
        vec![1.0, 1.0, 1.0, 0.0, 3.0]
    );
    assert_eq!(
        at(WHEN, "Step", 6.5, &["n", "y", "yp", "k"]),
        vec![1.0, 1.0, 1.0, 4.0]
    );
}

/// The same through an algebraic of a deadline held in a discrete, as
/// `Modelica.Blocks.Sources.TimeTable` steps (`y = a*time + b`,
/// `when {time >= pre(nextEvent), initial()}`).
#[test]
fn a_tick_reads_the_step_of_a_table_like_source_at_its_instant() {
    let source = "
model Table
  discrete Real b(start = 0);
  discrete Real nextEvent(start = 0, fixed = true);
  Real u = 0*time + b;
  discrete Real y(start = -1, fixed = true);
equation
  when {time >= pre(nextEvent), initial()} then
    b = if time >= 4 then 1 else 0;
    nextEvent = if time >= 4 then 1e60 else 4;
  end when;
  when sample(0, 2) then
    y = u;
  end when;
end Table;";
    assert_eq!(at(source, "Table", 3.5, &["u", "y"]), vec![0.0, 0.0]);
    assert_eq!(at(source, "Table", 4.5, &["u", "y"]), vec![1.0, 1.0]);
}

/// `time > 4` is false at 4 s itself: the tick there reads the value before
/// the step, and the step follows at the same instant.
#[test]
fn a_strict_relation_of_time_steps_after_the_tick_at_its_instant() {
    let source = WHEN.replace("time >= 4", "time > 4");
    assert_eq!(
        at(&source, "Step", 4.5, &["u", "n", "y", "k"]),
        vec![1.0, 1.0, 0.0, 3.0]
    );
    assert_eq!(at(&source, "Step", 6.5, &["y", "k"]), vec![1.0, 4.0]);
}
