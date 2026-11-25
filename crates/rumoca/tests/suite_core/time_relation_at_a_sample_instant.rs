//! A relation of time that crosses exactly at a sample's instant.
//!
//! `b = time > 6` beside `when sample(0, 2)`: the scheduled time event of the
//! relation and the clock's tick share the instant 6 and are one stop. The
//! stop took the tick's pre mode alone, the relation's right limit was not
//! observed (at 6 itself `6 > 6` is false), and `b` stayed false until the
//! next event of any kind, two seconds on, without a word. `time >= 6` and a
//! relation read inside a derivative were right.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

fn run(relation: &str, period: &str) -> (f64, f64) {
    let source = format!(
        "model Late\n  Real late(start = 0, fixed = true);\n  discrete Real k(start = 0, fixed = true);\n  Boolean b;\nequation\n  b = {relation};\n  der(late) = if b then 1 else 0;\n  when sample(0, {period}) then\n    k = pre(k) + 1;\n  end when;\nend Late;\n"
    );
    let compiled = Compiler::new()
        .model("Late")
        .compile_str(&source, "late.mo")
        .unwrap_or_else(|error| panic!("{relation}: {error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 10.0,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{relation}: {error}"));
    let last = |name: &str| {
        let column = result
            .names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no {name} in {:?}", result.names));
        *result.data[column].last().expect("a sample")
    };
    (last("late"), last("k"))
}

/// From 6 s on `late` grows: 4 at 10 s; the clock ticked at 0, 2, .., 10.
#[test]
fn a_strict_relation_of_time_switches_at_a_coinciding_tick() {
    let (late, ticks) = run("time > 6", "2");
    assert!((late - 4.0).abs() < 1e-6 && ticks == 6.0, "late {late}, ticks {ticks}");
}

/// The same with the tick at 0, 3, 6, 9.
#[test]
fn a_strict_relation_of_time_switches_at_another_clock_s_tick() {
    let (late, ticks) = run("time > 6", "3");
    assert!((late - 4.0).abs() < 1e-6 && ticks == 4.0, "late {late}, ticks {ticks}");
}

/// What was right already stays right: no coincidence, and `>=`.
#[test]
fn a_relation_of_time_beside_a_clock_is_right_where_it_was() {
    for (relation, period, ticks) in [("time > 6", "4", 3.0), ("time >= 6", "2", 6.0), ("time > 5", "2", 6.0)] {
        let (late, k) = run(relation, period);
        let want = if relation.ends_with('5') { 5.0 } else { 4.0 };
        assert!((late - want).abs() < 1e-6 && k == ticks, "{relation}, period {period}: late {late}, ticks {k}");
    }
}

/// The tick's own work is done once: its print reports once per tick, though
/// the relation that crosses at the tick's instant is read in a second pass.
#[test]
fn a_tick_s_print_reports_once_where_a_relation_of_time_crosses() {
    let source = r#"
package Streams
  impure function print
    input String string = "";
    input String fileName = "";
  external "C" ModelicaInternal_print(string, fileName);
  end print;
end Streams;

model Once
  Real late(start = 0, fixed = true);
  discrete Real k(start = 0, fixed = true);
  Boolean b;
equation
  b = time > 0.5;
  der(late) = if b then 1 else 0;
algorithm
  when sample(0.5, 0.5) then
    k := pre(k) + 1;
    Streams.print("tick");
  end when;
end Once;
"#;
    let compiled = Compiler::new()
        .model("Once")
        .compile_str(source, "once.mo")
        .unwrap_or_else(|error| panic!("{error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            dt: Some(0.125),
            ..SimOptions::default()
        },
    )
    .expect("the model simulates");
    let messages = result
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "WX002")
        .map(|diagnostic| (diagnostic.time, diagnostic.message.as_str()))
        .collect::<Vec<_>>();
    let last = |name: &str| {
        let column = result.names.iter().position(|n| n == name).expect(name);
        *result.data[column].last().expect("a sample")
    };
    assert_eq!(messages, vec![(0.5, "tick"), (1.0, "tick")], "one report per tick");
    assert!(
        (last("late") - 0.5).abs() < 1e-6 && last("k") == 2.0,
        "late {}, ticks {}",
        last("late"),
        last("k")
    );
}

/// A tick that terminates the simulation ends it: the relation that crosses
/// at the same instant is not read past it, and what it would set off there
/// (a print of its own) does not happen.
#[test]
fn a_terminating_tick_is_not_followed_by_the_relation_s_pass() {
    let source = r#"
package Streams
  impure function print
    input String string = "";
    input String fileName = "";
  external "C" ModelicaInternal_print(string, fileName);
  end print;
end Streams;

model Ends
  Real x(start = 0, fixed = true);
  Boolean b;
equation
  b = time > 0.5;
  der(x) = 1;
  when sample(0.5, 0.5) then
    terminate("done at the tick");
  end when;
algorithm
  when b then
    Streams.print("past the tick");
  end when;
end Ends;
"#;
    let compiled = Compiler::new()
        .model("Ends")
        .compile_str(source, "ends.mo")
        .unwrap_or_else(|error| panic!("{error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 1.0,
            dt: Some(0.125),
            ..SimOptions::default()
        },
    )
    .expect("the model simulates to its termination");
    let prints = result
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "WX002")
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    let end = *result.times.last().expect("a sample");
    assert!(prints.is_empty() && (end - 0.5).abs() < 1e-9, "prints {prints:?}, ended at {end}");
}
