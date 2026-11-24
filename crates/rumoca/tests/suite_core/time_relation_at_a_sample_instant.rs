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
