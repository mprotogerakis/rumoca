//! State-space linearization through algebraic loops and the steady-state
//! trim, against closed forms derived by hand.

use std::time::Instant;

use rumoca::Compiler;
use rumoca_sim::{LinearizationRequest, SimOptions, TrimRequest, linearize_dae, trim_dae};

fn compile(source: &str, model: &str) -> rumoca::CompilationResult {
    Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("{model} should compile: {error:?}"))
}

fn assert_close(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!(
        (got - want).abs() <= tolerance * (1.0 + want.abs()),
        "{what}: got {got}, want {want}"
    );
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(ToString::to_string).collect()
}

// ---------------------------------------------------------------------------
// An output through a nonlinear loop of two unknowns:
//   a = x + u - b^3,  b = (a + a^3) / 2,  der(x) = -b,  y = a.
// At x = u = 1 the loop settles at a = b = 1, and its tangent is
//   da = ds - 3 db,  db = 2 da   =>   da = ds / 7,  db = 2 ds / 7
// with ds = dx + du, so A = B = -2/7 and C = D = 1/7 for `y`.
// ---------------------------------------------------------------------------

const LOOP: &str = r#"
model Loop
  input Real u = 1;
  Real x(start = 1, fixed = true);
  Real a(start = 0.9);
  Real b(start = 0.9);
  output Real y;
equation
  a = x + u - b ^ 3;
  b = (a + a ^ 3) / 2;
  der(x) = -b;
  y = a;
end Loop;
"#;

fn loop_request() -> LinearizationRequest {
    LinearizationRequest {
        input_names: names(&["u"]),
        output_names: names(&["y", "b", "x", "u"]),
        ..LinearizationRequest::default()
    }
}

#[test]
fn an_output_through_a_nonlinear_loop_has_the_implicit_tangent() {
    let result = compile(LOOP, "Loop");
    let lin = linearize_dae(&result.dae, &SimOptions::default(), &loop_request())
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(lin.states, ["x"]);
    assert_eq!(lin.inputs, ["u"]);
    assert_eq!(lin.outputs, ["y", "b", "x", "u"]);
    assert_close(lin.a[0][0], -2.0 / 7.0, 1.0e-10, "A");
    assert_close(lin.b[0][0], -2.0 / 7.0, 1.0e-10, "B");
    // y = a, b, the state x itself (a unit row), the input u itself.
    for (row, (c, d)) in [
        (1.0 / 7.0, 1.0 / 7.0),
        (2.0 / 7.0, 2.0 / 7.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_close(lin.c[row][0], c, 1.0e-10, &format!("C[{row}]"));
        assert_close(lin.d[row][0], d, 1.0e-10, &format!("D[{row}]"));
    }
    assert_close(lin.state_values[0], 1.0, 0.0, "x0");
    assert_close(lin.input_values[0], 1.0, 0.0, "u0");
    for (row, want) in [1.0, 1.0, 1.0, 1.0].into_iter().enumerate() {
        assert_close(lin.output_values[row], want, 1.0e-9, "y0");
    }
    assert_close(lin.state_derivatives[0], -1.0, 1.0e-9, "dx0");
}

#[test]
fn the_operating_point_moves_with_the_named_state_and_input() {
    // At x = 3, u = 2 the loop settles where a + b^3 = 5; with
    // db/da = (1 + 3 a^2) / 2 its tangent is da = ds / (1 + 3 b^2 db/da).
    let result = compile(LOOP, "Loop");
    let lin = linearize_dae(
        &result.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            states: vec![("x".into(), 3.0)],
            inputs: vec![("u".into(), 2.0)],
            ..loop_request()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let (a, b) = (lin.output_values[0], lin.output_values[1]);
    assert_close(a + b.powi(3), 5.0, 1.0e-9, "loop residual 1");
    assert_close(b, (a + a.powi(3)) / 2.0, 1.0e-9, "loop residual 2");
    let db_da = (1.0 + 3.0 * a * a) / 2.0;
    let da = 1.0 / (1.0 + 3.0 * b * b * db_da);
    assert_close(lin.c[0][0], da, 1.0e-9, "C[y]");
    assert_close(lin.d[0][0], da, 1.0e-9, "D[y]");
    assert_close(lin.a[0][0], -db_da * da, 1.0e-9, "A");
    assert_eq!(lin.input_values, [2.0]);
    assert_eq!(lin.states_set, ["x"]);
}

#[test]
fn an_unknown_name_is_refused_with_the_valid_names() {
    let result = compile(LOOP, "Loop");
    let opts = SimOptions::default();
    for (request, needle) in [
        (
            LinearizationRequest {
                input_names: names(&["v"]),
                ..loop_request()
            },
            "valid inputs: [u]",
        ),
        (
            LinearizationRequest {
                output_names: names(&["nope"]),
                ..loop_request()
            },
            "`nope` is not a variable",
        ),
        (
            LinearizationRequest {
                states: vec![("a".into(), 1.0)],
                ..loop_request()
            },
            "valid states: [x]",
        ),
    ] {
        let error = linearize_dae(&result.dae, &opts, &request).expect_err(needle);
        assert!(error.to_string().contains(needle), "{error}");
    }
    let error = linearize_dae(
        &result.dae,
        &opts,
        &LinearizationRequest {
            output_names: names(&["nope"]),
            ..loop_request()
        },
    )
    .expect_err("unknown output");
    for valid in ["x", "a", "b", "y"] {
        assert!(error.to_string().contains(valid), "{error}");
    }
    let error = trim_dae(
        &result.dae,
        &opts,
        &TrimRequest {
            fixed_states: names(&["a"]),
            ..TrimRequest::default()
        },
    )
    .expect_err("unknown fixed state");
    assert!(error.to_string().contains("valid states: [x]"), "{error}");
}

#[test]
fn lenient_states_take_over_the_states_of_a_snapshot_and_say_which() {
    let result = compile(LOOP, "Loop");
    let snapshot = vec![("a".into(), 7.0), ("x".into(), 3.0), ("y".into(), 7.0)];
    let lin = linearize_dae(
        &result.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            states: snapshot.clone(),
            lenient_states: true,
            ..loop_request()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(lin.state_values, [3.0]);
    assert_eq!(lin.states_set, ["x"]);
    let trim = trim_dae(
        &result.dae,
        &SimOptions::default(),
        &TrimRequest {
            states: snapshot,
            lenient_states: true,
            ..TrimRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(trim.states_set, ["x"]);
}

// ---------------------------------------------------------------------------
// Trim: a tank filled at `qin` and drained through a valve,
//   area der(h) = qin - kv sqrt(h),
// is steady at h = (qin / kv)^2. Beside it, a second tank whose pump draws
// exactly its inflow is an integrator with balanced flows: its column of A is
// zero, so Newton's matrix is singular and the step is Levenberg–Marquardt.
// ---------------------------------------------------------------------------

const TANKS: &str = r#"
model Tanks
  parameter Real area = 2;
  parameter Real kv = 0.5;
  parameter Real pump = 0.3;
  input Real qin = 1;
  Real h(start = 1, fixed = true);
  Real v(start = 4, fixed = true);
  Real qout;
equation
  qout = kv * sqrt(h);
  area * der(h) = qin - qout;
  der(v) = pump - 0.3;
end Tanks;
"#;

#[test]
fn a_drained_tank_trims_to_the_level_where_the_flows_balance() {
    let result = compile(TANKS, "Tanks");
    let opts = SimOptions::default();
    let trim = trim_dae(
        &result.dae,
        &opts,
        &TrimRequest {
            inputs: vec![("qin".into(), 0.8)],
            ..TrimRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let state = |name: &str| {
        trim.states
            .iter()
            .find(|(state, _)| state == name)
            .unwrap_or_else(|| panic!("no state {name} in {:?}", trim.states))
            .1
    };
    assert_close(state("h"), (0.8_f64 / 0.5).powi(2), 1.0e-9, "h");
    assert_eq!(state("v"), 4.0, "the integrator stays at its guess");
    assert!(trim.residual <= 1.0e-10, "residual {}", trim.residual);
    assert!(trim.iterations > 0 && trim.iterations < 50, "{trim:?}");

    // Held, the level keeps its guess and the residual says it is not steady.
    let held = trim_dae(
        &result.dae,
        &opts,
        &TrimRequest {
            states: vec![("h".into(), 1.0)],
            inputs: vec![("qin".into(), 0.8)],
            fixed_states: names(&["h"]),
            ..TrimRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(held.states[0], ("h".to_string(), 1.0));
    assert_close(held.residual, (0.8 - 0.5) / 2.0, 1.0e-12, "held residual");

    // The trimmed point linearizes to a steady state: A = -kv / (2 area sqrt(h)).
    let lin = linearize_dae(
        &result.dae,
        &opts,
        &LinearizationRequest {
            states: trim.states.clone(),
            inputs: vec![("qin".into(), 0.8)],
            input_names: names(&["qin"]),
            output_names: names(&["qout"]),
            ..LinearizationRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        lin.state_derivatives
            .iter()
            .all(|value| value.abs() <= 1.0e-10)
    );
    let h = state("h");
    assert_close(
        lin.a[0][0],
        -0.5 / (2.0 * 2.0 * h.sqrt()),
        1.0e-9,
        "A[h][h]",
    );
    assert_eq!(lin.a[1], [0.0, 0.0], "the integrator's row");
    assert_close(lin.b[0][0], 0.5, 1.0e-12, "B[h][qin]");
    assert_close(lin.c[0][0], 0.5 / (2.0 * h.sqrt()), 1.0e-9, "C[qout][h]");
    assert_close(lin.output_values[0], 0.8, 1.0e-9, "qout at the trim");
}

// ---------------------------------------------------------------------------
// 200 states, each through a nonlinear algebraic: z = x + 0.1 x^3, so at
// x = 1, dz/dx = 1.3.
// ---------------------------------------------------------------------------

const CHAIN: &str = r#"
model Chain
  parameter Integer n = 200;
  input Real u = 1;
  Real x[n](each start = 1, each fixed = true);
  Real z[n];
  output Real y;
equation
  der(x[1]) = u - z[1];
  for i in 2:n loop
    der(x[i]) = z[i - 1] - z[i];
  end for;
  for i in 1:n loop
    z[i] = x[i] + 0.1 * x[i] ^ 3;
  end for;
  y = z[n];
end Chain;
"#;

#[test]
fn two_hundred_states_linearize_well_under_a_second_after_lowering() {
    let result = compile(CHAIN, "Chain");
    let opts = SimOptions::default();
    let lowering = Instant::now();
    rumoca_sim::lower_for_simulation_with_overrides(&result.dae, &opts).expect("lower");
    let lowering = lowering.elapsed();
    let total = Instant::now();
    let lin = linearize_dae(
        &result.dae,
        &opts,
        &LinearizationRequest {
            input_names: names(&["u"]),
            output_names: names(&["y"]),
            ..LinearizationRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let total = total.elapsed();
    let after_lowering = total.saturating_sub(lowering);
    eprintln!("200 states: lowering {lowering:?}, linearize {total:?}");
    assert!(
        after_lowering.as_secs_f64() < 0.5,
        "linearization took {after_lowering:?} after a {lowering:?} lowering"
    );
    assert_eq!(lin.states.len(), 200);
    for (row, values) in lin.a.iter().enumerate() {
        assert_eq!(lin.states[row], format!("x[{}]", row + 1));
        for (col, &value) in values.iter().enumerate() {
            let want = if col == row {
                -1.3
            } else if col + 1 == row {
                1.3
            } else {
                0.0
            };
            assert_close(value, want, 1.0e-12, &format!("A[{row}][{col}]"));
        }
    }
    assert_close(lin.b[0][0], 1.0, 1.0e-12, "B");
    assert_close(lin.c[0][199], 1.3, 1.0e-12, "C");
}

// ---------------------------------------------------------------------------
// Selected state coordinates: V and T are preferred over the m and U the
// balances differentiate, so every state is a `$state_coordinates[i]`, which
// numbers it within one compile only. With rho = 2, k = 0.5:
//   der(V) = (qin - k V) / rho,  der(T) = qin (300 - T) / (rho V),
//   der(I) = V - I,
// steady at V = qin / k, T = 300, I = V.
// ---------------------------------------------------------------------------

const COORDINATES: &str = r#"
model Coordinates
  parameter Real rho = 2;
  parameter Real k = 0.5;
  input Real qin = 1;
  Real m;
  Real V(stateSelect = StateSelect.prefer, start = 1, fixed = true);
  Real U;
  Real T(stateSelect = StateSelect.prefer, start = 300, fixed = true);
  Real I(start = 0.5, fixed = true);
equation
  m = rho * V;
  der(m) = qin - k * V;
  U = m * T;
  der(U) = qin * 300 - k * V * T;
  der(I) = V - I;
end Coordinates;
"#;

/// A run's snapshot, by name in no model order, with the coordinates of
/// another compile's numbering, which must not reach this one's states.
fn coordinates_snapshot() -> Vec<(String, f64)> {
    [
        ("$state_coordinates[1]", 0.7),
        ("I", 0.7),
        ("$state_coordinates[2]", 1.5),
        ("T", 310.0),
        ("m", 3.0),
        ("V", 1.5),
        ("$state_coordinates[3]", 310.0),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect()
}

#[test]
fn state_coordinates_are_set_and_reported_by_the_variables_they_stand_for() {
    let result = compile(COORDINATES, "Coordinates");
    let lin = linearize_dae(
        &result.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            states: coordinates_snapshot(),
            lenient_states: true,
            ..LinearizationRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let at = |name: &str| {
        lin.states
            .iter()
            .position(|state| state == name)
            .unwrap_or_else(|| panic!("no state {name} in {:?}", lin.states))
    };
    let (v, t, i) = (at("V"), at("T"), at("I"));
    assert_eq!(lin.states_set.len(), 3, "{:?}", lin.states_set);
    for (index, want) in [(v, 1.5), (t, 310.0), (i, 0.7)] {
        assert_eq!(lin.state_values[index], want, "{}", lin.states[index]);
    }
    for (index, want) in [(v, 0.125), (t, -10.0 / 3.0), (i, 0.8)] {
        assert_close(
            lin.state_derivatives[index],
            want,
            1.0e-12,
            &lin.states[index],
        );
    }
    let want_a = [
        (v, v, -0.25),
        (v, t, 0.0),
        (v, i, 0.0),
        (t, v, -(300.0 - 310.0) / (2.0 * 1.5 * 1.5)),
        (t, t, -1.0 / (2.0 * 1.5)),
        (t, i, 0.0),
        (i, v, 1.0),
        (i, t, 0.0),
        (i, i, -1.0),
    ];
    for (row, col, want) in want_a {
        let what = format!("A[{}][{}]", lin.states[row], lin.states[col]);
        assert_close(lin.a[row][col], want, 1.0e-12, &what);
    }

    let trim = trim_dae(
        &result.dae,
        &SimOptions::default(),
        &TrimRequest {
            states: coordinates_snapshot(),
            lenient_states: true,
            ..TrimRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(trim.states_set.len(), 3, "{:?}", trim.states_set);
    for (name, want) in [("V", 2.0), ("T", 300.0), ("I", 2.0)] {
        let got = trim
            .states
            .iter()
            .find(|(state, _)| state == name)
            .unwrap_or_else(|| panic!("no state {name} in {:?}", trim.states))
            .1;
        assert_close(got, want, 1.0e-9, name);
    }
    assert!(trim.residual <= 1.0e-10, "{trim:?}");

    // Strict, the compile-local name is refused with the names that hold.
    let error = linearize_dae(
        &result.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            states: vec![("$state_coordinates[1]".into(), 1.0)],
            ..LinearizationRequest::default()
        },
    )
    .expect_err("a coordinate's compile-local name");
    for valid in ["V", "T", "I"] {
        assert!(error.to_string().contains(valid), "{error}");
    }
}
