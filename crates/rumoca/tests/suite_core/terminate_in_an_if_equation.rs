//! `terminate(..)` in a branch of an if-equation (MLS 3.7 §8.3.8).
//!
//! An if-equation's branches become conditional residuals, and a call
//! without a result is none: `if x > 0.5 then terminate("..."); end if;` was
//! left out without a word and the run went on to its stop time. It ends the
//! run where the branch comes to hold, as a when-equation on that condition;
//! the branch's other equations stay the if-equation they were.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, SimulationSession, simulate_dae_with_diagnostics};

/// The time the run ended at, and `y` there.
fn run(body: &str) -> (f64, f64) {
    let source = format!("model Ends\n  Real x(start = 0, fixed = true);\n  Real y;\nequation\n  der(x) = 1;\n{body}\nend Ends;\n");
    let compiled = Compiler::new()
        .model("Ends")
        .compile_str(&source, "ends.mo")
        .unwrap_or_else(|error| panic!("{body}: {error:#}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 2.0,
            dt: Some(0.125),
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("{body}: {error}"));
    let y = result.names.iter().position(|n| n == "y").expect("y");
    (
        *result.times.last().expect("a sample"),
        *result.data[y].last().expect("a sample"),
    )
}

#[test]
fn a_branch_s_terminate_ends_the_run_where_the_branch_comes_to_hold() {
    let (end, _) = run("  y = 2*x;\n  if x > 0.5 then\n    terminate(\"x passed 0.5\");\n  end if;");
    assert!((end - 0.5).abs() < 1e-6, "ended at {end}");
}

/// The branch's other equations are the if-equation's as before.
#[test]
fn the_branch_s_other_equations_stay() {
    let (end, y) = run("  if x > 0.75 then\n    y = 1;\n    terminate(\"x passed 0.75\");\n  else\n    y = 0;\n  end if;");
    assert!((end - 0.75).abs() < 1e-6 && y == 1.0, "ended at {end} with y {y}");
}

/// A later branch holds only where none before it does; an else too.
#[test]
fn a_later_branch_s_terminate_waits_for_its_own_condition() {
    let (end, _) = run("  y = x;\n  if x < 0.25 then\n  elseif x > 1.0 then\n    terminate(\"x passed 1\");\n  end if;");
    assert!((end - 1.0).abs() < 1e-6, "elseif: ended at {end}");
    let (end, _) = run("  y = x;\n  if x < 1.25 then\n  else\n    terminate(\"x reached 1.25\");\n  end if;");
    assert!((end - 1.25).abs() < 1e-6, "else: ended at {end}");
}

/// Without a terminate nothing changes: the run goes to its stop time.
#[test]
fn an_if_equation_without_one_runs_on() {
    let (end, y) = run("  if x > 0.5 then\n    y = 1;\n  else\n    y = 0;\n  end if;");
    assert!((end - 2.0).abs() < 1e-9 && y == 1.0, "ended at {end} with y {y}");
}

/// A terminate in an if-equation nested in a branch waits for both
/// conditions, whichever of them the run decides.
#[test]
fn a_nested_branch_s_terminate_waits_for_both_conditions() {
    let (end, _) = run("  y = x;\n  if x > 0.25 then\n    if x > 0.75 then\n      terminate(\"x passed 0.75\");\n    end if;\n  end if;");
    assert!((end - 0.75).abs() < 1e-6, "inner later: ended at {end}");
    let (end, y) = run("  if x > 0.5 then\n    y = 1;\n    if x < 1.0 then\n    else\n      terminate(\"x reached 1\");\n    end if;\n  else\n    y = 0;\n  end if;");
    assert!((end - 1.0).abs() < 1e-6 && y == 1.0, "inner else: ended at {end} with y {y}");
}

/// A branch that holds from the start ends the run there: a when-equation
/// alone would wait for an edge that never comes.
#[test]
fn a_branch_that_holds_from_the_start_ends_the_run_there() {
    let (end, _) = run("  y = x;\n  if x > -1 then\n    terminate(\"from the start\");\n  end if;");
    assert!(end.abs() < 1e-9, "ended at {end}");
}

fn session(body: &str) -> Result<SimulationSession, String> {
    let source = format!("model Ends\n  Real x(start = 0, fixed = true);\n  Real y;\nequation\n  der(x) = 1;\n{body}\nend Ends;\n");
    let compiled = Compiler::new()
        .model("Ends")
        .compile_str(&source, "ends.mo")
        .unwrap_or_else(|error| panic!("{body}: {error:#}"));
    SimulationSession::new(
        &compiled.dae,
        SimOptions {
            t_end: 2.0,
            ..SimOptions::default()
        },
    )
    .map_err(|error| error.to_string())
}

/// A stepping session says which terminate ended its run, and stays there.
#[test]
fn a_stepping_session_tells_the_terminate_that_ended_it() {
    let mut stepped = session("  y = x;\n  if x > 0.5 then\n    terminate(\"x passed 0.5\");\n  end if;")
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(stepped.termination().is_none());
    stepped.advance_to(0.25).unwrap_or_else(|error| panic!("{error}"));
    assert!(stepped.termination().is_none(), "at 0.25 s");
    stepped.advance_to(1.0).unwrap_or_else(|error| panic!("{error}"));
    let ended = stepped.termination().expect("ended by its terminate");
    assert!((ended.time - 0.5).abs() < 1e-6 && ended.message == "x passed 0.5", "{ended:?}");
    assert!((stepped.time() - 0.5).abs() < 1e-6, "stands at {}", stepped.time());
}

/// A model that ends its run at initialization has no session to step: the
/// refusal names its terminate.
#[test]
fn a_session_of_a_model_that_ends_at_its_start_is_refused_by_that() {
    let refused = session("  y = x;\n  if x > -1 then\n    terminate(\"from the start\");\n  end if;")
        .err()
        .expect("no session");
    assert!(refused.contains("terminate") && refused.contains("from the start"), "{refused}");
}
