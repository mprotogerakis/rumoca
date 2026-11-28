//! `delay(x, T)` over the solver step that ends at a state event.
//!
//! A lag whose input a two-point controller switches: `x` is continuous at a
//! switching, its slope jumps there. The event's commit left no pair in the
//! delay history (a located root settles at its instant, and the commit was
//! taken for a step's), so the cubic stencil reached across the kink, into
//! the points a few ulps and a millisecond after it, and `delay` read the
//! whole last step before each switching off a curve bent to the slope after
//! it: 0.045 off `x(time - T)` where `x` spans 0.2, and above every value `x`
//! ever had. With the pair there, pruning took the stencil's left neighbour,
//! which left a straight line over that step (1.4e-3 off).
//!
//! MLS 3.7 §3.7.4: `delay(expr, delayTime)` is `expr(time - delayTime)`.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, SimSolverMode, SimulationSession};

const SOURCE: &str = "
model TwoPoint
  Real x(start = 0, fixed = true);
  Boolean on(start = true, fixed = true);
  Real y = delay(x, 4);
equation
  der(x) = ((if on then 1 else 0) - x)/5;
  when x > 0.8 then
    on = false;
  elsewhen x < 0.6 then
    on = true;
  end when;
end TwoPoint;";

/// The largest `|y(t) - x(t - 4)|` on a grid of 0.25 s to 30 s, and its time.
fn worst(solver_mode: SimSolverMode) -> (f64, f64) {
    let dae = Compiler::new()
        .model("TwoPoint")
        .compile_str(SOURCE, "TwoPoint.mo")
        .unwrap_or_else(|error| panic!("TwoPoint compiles: {error:#}"))
        .dae;
    let mut session = SimulationSession::new(
        &dae,
        SimOptions {
            solver_mode,
            t_end: 30.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("TwoPoint starts: {error}"));
    let mut x = vec![0.0];
    let mut worst = (0.0, 0.0);
    for step in 1..=120_usize {
        let time = step as f64 * 0.25;
        session
            .advance_to(time)
            .unwrap_or_else(|error| panic!("advance_to({time}): {error}"));
        x.push(session.get("x").unwrap().unwrap());
        let y = session.get("y").unwrap().unwrap();
        let error = (y - x[step.saturating_sub(16)]).abs();
        if error > worst.0 {
            worst = (error, time);
        }
    }
    worst
}

/// `x` switches five times in 30 s; what is left is the history's
/// interpolation over the solver's steps of about a second (2.3e-4).
#[test]
fn a_delay_reads_the_step_before_a_switching_rk() {
    let (error, time) = worst(SimSolverMode::RkLike);
    assert!(
        error < 6e-4,
        "delay(x, 4) is {error} off x(time - 4) at {time} s"
    );
}

#[test]
fn a_delay_reads_the_step_before_a_switching_bdf() {
    let (error, time) = worst(SimSolverMode::Bdf);
    assert!(
        error < 6e-4,
        "delay(x, 4) is {error} off x(time - 4) at {time} s"
    );
}
