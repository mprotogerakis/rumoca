//! An unclocked `when` reads an algebraic loop of explicit equations that
//! itself reads the when's target; the equations of the event instant are
//! simultaneous (MLS §8.5). Substituting the loop's definitions into the event
//! program comes back to where it started, so the program reads the loop from
//! the algebraic projection, and the settle pass iterates both to the
//! simultaneous solution. Before, construction stopped with "same-tick
//! algebraic definitions require a coupled owner".

use rumoca::Compiler;
use rumoca_sim::{SimOptions, SimulationSession};

const SOURCE: &str = "model LoopRead
  Real x;
  Real y;
  discrete Real offset(start = 0);
equation
  x = 1 + 0.5*sin(y) + offset;
  y = 2 - x;
  when time > 0.5 then
    offset = 0.1*y;
  end when;
end LoopRead;
";

#[test]
fn an_unclocked_when_reads_an_algebraic_loop_through_its_projection() {
    let dae = Compiler::new()
        .model("LoopRead")
        .compile_str(SOURCE, "LoopRead.mo")
        .unwrap_or_else(|error| panic!("{error}"))
        .dae;
    let mut session = SimulationSession::new(
        &dae,
        SimOptions {
            t_end: 1.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    session
        .advance_to(1.0)
        .unwrap_or_else(|error| panic!("{error}"));
    let get = |name: &str| session.get(name).unwrap().unwrap();
    let (x, y, offset) = (get("x"), get("y"), get("offset"));
    assert!((offset - 0.1 * y).abs() < 1e-8, "offset {offset}, y {y}");
    assert!(
        (x - (1.0 + 0.5 * y.sin() + offset)).abs() < 1e-8,
        "x {x}, y {y}"
    );
    assert!((y - (2.0 - x)).abs() < 1e-8, "x {x}, y {y}");
    assert!(offset > 0.05, "the event fired: offset {offset}");
}
