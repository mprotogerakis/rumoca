//! A mixing valve pair set from the mixed temperature, without a lag, and a
//! `when` equation that reads that temperature's deviation: the event program
//! reads an algebraic loop through the valves' medium calls. It now reads the
//! loop from the algebraic projection and settles with it; before, the
//! construction stopped with "typed record field 0 of `setState_phX` could
//! not be invoked: ... same-tick algebraic definitions require a coupled
//! owner".

use rumoca_sim::{SimOptions, SimulationSession};

use super::msl_sim_regression::require_msl_compiler;

const SOURCE: &str = "model MixingLoopRead
  package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater;
  inner Modelica.Fluid.System system;
  Modelica.Fluid.Sources.Boundary_pT hot(redeclare package Medium = Medium, p = 2e5, T = 343.15, nPorts = 1);
  Modelica.Fluid.Sources.Boundary_pT cold(redeclare package Medium = Medium, p = 2e5, T = 303.15, nPorts = 1);
  Modelica.Fluid.Sources.Boundary_pT sink(redeclare package Medium = Medium, p = 1e5, T = 303.15, nPorts = 1);
  Modelica.Fluid.Valves.ValveLinear vh(redeclare package Medium = Medium, dp_nominal = 1e5, m_flow_nominal = 1);
  Modelica.Fluid.Valves.ValveLinear vc(redeclare package Medium = Medium, dp_nominal = 1e5, m_flow_nominal = 1);
  Modelica.Fluid.Sensors.TemperatureTwoPort ts(redeclare package Medium = Medium);
  Real e = 45 - (ts.T - 273.15);
  discrete Real offset(start = 0);
equation
  when time > 0.5 then
    offset = 0.01*e;
  end when;
  vh.opening = min(max(0.5 + 0.02*e + offset, 0), 1);
  vc.opening = 1 - vh.opening;
  connect(hot.ports[1], vh.port_a);
  connect(cold.ports[1], vc.port_a);
  connect(vh.port_b, ts.port_a);
  connect(vc.port_b, ts.port_a);
  connect(ts.port_b, sink.ports[1]);
end MixingLoopRead;
";

#[test]
fn a_when_reads_a_mixing_loop_through_its_medium_calls() {
    let dae = require_msl_compiler()
        .model("MixingLoopRead")
        .compile_str(SOURCE, "MixingLoopRead.mo")
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
    let (e, offset, t_mixed) = (get("e"), get("offset"), get("ts.T"));
    assert!((offset - 0.01 * e).abs() < 1e-8, "offset {offset}, e {e}");
    assert!(offset < -0.01, "the event fired: offset {offset}");
    assert!(
        (e - (45.0 - (t_mixed - 273.15))).abs() < 1e-8,
        "e {e}, ts.T {t_mixed}"
    );
}
