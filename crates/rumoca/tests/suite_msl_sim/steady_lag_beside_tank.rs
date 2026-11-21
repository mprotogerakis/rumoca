//! A tank at 60 degC (`DynamicFreeInitial`: its `T_start` is a guess), its
//! temperature through a `FirstOrder` started steady (`y_start = 0`) and a
//! state set from it by `initial equation Tw = ts.T` (no start). Each short row
//! determines the state at the default start, not the tank's temperature.

use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

use super::msl_sim_regression::require_msl_compiler;

#[test]
fn a_steady_lag_and_a_measured_state_start_at_the_tanks_temperature() {
    let source = "model B
  inner Modelica.Fluid.System system;
  Modelica.Fluid.Vessels.OpenTank tank(redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater, height = 2, crossArea = 1, level_start = 1, T_start = 333.15, nPorts = 2, portsData = {Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.05), Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.05)});
  Modelica.Fluid.Sources.MassFlowSource_T src(redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater, m_flow = 0, T = 293.15, nPorts = 1);
  Modelica.Fluid.Sensors.Temperature ts(redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater);
  Modelica.Blocks.Continuous.FirstOrder lag(T = 10, initType = Modelica.Blocks.Types.Init.SteadyState);
  Real Tw;
initial equation
  Tw = ts.T;
equation
  connect(src.ports[1], tank.ports[1]);
  connect(ts.port, tank.ports[2]);
  connect(ts.T, lag.u);
  der(Tw) = (ts.T - Tw)/10;
end B;";
    let compiled = require_msl_compiler()
        .model("B")
        .compile_str(source, "B.mo")
        .unwrap_or_else(|error| panic!("B compiles: {error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 10.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("B simulates: {error}"));
    let first = |name: &str| {
        let index = result.names.iter().position(|n| n == name).expect(name);
        result.data[index][0]
    };
    for name in ["tank.medium.T", "lag.y", "Tw"] {
        let value = first(name);
        assert!((value - 333.15).abs() < 1.0e-6, "{name}(0) = {value}");
    }
}
