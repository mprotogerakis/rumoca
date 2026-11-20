//! `Modelica.Fluid.Examples.TraceSubstances.RoomCO2`: the room volume's mass
//! and energy are states without a start of their own (`m = V*d`,
//! `U = m*u`); the trial point that selects the medium's preferred `p`, `Xi`
//! and `T` seeds them from those definitions instead of holding them at zero.

use rumoca_sim::{SimOptions, simulate_dae_with_diagnostics};

use super::msl_sim_regression::require_msl_compiler;

#[test]
fn a_room_starts_from_its_medium_state_and_is_aired_of_its_co2() {
    let source = "model Room
  extends Modelica.Fluid.Examples.TraceSubstances.RoomCO2;
end Room;";
    let compiled = require_msl_compiler()
        .model("Room")
        .compile_str(source, "Room.mo")
        .unwrap_or_else(|error| panic!("Room compiles: {error}"));
    let result = simulate_dae_with_diagnostics(
        &compiled.dae,
        &SimOptions {
            t_end: 60.0,
            ..SimOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("Room simulates: {error}"));
    let column = |name: &str| {
        let index = result.names.iter().position(|n| n == name).expect(name);
        *result.data[index].last().expect("a sample")
    };
    let (t, c) = (column("volume.medium.T"), column("traceVolume.C"));
    assert!((t - 293.15).abs() < 1.0, "volume.medium.T(60) = {t}");
    assert!(c > 1.2e-3, "traceVolume.C(60) = {c}");
}
