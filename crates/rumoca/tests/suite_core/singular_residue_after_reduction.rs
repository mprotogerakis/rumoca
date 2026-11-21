//! The triage of a model singular for a reason no index reduction removes
//! (`sim --inspect structure`): it names that defect, not the parts a
//! reduction already matched.

use rumoca::Compiler;
use rumoca_sim::{SimOptions, diagnose_structural_singularity};

/// An incompressible volume (Modelica.Fluid's ClosedVolume with a constant
/// density: `m = V*d`, `der(m)` the net inflow) beside an unconnected
/// mass-flow source, whose port flow is set twice (MLS §9.2's `= 0` and the
/// source's own) and its pressure by nothing. Demoting `m` matches the volume
/// (`der(m) = 0`, its pressure fixed by the flows); the source is what stays
/// singular where the reduction stalls, and it alone is named, not the
/// volume's `m = V*d` the source system leaves unmatched.
#[test]
fn a_singularity_names_what_no_reduction_matches() {
    let source = "
model VolumeBesideOpenSource
  parameter Real V = 0.02;
  parameter Real m_flow = 0.3;
  Real d = 1000;
  Real m;
  Real p(start = 1.5);
  Real inflow = 2 - p;
  Real outflow = p - 1;
  Real port_m_flow;
  Real port_p;
equation
  m = V*d;
  der(m) = inflow - outflow;
  port_m_flow = 0;
  port_m_flow = -m_flow;
end VolumeBesideOpenSource;";
    let compiled = Compiler::new()
        .model("VolumeBesideOpenSource")
        .compile_str(source, "VolumeBesideOpenSource.mo")
        .unwrap_or_else(|error| panic!("compiles: {error}"));
    let diagnosis = diagnose_structural_singularity(&compiled.dae, &SimOptions::default())
        .unwrap_or_else(|error| panic!("diagnoses: {error}"))
        .expect("the open source leaves the model singular");
    let unknowns = diagnosis
        .unknowns
        .iter()
        .map(|unknown| unknown.name.as_str())
        .collect::<Vec<_>>();
    let equations = diagnosis
        .equations
        .iter()
        .map(|equation| equation.summary.as_str())
        .collect::<Vec<_>>();
    assert!(diagnosis.after_reduction);
    assert_eq!(unknowns, ["port_p"], "{equations:?}");
    assert_eq!(equations.len(), 1, "{equations:?}");
    assert!(equations[0].starts_with("port_m_flow ="), "{equations:?}");
}
