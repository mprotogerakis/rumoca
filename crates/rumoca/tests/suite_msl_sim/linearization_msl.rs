//! State-space linearization and trim of Modelica Standard Library blocks,
//! mechanics and fluid systems, against closed forms derived by hand and, for
//! the fluid tank, against central differences of the evaluated model.

use rumoca_sim::{
    Linearization, LinearizationRequest, SimOptions, TrimRequest, eval_dae_at, jacobian_for_dae,
    linearize_dae, trim_dae,
};

use super::msl_sim_regression::require_msl_compiler;

fn compile(source: &str, model: &str) -> rumoca::CompilationResult {
    require_msl_compiler()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("{model}: {error}"))
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

fn linearize(
    compiled: &rumoca::CompilationResult,
    inputs: &[(&str, f64)],
    outputs: &[&str],
) -> Linearization {
    linearize_dae(
        &compiled.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            inputs: inputs
                .iter()
                .map(|(name, value)| (name.to_string(), *value))
                .collect(),
            input_names: inputs.iter().map(|(name, _)| name.to_string()).collect(),
            output_names: names(outputs),
            ..LinearizationRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// `der(y) = (k u - y) / T`: `A = -1/T`, `B = k/T`, `C = 1`, `D = 0`.
#[test]
fn first_order_block_linearizes_to_its_transfer_function() {
    let compiled = compile(
        "model Lag\n  extends Modelica.Blocks.Continuous.FirstOrder(k = 2, T = 0.5);\nend Lag;\n",
        "Lag",
    );
    let lin = linearize(&compiled, &[("u", 1.5)], &["y"]);
    assert_eq!(lin.states, ["y"]);
    assert_close(lin.a[0][0], -2.0, 1.0e-12, "A");
    assert_close(lin.b[0][0], 4.0, 1.0e-12, "B");
    assert_close(lin.c[0][0], 1.0, 0.0, "C");
    assert_close(lin.d[0][0], 0.0, 0.0, "D");
    // y starts at 0: der(y) = k u / T.
    assert_close(lin.state_derivatives[0], 6.0, 1.0e-12, "dx0");
}

/// `y = k u`: no states, `D = k`.
#[test]
fn gain_block_is_a_pure_feedthrough() {
    let compiled = compile(
        "model Amp\n  extends Modelica.Blocks.Math.Gain(k = 3);\nend Amp;\n",
        "Amp",
    );
    let lin = linearize(&compiled, &[("u", 2.0)], &["y"]);
    assert!(lin.states.is_empty() && lin.a.is_empty() && lin.b.is_empty());
    assert_eq!(lin.c, [Vec::<f64>::new()]);
    assert_close(lin.d[0][0], 3.0, 1.0e-12, "D");
    assert_close(lin.output_values[0], 6.0, 1.0e-12, "y0");
}

/// `m s'' = f - c s - d s'` from connected components:
/// `A = [[0, 1], [-c/m, -d/m]]`, `B = [0, 1/m]`; the acceleration, an
/// algebraic, has `C = [-c/m, -d/m]` and `D = 1/m`.
#[test]
fn mass_spring_damper_has_the_textbook_state_space() {
    let compiled = compile(
        "model Msd
  Modelica.Blocks.Interfaces.RealInput f;
  Modelica.Mechanics.Translational.Components.Fixed fixed;
  Modelica.Mechanics.Translational.Components.Spring spring(c = 50);
  Modelica.Mechanics.Translational.Components.Damper damper(d = 3,
    stateSelect = StateSelect.never);
  Modelica.Mechanics.Translational.Components.Mass mass(m = 2,
    s(start = 0.1, fixed = true), v(start = 0, fixed = true));
  Modelica.Mechanics.Translational.Sources.Force force;
equation
  connect(fixed.flange, spring.flange_a);
  connect(fixed.flange, damper.flange_a);
  connect(spring.flange_b, mass.flange_a);
  connect(damper.flange_b, mass.flange_a);
  connect(force.flange, mass.flange_b);
  connect(f, force.f);
end Msd;
",
        "Msd",
    );
    let lin = linearize(&compiled, &[("f", 1.0)], &["mass.s", "mass.a"]);
    assert_eq!(lin.states, ["mass.s", "mass.v"]);
    let (m, c, d) = (2.0, 50.0, 3.0);
    let want_a = [[0.0, 1.0], [-c / m, -d / m]];
    for (row, want) in want_a.iter().enumerate() {
        for (col, &want) in want.iter().enumerate() {
            assert_close(lin.a[row][col], want, 1.0e-12, &format!("A[{row}][{col}]"));
        }
    }
    assert_close(lin.b[0][0], 0.0, 1.0e-12, "B[s]");
    assert_close(lin.b[1][0], 1.0 / m, 1.0e-12, "B[v]");
    assert_eq!(lin.c[0], [1.0, 0.0], "a state output is a unit row");
    assert_close(lin.c[1][0], -c / m, 1.0e-12, "C[a][s]");
    assert_close(lin.c[1][1], -d / m, 1.0e-12, "C[a][v]");
    assert_close(lin.d[0][0], 0.0, 0.0, "D[s]");
    assert_close(lin.d[1][0], 1.0 / m, 1.0e-12, "D[a]");
    // s = 0.1, v = 0, f = 1: a = (1 - 5) / 2.
    assert_close(lin.output_values[1], -2.0, 1.0e-12, "a0");
}

/// An open tank filled at `m_in` and drained through a linear valve to
/// ambient pressure: `m_out = opening k rho g level`, so
/// `der(level) = (m_in - m_out) / (rho area)` and the flows balance at
/// `level = m_in / (opening k rho g)`.
const TANK: &str = "model TankValve
  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater;
  Modelica.Blocks.Interfaces.RealInput m_in;
  Modelica.Blocks.Interfaces.RealInput opening;
  inner Modelica.Fluid.System system;
  Modelica.Fluid.Sources.MassFlowSource_T source(redeclare package Medium = Medium,
    use_m_flow_in = true, T = 293.15, nPorts = 1);
  Modelica.Fluid.Vessels.OpenTank tank(redeclare package Medium = Medium,
    height = 10, crossArea = 1, level_start = 1, nPorts = 2, use_portsData = false);
  Modelica.Fluid.Valves.ValveLinear valve(redeclare package Medium = Medium,
    dp_nominal = 1e4, m_flow_nominal = 1);
  Modelica.Fluid.Sources.Boundary_pT sink(redeclare package Medium = Medium,
    p = 101325, T = 293.15, nPorts = 1);
equation
  connect(m_in, source.m_flow_in);
  connect(opening, valve.opening);
  connect(source.ports[1], tank.ports[1]);
  connect(tank.ports[2], valve.port_a);
  connect(valve.port_b, sink.ports[1]);
end TankValve;
";

const RHO: f64 = 995.586;
const G: f64 = 9.80665;
const K: f64 = 1.0e-4;

#[test]
fn draining_tank_matches_central_differences_of_the_evaluated_model() {
    let compiled = compile(TANK, "TankValve");
    let inputs = [("m_in", 1.0), ("opening", 0.5)];
    let outputs = ["valve.m_flow", "tank.level", "tank.ports[2].p"];
    let lin = linearize(&compiled, &inputs, &outputs);
    let level = lin.state_values[lin.states.iter().position(|s| s == "tank.level").unwrap()];
    let row = lin.states.iter().position(|s| s == "tank.level").unwrap();
    assert_close(lin.a[row][row], -0.5 * K * G, 1.0e-9, "A[level][level]");
    assert_close(lin.b[row][0], 1.0 / RHO, 1.0e-9, "B[level][m_in]");
    assert_close(lin.b[row][1], -K * G * level, 1.0e-9, "B[level][opening]");
    assert_close(lin.c[0][row], 0.5 * K * RHO * G, 1.0e-9, "C[m_flow][level]");
    assert_close(
        lin.d[0][1],
        K * RHO * G * level,
        1.0e-9,
        "D[m_flow][opening]",
    );

    // The same A as the state-Jacobian inspector's exact JVP.
    let at_inputs = SimOptions {
        initial_inputs: inputs.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        ..SimOptions::default()
    };
    let probe = jacobian_for_dae(&compiled.dae, &at_inputs, &[], 0.0)
        .unwrap_or_else(|error| panic!("{error}"));
    for (row, values) in probe.report.matrix.iter().enumerate() {
        for (col, &value) in values.iter().enumerate() {
            assert_close(lin.a[row][col], value, 1.0e-12, &format!("A[{row}][{col}]"));
        }
    }

    // Central differences of `eval_at` along every state and input.
    let eval = |states: &[(String, f64)], inputs: &[(&str, f64)]| {
        let opts = SimOptions {
            initial_inputs: inputs.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
            ..SimOptions::default()
        };
        let report = eval_dae_at(&compiled.dae, &opts, states, 0.0)
            .unwrap_or_else(|error| panic!("{error}"))
            .report;
        // `eval_at`'s event actions refuse the medium's `0 <= X <= 1` check at
        // X = 1, which a simulation passes; the values are evaluated before.
        assert!(
            report
                .error
                .as_deref()
                .is_none_or(|error| error.starts_with("assertion failed: Mass fraction")),
            "{:?}",
            report.error
        );
        let named = |name: &str| {
            report
                .solver_y
                .iter()
                .find(|slot| slot.name == name)
                .unwrap_or_else(|| panic!("no `{name}`"))
                .value
        };
        let mut values: Vec<f64> = report.derivatives.iter().map(|slot| slot.value).collect();
        values.extend(outputs.iter().map(|name| named(name)));
        values
    };
    // `eval_at` names the selected coordinates `$state_coordinates[i]`.
    let base: Vec<(String, f64)> = probe
        .state_names
        .iter()
        .cloned()
        .zip(lin.state_values.iter().copied())
        .collect();
    let n = lin.states.len();
    let compare = |column: Vec<f64>, ad: Vec<f64>, what: String| {
        for (index, (fd, ad)) in column.iter().zip(&ad).enumerate() {
            assert!(
                (fd - ad).abs() <= 1.0e-6 * ad.abs().max(1.0e-6),
                "{what} row {index}: AD {ad}, central difference {fd}"
            );
        }
    };
    for col in 0..n {
        let h = 1.0e-6 * base[col].1.abs().max(1.0);
        let (mut plus, mut minus) = (base.clone(), base.clone());
        plus[col].1 += h;
        minus[col].1 -= h;
        let (f_plus, f_minus) = (eval(&plus, &inputs), eval(&minus, &inputs));
        let fd = f_plus
            .iter()
            .zip(&f_minus)
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect();
        let ad = (0..n)
            .map(|row| lin.a[row][col])
            .chain(lin.c.iter().map(|row| row[col]));
        compare(fd, ad.collect(), format!("d/d{}", lin.states[col]));
    }
    for col in 0..inputs.len() {
        let h = 1.0e-6 * inputs[col].1.abs().max(1.0);
        let (mut plus, mut minus) = (inputs, inputs);
        plus[col].1 += h;
        minus[col].1 -= h;
        let (f_plus, f_minus) = (eval(&base, &plus), eval(&base, &minus));
        let fd = f_plus
            .iter()
            .zip(&f_minus)
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect();
        let ad = (0..n)
            .map(|row| lin.b[row][col])
            .chain(lin.d.iter().map(|row| row[col]));
        compare(fd, ad.collect(), format!("d/d{}", inputs[col].0));
    }
}

#[test]
fn draining_tank_trims_to_the_level_where_inflow_meets_valve_flow() {
    let compiled = compile(TANK, "TankValve");
    let inputs = vec![("m_in".to_string(), 1.0), ("opening".to_string(), 0.5)];
    let trim = trim_dae(
        &compiled.dae,
        &SimOptions::default(),
        &TrimRequest {
            inputs: inputs.clone(),
            ..TrimRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let level = trim
        .states
        .iter()
        .find(|(name, _)| name == "tank.level")
        .unwrap_or_else(|| panic!("{:?}", trim.states))
        .1;
    assert_close(level, 1.0 / (0.5 * K * RHO * G), 1.0e-9, "trimmed level");
    assert!(trim.residual <= 1.0e-10, "{trim:?}");

    let lin = linearize_dae(
        &compiled.dae,
        &SimOptions::default(),
        &LinearizationRequest {
            // As from a run's snapshot of every variable.
            states: [("valve.m_flow".to_string(), 3.0)]
                .into_iter()
                .chain(trim.states)
                .collect(),
            lenient_states: true,
            inputs,
            output_names: names(&["valve.m_flow"]),
            ..LinearizationRequest::default()
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        lin.states_set, lin.states,
        "every state taken over by its name"
    );
    assert!(
        lin.state_derivatives
            .iter()
            .all(|value| value.abs() <= 1.0e-10)
    );
    assert_close(lin.output_values[0], 1.0, 1.0e-9, "valve flow at the trim");
}
