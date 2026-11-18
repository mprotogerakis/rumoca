//! State-space linearization `x' = A x + B u`, `y = C x + D u` at an operating
//! point, and the steady-state trim that finds one.
//!
//! The point's algebraics are settled once. Each state and each input is then
//! one seed over `[solver-y | parameter]` (inputs live in parameter slots after
//! the model parameters), pushed through the full algebraic projection's
//! tangent and the derivative rows' JVP: one propagation yields a column of
//! `A` or `B` and, from the tangents of the named solver variables, a column of
//! `C` or `D`. Every entry is an automatic directional derivative of the
//! lowered rows; none is a finite difference.

use rumoca_core::linear_solve::{AugmentedMatrix, gaussian_eliminate};
use rumoca_ir_dae as dae;
use rumoca_ir_solve as solve;
use rumoca_solver::{AlgebraicLinearization, AlgebraicSettle, SimOptions, SolveRuntime};

use super::diagnostics::SimulationDiagnosticError;
use super::overrides::{
    input_scalar_names, inputs_without_default, lower_for_simulation_with_overrides,
};
use super::probe::{EVAL_AT_REFRESH_MAX_ITERS, EVAL_AT_REFRESH_TOL, resolve_probe_state};

const SETTLE: AlgebraicSettle = AlgebraicSettle {
    tol: EVAL_AT_REFRESH_TOL,
    max_iters: EVAL_AT_REFRESH_MAX_ITERS,
};

/// Where to linearize, and along which inputs into which outputs.
#[derive(Debug, Clone, Default)]
pub struct LinearizationRequest {
    pub t: f64,
    /// Operating state by name, as [`Linearization::states`] names it (a
    /// selected state coordinate by the variable it stands for; its
    /// `$state_coordinates[i]` numbers it within one compile only, so it is
    /// not a state's name); an unset state keeps its model initial value.
    pub states: Vec<(String, f64)>,
    /// Ignore names in `states` that are not states of this model (a run's
    /// snapshot of every variable) instead of refusing them.
    pub lenient_states: bool,
    /// Input values at the operating point; an unset input keeps its default
    /// binding, and an input without one must be given.
    pub inputs: Vec<(String, f64)>,
    /// `u`: the inputs differentiated along (the columns of `B` and `D`).
    pub input_names: Vec<String>,
    /// `y`: any model variable, state, algebraic or output (the rows of `C`
    /// and `D`).
    pub output_names: Vec<String>,
}

/// A linearized model with named axes.
#[derive(Debug, Clone)]
pub struct Linearization {
    pub t: f64,
    /// State names; a selected state coordinate goes by the variable it
    /// stands for.
    pub states: Vec<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Operating state `x0`, aligned with `states`.
    pub state_values: Vec<f64>,
    /// The states the request set, in model order; every other state is at
    /// its model initial value.
    pub states_set: Vec<String>,
    /// Operating input values `u0`, aligned with `inputs`.
    pub input_values: Vec<f64>,
    /// Output values `y0` at the point, aligned with `outputs`.
    pub output_values: Vec<f64>,
    /// `der(state)` at the point (`dx0`); away from zero, the point is not a
    /// steady state.
    pub state_derivatives: Vec<f64>,
    /// `a[row][col] = ∂(der(state_row))/∂(state_col)`.
    pub a: Vec<Vec<f64>>,
    /// `b[row][col] = ∂(der(state_row))/∂(input_col)`.
    pub b: Vec<Vec<f64>>,
    /// `c[row][col] = ∂(output_row)/∂(state_col)`.
    pub c: Vec<Vec<f64>>,
    /// `d[row][col] = ∂(output_row)/∂(input_col)`.
    pub d: Vec<Vec<f64>>,
}

/// Lower `model` and linearize it at the request's operating point.
pub fn linearize_dae(
    model: &dae::Dae,
    opts: &SimOptions,
    request: &LinearizationRequest,
) -> Result<Linearization, SimulationDiagnosticError> {
    OperatingPoint::lower(
        model,
        opts,
        request.t,
        &request.inputs,
        (&request.states, request.lenient_states),
    )?
    .linearize(&request.input_names, &request.output_names)
}

/// Lower `model` and linearize it along every declared input into every
/// declared output, at the named overrides (a named input fixes the operating
/// input, every other name is a state) and time `t`.
pub fn linearization_for_dae(
    model: &dae::Dae,
    opts: &SimOptions,
    overrides: &[(String, f64)],
    t: f64,
) -> Result<Linearization, SimulationDiagnosticError> {
    let input_names = input_scalar_names(model);
    let (inputs, states): (Vec<_>, Vec<_>) = overrides
        .iter()
        .cloned()
        .partition(|(name, _)| input_names.contains(name));
    let point = OperatingPoint::lower(model, opts, t, &inputs, (&states, false))?;
    let input_names: Vec<String> = point
        .model
        .problem
        .solve_layout
        .input_scalar_names()
        .iter()
        .collect();
    let output_names: Vec<String> = point
        .model
        .visible_metadata()
        .into_iter()
        .filter(|meta| meta.role == "output")
        .map(|meta| meta.name)
        .collect();
    point.linearize(&input_names, &output_names)
}

/// A steady-state search: `der(x) = 0` over the states not held, with the
/// inputs held at their given values.
#[derive(Debug, Clone, Default)]
pub struct TrimRequest {
    pub t: f64,
    /// Start guess by state name; an unset state starts at its model initial
    /// value.
    pub states: Vec<(String, f64)>,
    /// See [`LinearizationRequest::lenient_states`].
    pub lenient_states: bool,
    /// Input values, held (see [`LinearizationRequest::inputs`]).
    pub inputs: Vec<(String, f64)>,
    /// States held at their start value rather than solved for.
    pub fixed_states: Vec<String>,
}

/// The trimmed operating point.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// Every state, in model order.
    pub states: Vec<(String, f64)>,
    /// `max |der(state)|` over every state there; zero at a steady state.
    pub residual: f64,
    /// Accepted Newton/Levenberg–Marquardt steps.
    pub iterations: usize,
    /// The states the request's start guess set.
    pub states_set: Vec<String>,
}

const TRIM_MAX_ITERS: usize = 100;
const TRIM_TOL: f64 = 1.0e-10;

/// Lower `model` and solve `der(x)(x, u) = 0` for the free states.
///
/// Each step is a Newton step on the exact `A` restricted to the free columns.
/// When that matrix is singular or rectangular (an integrator whose flows
/// balance, or held states), or the step does not reduce `|der(x)|`, it is a
/// Levenberg–Marquardt step `(JᵀJ + μI) δ = -Jᵀ der` with `μ` raised until
/// the residual falls. The search stops at `max |der| <= 1e-10`, at a step
/// below rounding, or when no damping reduces the residual; `residual` reports
/// where it stood.
pub fn trim_dae(
    model: &dae::Dae,
    opts: &SimOptions,
    request: &TrimRequest,
) -> Result<TrimResult, SimulationDiagnosticError> {
    let point = OperatingPoint::lower(
        model,
        opts,
        request.t,
        &request.inputs,
        (&request.states, request.lenient_states),
    )?;
    let mut free: Vec<usize> = (0..point.names.len()).collect();
    for name in &request.fixed_states {
        let fixed = point
            .names
            .iter()
            .position(|state| state == name)
            .ok_or_else(|| not_a_state(name, &point.names))?;
        free.retain(|&index| index != fixed);
    }
    let mut x = point.state.clone();
    let mut f = point.derivatives(&x)?;
    let mut iterations = 0;
    let mut mu = 0.0_f64;
    while iterations < TRIM_MAX_ITERS && max_abs(&f) > TRIM_TOL && !free.is_empty() {
        let solver_y = point.settle_at(&x)?;
        let mut jacobian = vec![Vec::with_capacity(free.len()); f.len()];
        for &state in &free {
            let (column, _) = point.tangent(&solver_y, state)?;
            for (row, value) in jacobian.iter_mut().zip(column) {
                row.push(value);
            }
        }
        let Some((trial, trial_f)) = damped_descent(&point, (&jacobian, &free), (&x, &f), &mut mu)
        else {
            break;
        };
        let moved = trial
            .iter()
            .zip(&x)
            .any(|(new, old)| (new - old).abs() > 1.0e-14 * (1.0 + old.abs()));
        x = trial;
        f = trial_f;
        iterations += 1;
        if !moved {
            break;
        }
    }
    Ok(TrimResult {
        residual: max_abs(&f),
        states: point.names.into_iter().zip(x).collect(),
        states_set: point.states_set,
        iterations,
    })
}

/// The first trial from `x` that lowers `|der(x)|`: Newton's step while `mu`
/// is zero and the system square, else `(JᵀJ + μI) δ = -Jᵀ der` with `mu`
/// raised until one does. An accepted step lowers `mu` again; `None` when no
/// damping helps.
fn damped_descent(
    point: &OperatingPoint,
    (jacobian, free): (&[Vec<f64>], &[usize]),
    (x, f): (&[f64], &[f64]),
    mu: &mut f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (normal, gradient) = normal_equations(jacobian, f);
    let scale = (0..free.len())
        .map(|i| normal[i][i])
        .fold(f64::MIN_POSITIVE, f64::max);
    let start = norm2(f);
    loop {
        let step = if *mu == 0.0 && free.len() == f.len() {
            solve_dense(jacobian, &f.iter().map(|v| -v).collect::<Vec<_>>())
        } else {
            let damped: Vec<Vec<f64>> = (0..free.len())
                .map(|i| {
                    let mut row = normal[i].clone();
                    row[i] += *mu;
                    row
                })
                .collect();
            solve_dense(&damped, &gradient)
        };
        if let Some(trial) = step.map(|step| moved(x, free, &step))
            && let Ok(trial_f) = point.derivatives(&trial)
            && norm2(&trial_f) < start
        {
            *mu = if *mu < 1.0e-9 * scale {
                0.0
            } else {
                *mu / 10.0
            };
            return Some((trial, trial_f));
        }
        *mu = (*mu * 10.0).max(1.0e-6 * scale);
        if *mu > 1.0e16 * scale {
            return None;
        }
    }
}

/// `x` with `step` added to its `free` entries.
fn moved(x: &[f64], free: &[usize], step: &[f64]) -> Vec<f64> {
    let mut trial = x.to_vec();
    for (&state, delta) in free.iter().zip(step) {
        trial[state] += delta;
    }
    trial
}

/// A lowered model, its runtime and a named operating state.
struct OperatingPoint {
    model: solve::SolveModel,
    runtime: SolveRuntime,
    t: f64,
    state: Vec<f64>,
    names: Vec<String>,
    /// The states the request set.
    states_set: Vec<String>,
    /// Where the parameter (and input) seeds start in `[solver-y | parameter]`.
    p_seed_offset: usize,
}

impl OperatingPoint {
    fn lower(
        model: &dae::Dae,
        opts: &SimOptions,
        t: f64,
        inputs: &[(String, f64)],
        (states, lenient): (&[(String, f64)], bool),
    ) -> Result<Self, SimulationDiagnosticError> {
        // An input with no default has no operating value until one is named.
        if let Some(missing) = inputs_without_default(model)
            .into_iter()
            .find(|name| !inputs.iter().any(|(given, _)| given == name))
        {
            return Err(invalid(format!(
                "the operating point needs a value for input `{missing}` \
                 (on the command line: `--at {missing}=<value>`)"
            )));
        }
        let opts = SimOptions {
            initial_inputs: inputs.to_vec(),
            ..opts.clone()
        };
        let solve_model = lower_for_simulation_with_overrides(model, &opts)?;
        let (mut state, solver_names) = resolve_probe_state(&solve_model, &[], "operating point")?;
        // A selected state coordinate goes by the variable it stands for: its
        // `$state_coordinates[i]` is numbered by this compile, and a snapshot
        // of another (the run before an analysis point was added) numbers
        // them otherwise.
        let visible = solve_model.visible_projection();
        let names: Vec<String> = solver_names
            .iter()
            .map(|name| {
                visible
                    .index(name)
                    .and_then(|point| visible.metadata(point)?.state_coordinate)
                    .map_or_else(|| name.clone(), |coordinate| coordinate.source_name())
            })
            .collect();
        let mut set = vec![false; names.len()];
        for (name, value) in states {
            match names.iter().position(|state| state == name) {
                Some(index) => (state[index], set[index]) = (*value, true),
                None if lenient => {}
                None => return Err(not_a_state(name, &names)),
            }
        }
        let states_set = names
            .iter()
            .zip(set)
            .filter(|(_, set)| *set)
            .map(|(name, _)| name.clone())
            .collect();
        let runtime = SolveRuntime::new(&solve_model).map_err(SimulationDiagnosticError::from)?;
        Ok(Self {
            p_seed_offset: solve_model.problem.layout.y_scalars(),
            model: solve_model,
            runtime,
            t,
            state,
            names,
            states_set,
        })
    }

    fn linearize(
        self,
        input_names: &[String],
        output_names: &[String],
    ) -> Result<Linearization, SimulationDiagnosticError> {
        let layout = &self.model.problem.solve_layout;
        let inputs = input_names
            .iter()
            .map(|name| {
                layout
                    .input_parameter_index(name)
                    .map(|slot| (name.clone(), slot))
                    .ok_or_else(|| {
                        invalid(format!(
                            "linearize: `{name}` is not an input of this model; valid inputs: [{}]",
                            layout
                                .input_scalar_names()
                                .iter()
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = output_names
            .iter()
            .map(|name| match self.model.problem.layout.binding(name) {
                Some(
                    slot @ (solve::ScalarSlot::Y { .. }
                    | solve::ScalarSlot::P { .. }
                    | solve::ScalarSlot::Constant(_)),
                ) => Ok((name.clone(), slot)),
                Some(solve::ScalarSlot::Time) | None => Err(invalid(format!(
                    "linearize: `{name}` is not a variable of this model; valid names: [{}]",
                    self.model.visible_names().join(", ")
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let solver_y = self.settle_at(&self.state)?;
        let params = &self.model.parameters;
        let n = self.state.len();
        let mut a = vec![vec![0.0; n]; n];
        let mut b = vec![vec![0.0; inputs.len()]; n];
        let mut c = vec![vec![0.0; n]; outputs.len()];
        let mut d = vec![vec![0.0; inputs.len()]; outputs.len()];
        let seeds = (0..n).map(|state| (state, None)).chain(
            inputs
                .iter()
                .enumerate()
                .map(|(col, (_, slot))| (col, Some(*slot))),
        );
        for (col, input_slot) in seeds {
            let seed = input_slot.map_or(col, |slot| self.p_seed_offset + slot);
            let (der, tangent) = self.tangent(&solver_y, seed)?;
            let (columns, rows) = match input_slot {
                None => (&mut a, &mut c),
                Some(_) => (&mut b, &mut d),
            };
            for (row, value) in der.into_iter().enumerate() {
                columns[row][col] = value;
            }
            for (row, (_, slot)) in outputs.iter().enumerate() {
                rows[row][col] = match *slot {
                    solve::ScalarSlot::Y { index, .. } => tangent[index],
                    solve::ScalarSlot::P { index, .. } if Some(index) == input_slot => 1.0,
                    _ => 0.0,
                };
            }
        }
        let output_values = outputs
            .iter()
            .map(|(_, slot)| match *slot {
                solve::ScalarSlot::Y { index, .. } => solver_y[index],
                solve::ScalarSlot::P { index, .. } => params[index],
                solve::ScalarSlot::Constant(value) => value,
                solve::ScalarSlot::Time => self.t,
            })
            .collect();
        Ok(Linearization {
            t: self.t,
            state_derivatives: self.derivatives(&self.state)?,
            input_values: inputs.iter().map(|(_, slot)| params[*slot]).collect(),
            inputs: inputs.into_iter().map(|(name, _)| name).collect(),
            outputs: outputs.into_iter().map(|(name, _)| name).collect(),
            output_values,
            states: self.names,
            state_values: self.state,
            states_set: self.states_set,
            a,
            b,
            c,
            d,
        })
    }

    fn lin(&self) -> AlgebraicLinearization<'_> {
        AlgebraicLinearization {
            t: self.t,
            params: &self.model.parameters,
            settle: SETTLE,
        }
    }

    /// Solver-y at `state`, its algebraics settled.
    fn settle_at(&self, state: &[f64]) -> Result<Vec<f64>, SimulationDiagnosticError> {
        self.runtime
            .full_solver_y(
                self.t,
                state,
                &self.model.parameters,
                SETTLE.tol,
                SETTLE.max_iters,
            )
            .map_err(SimulationDiagnosticError::from)
    }

    fn derivatives(&self, state: &[f64]) -> Result<Vec<f64>, SimulationDiagnosticError> {
        self.runtime
            .eval_state_derivatives(
                self.t,
                state,
                &self.model.parameters,
                SETTLE.tol,
                SETTLE.max_iters,
            )
            .map_err(SimulationDiagnosticError::from)
    }

    /// `(∂der, ∂solver_y)` along the unit seed at `seed` of
    /// `[solver-y | parameter]`, at a settled `solver_y`.
    fn tangent(
        &self,
        solver_y: &[f64],
        seed: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), SimulationDiagnosticError> {
        let mut direction = vec![0.0; self.p_seed_offset + self.model.problem.layout.p_scalars()];
        direction[seed] = 1.0;
        let mut der = vec![0.0; self.state.len()];
        let mut tangent = vec![0.0; self.runtime.solver_count()];
        self.runtime
            .eval_linearization_tangent_into(
                self.lin(),
                solver_y,
                &direction,
                &mut tangent,
                &mut der,
            )
            .map_err(SimulationDiagnosticError::from)?;
        Ok((der, tangent))
    }
}

fn not_a_state(name: &str, names: &[String]) -> SimulationDiagnosticError {
    invalid(format!(
        "`{name}` is not a state of this model; valid states: [{}]",
        names.join(", ")
    ))
}

fn invalid(message: String) -> SimulationDiagnosticError {
    SimulationDiagnosticError::InvalidOverride { message }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |max, value| max.max(value.abs()))
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// `(JᵀJ, -Jᵀf)`.
fn normal_equations(jacobian: &[Vec<f64>], f: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let k = jacobian.first().map_or(0, Vec::len);
    let lhs = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| jacobian.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect();
    let rhs = (0..k)
        .map(|i| {
            -jacobian
                .iter()
                .zip(f)
                .map(|(row, v)| row[i] * v)
                .sum::<f64>()
        })
        .collect();
    (lhs, rhs)
}

/// `matrix · x = rhs` for square `matrix`, or `None` when it is singular.
fn solve_dense(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    let mut augmented = AugmentedMatrix::zeroed(n).ok()?;
    for (row, values) in matrix.iter().enumerate() {
        for (col, &value) in values.iter().enumerate() {
            augmented.set(row, col, value);
        }
        augmented.set(row, n, rhs[row]);
    }
    gaussian_eliminate(&mut augmented)?;
    Some(
        (0..n)
            .map(|row| augmented.solution_component(row))
            .collect(),
    )
}
