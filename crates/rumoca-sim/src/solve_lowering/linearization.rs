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
use rumoca_solver::{
    AlgebraicLinearization, AlgebraicSettle, EventUpdateRowFilter, ProjectedEventUpdateInput,
    SimOptions, SolveRuntime,
};

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

/// Lower `model` and linearize it at the request's operating point, its
/// discrete values those a run starts with, switched as an event at the point
/// would switch them.
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
/// Pseudo-time steps, rejected ones included, before the search gives up.
const PTC_MAX_STEPS: usize = 500;
/// A pseudo-time step past which `I/Δt` is lost to rounding against any `A`.
const PTC_MAX_DT: f64 = 1.0e30;
/// The largest move of a pseudo-time step, as a fraction of each state's scale.
const TRUST: f64 = 0.2;
/// Discrete mode switches at a steady state before the search keeps the last.
const MODE_SWITCHES: usize = 3;

/// Lower `model` and solve `der(x)(x, u) = 0` for the free states.
///
/// First Newton on the exact `A` restricted to the free columns, with
/// Levenberg–Marquardt steps `(JᵀJ + μI) δ = -Jᵀ der` where that matrix is
/// singular or rectangular (an integrator whose flows balance, held states)
/// or the step does not reduce `|der(x)|`. Where that stalls (a rate-limited
/// or saturated state has a constant derivative and a zero column in `A`, so
/// no step has a gradient to follow), pseudo-transient continuation from the
/// start guess follows the system's own trajectory towards its steady state
/// ([`pseudo_transient`]), and Newton finishes from where it ends.
///
/// Discrete values start as a run's initialization leaves them; each point
/// reads the relations its equations read off itself; the pseudo-time
/// trajectory switches discrete modes as an event at each step would; and a
/// steady state an event there would switch is searched again in its new
/// mode. The search stops at `max |der| <= 1e-10` or its step budget;
/// `residual` reports where it stood, so a system with no steady state
/// reachable from the guess (an unstable equilibrium) reports a residual, not
/// a point.
pub fn trim_dae(
    model: &dae::Dae,
    opts: &SimOptions,
    request: &TrimRequest,
) -> Result<TrimResult, SimulationDiagnosticError> {
    let mut point = OperatingPoint::lower(
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
    let mut iterations = 0;
    let start = point.state.clone();
    let (mut x, mut f) = steady_state(&mut point, &free, start, &mut iterations)?;
    // A steady state whose discrete variables an event there would switch is
    // the steady state of a mode it is not in: that mode's, then.
    for _ in 0..MODE_SWITCHES {
        let params = point.event_at(&x);
        if max_abs(&f) > TRIM_TOL || params == point.params {
            break;
        }
        point.params = params;
        (x, f) = steady_state(&mut point, &free, x, &mut iterations)?;
    }
    Ok(TrimResult {
        residual: max_abs(&f),
        states: point.names.into_iter().zip(x).collect(),
        states_set: point.states_set,
        iterations,
    })
}

/// Newton from `start`; where it stalls, pseudo-transient continuation from
/// `start`, finished by Newton. `(x, der(x))` where the better one ended, the
/// point's discrete values those of its mode.
fn steady_state(
    point: &mut OperatingPoint,
    free: &[usize],
    start: Vec<f64>,
    iterations: &mut usize,
) -> Result<(Vec<f64>, Vec<f64>), SimulationDiagnosticError> {
    let mut x = start.clone();
    let f = newton(point, free, &mut x, iterations)?;
    if free.iter().all(|&state| f[state].abs() <= TRIM_TOL) {
        return Ok((x, f));
    }
    let mode = point.params.clone();
    let mut y = start;
    pseudo_transient(point, free, &mut y, iterations)?;
    let g = newton(point, free, &mut y, iterations)?;
    if max_abs(&g) < max_abs(&f) {
        return Ok((y, g));
    }
    point.params = mode;
    Ok((x, f))
}

/// Newton/Levenberg–Marquardt from `x` over the `free` states: `der(x)` where
/// it stops, `x` moved there, `iterations` counting its steps.
fn newton(
    point: &OperatingPoint,
    free: &[usize],
    x: &mut Vec<f64>,
    iterations: &mut usize,
) -> Result<Vec<f64>, SimulationDiagnosticError> {
    let mut f = point.derivatives(x)?;
    let mut mu = 0.0_f64;
    for _ in 0..TRIM_MAX_ITERS {
        if max_abs(&f) <= TRIM_TOL || free.is_empty() {
            break;
        }
        let jacobian = point.columns(x, free)?;
        let Some((trial, trial_f)) = damped_descent(point, (&jacobian, free), (x, &f), &mut mu)
        else {
            break;
        };
        let moved = trial
            .iter()
            .zip(x.iter())
            .any(|(new, old)| (new - old).abs() > 1.0e-14 * (1.0 + old.abs()));
        *x = trial;
        f = trial_f;
        *iterations += 1;
        if !moved {
            break;
        }
    }
    Ok(f)
}

/// Pseudo-transient continuation (C. T. Kelley and D. E. Keyes, "Convergence
/// analysis of pseudo-transient continuation", SIAM J. Numer. Anal. 35(2),
/// 1998): `x ← x + (I/Δt − A)⁻¹ der(x)` over the free states, a linearly
/// implicit Euler step of the system's own dynamics, which follows its
/// trajectory while `Δt` is small and becomes Newton's step as `Δt` grows.
///
/// - `Δt` starts at [`initial_step`] and grows by switched evolution
///   relaxation (W. A. Mulder and B. van Leer, J. Comput. Phys. 59, 1985),
///   `Δt ← Δt·‖der_prev‖/‖der‖` within ×10 a step, and at least doubles while
///   `‖der‖` does not double: a state on its rate limit moves at a constant
///   speed, which plain SER would never reward.
/// - A step moves no state by more than a fifth of its scale (a trust region:
///   the Newton step of a large `Δt` does not leap past the nonlinearity it
///   linearized), and is shortened, without growing `Δt`, to fit.
/// - A step across a state's target is bisected: where a derivative changes
///   sign without shrinking by half (a saturated or switched derivative jumps
///   there), until it does not, which lands the state in its linear zone; a
///   step whose derivative is not finite is bisected too, and `Δt` halves
///   where no length of it is admissible.
/// - Each trial is evaluated in the discrete mode an event there leaves
///   ([`OperatingPoint::event_at`]), so the trajectory passes a tank's spill
///   port as a run would.
fn pseudo_transient(
    point: &mut OperatingPoint,
    free: &[usize],
    x: &mut Vec<f64>,
    iterations: &mut usize,
) -> Result<(), SimulationDiagnosticError> {
    let rows = |f: &[f64]| -> Vec<f64> { free.iter().map(|&state| f[state]).collect() };
    let mut f = rows(&point.derivatives(x)?);
    let mut dt = 0.0_f64;
    for _ in 0..PTC_MAX_STEPS {
        if max_abs(&f) <= TRIM_TOL {
            break;
        }
        let columns = point.columns(x, free)?;
        let a: Vec<Vec<f64>> = free.iter().map(|&state| columns[state].clone()).collect();
        if dt == 0.0 {
            dt = initial_step(&a, (x, free), &f);
        }
        let implicit: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| f64::from(u8::from(i == j)) / dt - v)
                    .collect()
            })
            .collect();
        let Some(Trial {
            length,
            x: trial,
            f: trial_f,
            mode,
        }) = solve_dense(&implicit, &f)
            .and_then(|step| admissible_step(point, (x, free), &f, &step))
        else {
            dt /= 2.0;
            continue;
        };
        let relaxation = norm2(&f) / norm2(&trial_f);
        if length == 1.0 {
            let growth = if relaxation >= 0.5 {
                relaxation.clamp(2.0, 10.0)
            } else {
                relaxation
            };
            dt = (dt * growth).min(PTC_MAX_DT);
        }
        *x = trial;
        f = trial_f;
        point.params = mode;
        *iterations += 1;
    }
    Ok(())
}

/// An admissible pseudo-time step: `x + length·step`, `der` there over the
/// free rows, and the discrete values of its mode.
struct Trial {
    length: f64,
    x: Vec<f64>,
    f: Vec<f64>,
    mode: Vec<f64>,
}

/// The longest of `step`, `step/2`, `step/4`, … from `x` (at most 30
/// halvings, within the trust region) whose derivative is finite and sends no
/// state across its target, each trial in the discrete mode an event there
/// leaves.
fn admissible_step(
    point: &OperatingPoint,
    (x, free): (&[f64], &[usize]),
    f: &[f64],
    step: &[f64],
) -> Option<Trial> {
    // A trust region: no state moves by more than a fifth of its own scale
    // in one step, so a Newton step of a large `Δt` cannot leap past the
    // nonlinearity it linearized.
    let reach = free
        .iter()
        .zip(step)
        .map(|(&state, delta)| {
            delta.abs() / (TRUST * x[state].abs().max(point.model.solver_variable_scale(state)))
        })
        .fold(1.0, f64::max);
    let mut length = 1.0 / reach;
    for _ in 0..30 {
        let scaled: Vec<f64> = step.iter().map(|delta| delta * length).collect();
        let trial = moved(x, free, &scaled);
        let mode = point.event_at(&trial);
        if let Ok(trial_f) = point.derivatives_in(&trial, &mode) {
            let trial_f: Vec<f64> = free.iter().map(|&state| trial_f[state]).collect();
            let bounced = f
                .iter()
                .zip(&trial_f)
                .any(|(old, new)| old * new < 0.0 && new.abs() >= 0.5 * old.abs());
            if !bounced {
                return Some(Trial {
                    length,
                    x: trial,
                    f: trial_f,
                    mode,
                });
            }
        }
        length /= 2.0;
    }
    None
}

/// The first pseudo-time step: no longer than the fastest mode's time
/// constant, `1/‖A‖∞` (the row-sum norm bounds the spectral radius), so the
/// first step resolves every mode the linearization sees; and no longer than
/// moves a state by a hundredth of its own scale (`max(|x|, 1)`), which bounds
/// it where `A` is zero (every state on a rate limit or saturated).
fn initial_step(a: &[Vec<f64>], (x, free): (&[f64], &[usize]), f: &[f64]) -> f64 {
    let fastest = a
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let motion = free
        .iter()
        .zip(f)
        .filter(|(_, derivative)| **derivative != 0.0)
        .map(|(&state, derivative)| x[state].abs().max(1.0) / derivative.abs())
        .fold(f64::INFINITY, f64::min);
    (1.0 / fastest).min(0.01 * motion)
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

/// The discrete values a run starts with, by name into the parameter slots
/// that hold them: its initialization and initial event settle them (a pump's
/// `Run`, copied from its input at that event), where the lowered model holds
/// their `start`. A model whose run does not start keeps those.
#[cfg(any(feature = "solver-diffsol", feature = "solver-rk45"))]
fn initialized_discrete(
    model: &dae::Dae,
    opts: &SimOptions,
    solve_model: &solve::SolveModel,
    params: &mut [f64],
) {
    let layout = &solve_model.problem.solve_layout;
    if layout.discrete_real_scalar_names.is_empty()
        && layout.discrete_valued_scalar_names.is_empty()
    {
        return;
    }
    let Ok(session) = crate::SimulationSession::new(model, opts.clone()) else {
        return;
    };
    for name in layout
        .discrete_real_scalar_names
        .iter()
        .chain(&layout.discrete_valued_scalar_names)
    {
        if let Some(solve::ScalarSlot::P { index, .. }) = solve_model.problem.layout.binding(name)
            && let Ok(Some(value)) = session.get(name)
        {
            params[index] = value;
        }
    }
}

#[cfg(not(any(feature = "solver-diffsol", feature = "solver-rk45")))]
fn initialized_discrete(_: &dae::Dae, _: &SimOptions, _: &solve::SolveModel, _: &mut [f64]) {}

/// A state settled for evaluation.
struct Settled {
    params: Vec<f64>,
    solver_y: Vec<f64>,
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
    /// Its parameters with the discrete values of the point.
    params: Vec<f64>,
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
        let mut params = solve_model.parameters.to_vec();
        initialized_discrete(
            model,
            &SimOptions {
                t_start: t,
                t_end: t + 1.0,
                // (the rk-like start settles a tank whose spill port is above
                // its level, which the BDF start refuses as singular)
                solver_mode: rumoca_solver::SimSolverMode::RkLike,
                ..opts.clone()
            },
            &solve_model,
            &mut params,
        );
        let mut point = Self {
            p_seed_offset: solve_model.problem.layout.y_scalars(),
            model: solve_model,
            runtime,
            t,
            state,
            names,
            states_set,
            params,
        };
        point.params = point.event_at(&point.state);
        Ok(point)
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

        let at = self.settle_at(&self.state)?;
        let (solver_y, params) = (&at.solver_y, &at.params);
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
            let (der, tangent) = self.tangent(&at, seed)?;
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

    /// The parameters at `state`: the point's discrete values, with the
    /// memory of the relations its equations read (a switched derivative)
    /// read off `state`.
    fn params_at(&self, state: &[f64]) -> Result<Vec<f64>, SimulationDiagnosticError> {
        self.params_in(state, &self.params)
    }

    /// [`Self::params_at`] with the discrete values of `mode`.
    fn params_in(
        &self,
        state: &[f64],
        mode: &[f64],
    ) -> Result<Vec<f64>, SimulationDiagnosticError> {
        let mut params = mode.to_vec();
        self.runtime.update_relation_memory_from_state(
            self.t,
            state,
            &mut params,
            SETTLE.tol,
            SETTLE.max_iters,
        )?;
        Ok(params)
    }

    /// The discrete values an event at `state` leaves, `pre` being the
    /// point's own (no edge fires): where its discrete variables follow the
    /// state, they switch there (a tank's port the level passed). Where the
    /// event iteration finds no consistent mode at `state` (a branch it
    /// selects is singular there), the point keeps the ones it has, as a run
    /// would until a relation crosses.
    fn event_at(&self, state: &[f64]) -> Vec<f64> {
        let Ok(pre_p) = self.params_at(state) else {
            return self.params.clone();
        };
        let mut params = pre_p.clone();
        let Ok(mut solver_y) =
            self.runtime
                .full_solver_y(self.t, state, &params, SETTLE.tol, SETTLE.max_iters)
        else {
            return self.params.clone();
        };
        let pre_y = solver_y.clone();
        let event = self.runtime.apply_projected_event_update(
            ProjectedEventUpdateInput {
                y: &mut solver_y,
                p: &mut params,
                t: self.t,
                tol: SETTLE.tol,
                event_pre_y: &pre_y,
                event_pre_p: &pre_p,
                max_iters: SETTLE.max_iters,
                row_filter: EventUpdateRowFilter::UnownedOnly,
                root_relation_overrides: &mut Vec::new(),
            },
            |y, p| {
                let before = y.to_vec();
                self.runtime.refresh_algebraic_and_output_slots(
                    self.t,
                    y,
                    p,
                    SETTLE.tol,
                    SETTLE.max_iters,
                )?;
                Ok(before != y)
            },
        );
        match event {
            Ok(_) => params,
            Err(_) => self.params.clone(),
        }
    }

    /// `state` settled: its parameters and its algebraics.
    fn settle_at(&self, state: &[f64]) -> Result<Settled, SimulationDiagnosticError> {
        let params = self.params_at(state)?;
        let solver_y =
            self.runtime
                .full_solver_y(self.t, state, &params, SETTLE.tol, SETTLE.max_iters)?;
        Ok(Settled { params, solver_y })
    }

    fn derivatives(&self, state: &[f64]) -> Result<Vec<f64>, SimulationDiagnosticError> {
        self.derivatives_in(state, &self.params)
    }

    /// `der(state)` with the discrete values of `mode`.
    fn derivatives_in(
        &self,
        state: &[f64],
        mode: &[f64],
    ) -> Result<Vec<f64>, SimulationDiagnosticError> {
        let params = self.params_in(state, mode)?;
        Ok(self.runtime.eval_state_derivatives(
            self.t,
            state,
            &params,
            SETTLE.tol,
            SETTLE.max_iters,
        )?)
    }

    /// The columns `cols` of `A` at `state`: `columns[row][k] =
    /// ∂der(state_row)/∂state_cols[k]`, through the algebraics the
    /// derivatives read only (a leaf algebraic need not be solvable there).
    fn columns(
        &self,
        state: &[f64],
        cols: &[usize],
    ) -> Result<Vec<Vec<f64>>, SimulationDiagnosticError> {
        let params = self.params_at(state)?;
        let lin = AlgebraicLinearization {
            t: self.t,
            params: &params,
            settle: SETTLE,
        };
        let mut seed = vec![0.0; state.len()];
        let mut column = vec![0.0; state.len()];
        let mut columns = vec![Vec::with_capacity(cols.len()); state.len()];
        for &col in cols {
            seed[col] = 1.0;
            let result = self
                .runtime
                .eval_state_jacobian_v_ad_into(lin, state, &seed, &mut column);
            seed[col] = 0.0;
            result?;
            for (row, value) in columns.iter_mut().zip(&column) {
                row.push(*value);
            }
        }
        Ok(columns)
    }

    /// `(∂der, ∂solver_y)` along the unit seed at `seed` of
    /// `[solver-y | parameter]`, at a settled point.
    fn tangent(
        &self,
        at: &Settled,
        seed: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), SimulationDiagnosticError> {
        let mut direction = vec![0.0; self.p_seed_offset + self.model.problem.layout.p_scalars()];
        direction[seed] = 1.0;
        let mut der = vec![0.0; self.state.len()];
        let mut tangent = vec![0.0; self.runtime.solver_count()];
        self.runtime
            .eval_linearization_tangent_into(
                AlgebraicLinearization {
                    t: self.t,
                    params: &at.params,
                    settle: SETTLE,
                },
                &at.solver_y,
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
