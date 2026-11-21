//! Human-readable structural reports over the same prepared DAE used by simulation.

use rumoca_ir_dae as dae;
use rumoca_solver::SimOptions;

use super::diagnostics::SimulationDiagnosticError;

/// Report the structure Solve lowering analyzes: the STRUCT-T02 alias quotient
/// of `model` after STRUCT-T10(a) parameter folding (with every class it left
/// unchanged) followed by the matching and BLT of the prepared quotient, and,
/// for a constrained system, the formal-derivative application of the quotient
/// on the reduced candidate Solve lowering executes.
pub fn structural_report_for_dae(
    model: &dae::Dae,
    _: &SimOptions,
) -> Result<rumoca_phase_structural::StructuralReport, SimulationDiagnosticError> {
    let folded =
        rumoca_phase_structural::fold_evaluable_parameters(model).map_err(structural_error)?;
    let model = folded.as_ref().unwrap_or(model);
    let aliases = rumoca_phase_structural::alias_quotient_report(model);
    let quotient = rumoca_phase_structural::quotient_aliases(model).map_err(structural_error)?;
    let literal = rumoca_phase_structural::fold_constant_values(quotient.as_ref().unwrap_or(model))
        .map_err(structural_error)?
        .or(quotient);
    let analyzed = literal.as_ref().unwrap_or(model);
    let prepared =
        rumoca_phase_structural::prepare_for_solve(analyzed).map_err(structural_error)?;
    let mut report = prepared.structural_report();
    report.aliases = aliases;
    for request in rumoca_phase_solve::withheld_state_preferences(model).unwrap_or_default() {
        report.notes.push(format!(
            "a StateSelect.prefer request is not honored: {}",
            request.message(model)
        ));
    }
    if rumoca_phase_solve::executes_reduced_state_selection(analyzed, &prepared)
        .map_err(structural_error)?
    {
        report.notes.push(
            "this is the structural reducer's BLT; Solve lowering instead executes a state \
             selection built from formal derivatives (for a constrained state manifold or for \
             the StateSelect preferences), whose blocks differ (inspect the emitted Solve IR \
             for those)"
                .to_string(),
        );
        if let Ok(names) = rumoca_phase_solve::integrated_state_names(model) {
            report
                .notes
                .push(format!("Solve lowering integrates: {}", names.join(", ")));
        }
        // The reduced candidate is prepared only by Solve lowering; a model that
        // does not lower still reports its reducer structure above.
        if let Ok(lowered) =
            rumoca_phase_solve::lower_solve_model(model, &std::collections::HashMap::new(), |_| {})
        {
            report.formal_aliases = lowered.formal_alias_report().clone();
        }
    }
    Ok(report)
}

#[derive(Debug, Clone)]
pub struct SingularityDiagnosis {
    /// The counts and names are those of the closest system index reduction
    /// stalled at, not of the source: its reducible parts are reduced.
    pub after_reduction: bool,
    pub n_equations: usize,
    pub n_unknowns: usize,
    pub n_matched: usize,
    pub unknowns: Vec<UnmatchedUnknownDiagnosis>,
    pub equations: Vec<UnmatchedEquationDiagnosis>,
}

#[derive(Debug, Clone)]
pub struct UnmatchedEquationDiagnosis {
    pub name: String,
    pub origin: String,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct UnmatchedUnknownDiagnosis {
    pub name: String,
    pub category: String,
    pub referencing_rows: Vec<usize>,
}

pub fn diagnose_structural_singularity(
    model: &dae::Dae,
    _: &SimOptions,
) -> Result<Option<SingularityDiagnosis>, SimulationDiagnosticError> {
    let folded =
        rumoca_phase_structural::fold_evaluable_parameters(model).map_err(structural_error)?;
    let model = folded.as_ref().unwrap_or(model);
    let quotient = rumoca_phase_structural::quotient_aliases(model).map_err(structural_error)?;
    let literal = rumoca_phase_structural::fold_constant_values(quotient.as_ref().unwrap_or(model))
        .map_err(structural_error)?
        .or(quotient);
    let model = literal.as_ref().unwrap_or(model);
    let (prepared, reduction) = rumoca_phase_structural::inspect_prepare_for_solve(model);
    let Err(error) = prepared else {
        return Ok(None);
    };
    // The reducer reports the source's singularity. Where it stalled closer,
    // the reducible parts are reduced (an incompressible volume's `der(m)`),
    // so what stays unmatched there is what no reduction matches: the defect.
    let stalled = reduction
        .stalled
        .as_ref()
        .map(|snapshot| snapshot.error())
        .filter(|stalled| match (unmatched(stalled), unmatched(&error)) {
            (Some(stalled), Some(source)) => stalled < source,
            _ => false,
        });
    let after_reduction = stalled.is_some();
    let error = stalled.cloned().unwrap_or(error);
    let rumoca_phase_structural::StructuralError::Singular {
        n_equations,
        n_unknowns,
        n_matched,
        unmatched_equations,
        unmatched_unknowns,
        ..
    } = error
    else {
        return Err(structural_error(error));
    };
    let unknowns = unmatched_unknowns
        .into_iter()
        .map(|name| UnmatchedUnknownDiagnosis {
            category: classify_unknown(&name).to_string(),
            name,
            referencing_rows: Vec::new(),
        })
        .collect();
    let equations = unmatched_equations
        .into_iter()
        .map(equation_diagnosis)
        .collect();
    Ok(Some(SingularityDiagnosis {
        after_reduction,
        n_equations,
        n_unknowns,
        n_matched,
        unknowns,
        equations,
    }))
}

fn unmatched(error: &rumoca_phase_structural::StructuralError) -> Option<usize> {
    match error {
        rumoca_phase_structural::StructuralError::Singular {
            unmatched_equations,
            unmatched_unknowns,
            ..
        } => Some(unmatched_equations.len() + unmatched_unknowns.len()),
        _ => None,
    }
}

/// A row's source text, read off its label `f_x[i] (<text>)`: the label was
/// spelled in the DAE the singularity was proved on, which after an index
/// reduction is not `model`, so `i` need not name the same row of `model`.
fn equation_diagnosis(name: String) -> UnmatchedEquationDiagnosis {
    let source = name
        .split_once(" (")
        .and_then(|(_, text)| text.strip_suffix(')'))
        .unwrap_or("")
        .to_string();
    UnmatchedEquationDiagnosis {
        name,
        origin: source.clone(),
        summary: source,
    }
}

fn classify_unknown(name: &str) -> &'static str {
    if name.starts_with("der(") {
        "unmatched state derivative"
    } else if name.ends_with(".v") {
        "unmatched connector potential"
    } else if name.ends_with(".tau") || name.ends_with(".f") {
        "unmatched connector flow"
    } else {
        "unmatched algebraic coordinate"
    }
}

fn structural_error(error: rumoca_phase_structural::StructuralError) -> SimulationDiagnosticError {
    SimulationDiagnosticError::RuntimePreparation {
        message: error.to_string(),
        span: error.source_span(),
    }
}
