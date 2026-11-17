use std::collections::BTreeMap;

use super::*;
use crate::balance::BalanceLeftover;

pub(super) struct SourceBalanceInput<'scope> {
    pub(super) flat: &'scope flat::Model,
    pub(super) roles: &'scope HashMap<VarName, PlannedRole>,
    pub(super) assigned_targets: &'scope HashSet<VarName>,
    pub(super) excluded_equation_rows: &'scope HashSet<usize>,
    pub(super) record_equations: &'scope HashMap<usize, RecordEquationPlan>,
    pub(super) multi_output_equations: &'scope HashMap<usize, MultiOutputEquationPlan>,
    pub(super) connection_ranks: &'scope DiscreteConnectionRanks,
    pub(super) aggregate_connections: &'scope AggregateDiscreteConnections,
}

/// One source of the balance counts: an unknown with its scalars, or a Flat
/// equation row or a variable definition (binding, `when` or algorithm target)
/// with the scalar equations it contributes. ED001 matches them to name what
/// an unbalanced model leaves over.
pub(super) enum BalanceOwner<'a> {
    Unknown(&'a VarName, usize),
    Equation(usize, usize),
    Definition(&'a VarName, usize),
}

pub(super) fn source_balance(input: SourceBalanceInput<'_>) -> Result<BalanceDetail, ToDaeError> {
    source_balance_owners(input, &mut |_| {})
}

fn source_balance_owners(
    input: SourceBalanceInput<'_>,
    record: &mut dyn FnMut(BalanceOwner<'_>),
) -> Result<BalanceDetail, ToDaeError> {
    let SourceBalanceInput {
        flat,
        roles,
        assigned_targets,
        excluded_equation_rows,
        record_equations,
        multi_output_equations,
        connection_ranks,
        aggregate_connections,
    } = input;
    let mut detail = BalanceDetail::default();
    for (name, role) in roles {
        let variable = &flat.variables[name];
        let scalar_count = checked_shape_size(name, variable)?;
        let before = detail.unknowns();
        match role {
            PlannedRole::UnusedExpandable
            | PlannedRole::Parameter
            | PlannedRole::Constant
            | PlannedRole::Input
            | PlannedRole::Clock => {}
            PlannedRole::State => detail.state_unknowns += scalar_count,
            PlannedRole::Algebraic => detail.algebraic_unknowns += scalar_count,
            PlannedRole::Output => detail.output_unknowns += scalar_count,
            PlannedRole::DiscreteReal | PlannedRole::DiscreteValue
                if variable.binding.is_none() && !assigned_targets.contains(name) => {}
            PlannedRole::DiscreteReal => detail.discrete_real_unknowns += scalar_count,
            PlannedRole::DiscreteValue => detail.discrete_value_unknowns += scalar_count,
            PlannedRole::EnumerationLiteral | PlannedRole::Aggregate => {
                unreachable!("expression-only roles are not model variables")
            }
        }
        if detail.unknowns() > before {
            record(BalanceOwner::Unknown(name, scalar_count));
        }
    }
    for (row, equation) in flat.equations.iter().enumerate() {
        if excluded_equation_rows.contains(&row) {
            continue;
        }
        let before = detail.equations();
        match equation_partition(
            flat,
            row,
            equation,
            roles,
            connection_ranks,
            aggregate_connections,
        )? {
            EquationPartition::Continuous => {
                if let Some(plan) = record_equations.get(&row) {
                    add_record_equation_counts(&mut detail, flat, equation, plan, roles)?;
                    record(BalanceOwner::Equation(row, detail.equations() - before));
                    continue;
                }
                detail.continuous_equations += if let Some(plan) = multi_output_equations.get(&row)
                {
                    multi_output_equation_scalar_count(flat, equation, plan)?
                } else {
                    equation.scalar_count
                };
            }
            EquationPartition::DiscreteReal { .. } => {
                detail.discrete_real_equations += equation.scalar_count;
            }
            EquationPartition::DiscreteValue(plan) => {
                detail.discrete_value_definitions +=
                    plan.scalar_count.unwrap_or(equation.scalar_count);
            }
            EquationPartition::DiscreteElements(plans) => {
                detail.discrete_value_definitions += plans.len();
            }
            EquationPartition::MultiOutput { receivers, .. } => {
                add_multi_output_receivers(&mut detail, flat, roles, &receivers)?;
            }
            EquationPartition::ConsumedDiscreteValue => {}
        }
        record(BalanceOwner::Equation(row, detail.equations() - before));
    }
    add_binding_counts(&mut detail, flat, roles, record)?;
    let algorithm_targets = flat
        .algorithms
        .iter()
        .flat_map(|algorithm| model_algorithm_targets(flat, algorithm));
    for target in when_chain_targets(flat)
        .into_iter()
        .chain(algorithm_targets)
    {
        let before = detail.equations();
        add_algorithm_target(&mut detail, flat, roles, &target)?;
        record(BalanceOwner::Definition(
            &target,
            detail.equations() - before,
        ));
    }
    Ok(detail)
}

/// Each receiver of a multi-result equation is defined by that equation in
/// the system its role selects.
fn add_multi_output_receivers(
    detail: &mut BalanceDetail,
    flat: &flat::Model,
    roles: &HashMap<VarName, PlannedRole>,
    receivers: &[VarName],
) -> Result<(), ToDaeError> {
    for receiver in receivers {
        let scalar_count = checked_shape_size(receiver, &flat.variables[receiver])?;
        match roles[receiver] {
            PlannedRole::DiscreteReal => detail.discrete_real_equations += scalar_count,
            PlannedRole::DiscreteValue => detail.discrete_value_definitions += scalar_count,
            _ => detail.continuous_equations += scalar_count,
        }
    }
    Ok(())
}

fn multi_output_equation_scalar_count(
    flat: &flat::Model,
    equation: &flat::Equation,
    plan: &MultiOutputEquationPlan,
) -> Result<usize, ToDaeError> {
    let records = plan
        .records
        .iter()
        .flat_map(|record| record.plan.fields.iter().map(|field| &field.target));
    plan.outputs
        .iter()
        .flatten()
        .chain(records)
        .try_fold(0usize, |count, target| {
            count
                .checked_add(checked_shape_size(target, &flat.variables[target])?)
                .ok_or_else(|| {
                    ToDaeError::unsupported_flat(
                        "multi-output equation",
                        "receiving tuple scalar count overflowed",
                        equation.span,
                    )
                })
        })
}

fn add_algorithm_target(
    detail: &mut BalanceDetail,
    flat: &flat::Model,
    roles: &HashMap<VarName, PlannedRole>,
    target: &VarName,
) -> Result<(), ToDaeError> {
    let scalar_count = checked_shape_size(target, &flat.variables[target])?;
    match roles[target] {
        PlannedRole::State | PlannedRole::Algebraic | PlannedRole::Output => {
            detail.continuous_equations += scalar_count;
        }
        PlannedRole::DiscreteReal => detail.discrete_real_equations += scalar_count,
        PlannedRole::DiscreteValue => detail.discrete_value_definitions += scalar_count,
        PlannedRole::UnusedExpandable
        | PlannedRole::Parameter
        | PlannedRole::Constant
        | PlannedRole::Input
        | PlannedRole::Clock
        | PlannedRole::EnumerationLiteral
        | PlannedRole::Aggregate => {}
    }
    Ok(())
}

/// Count each field equation of a record equation in the system its target
/// belongs to.
fn add_record_equation_counts(
    detail: &mut BalanceDetail,
    flat: &flat::Model,
    equation: &flat::Equation,
    plan: &RecordEquationPlan,
    roles: &HashMap<VarName, PlannedRole>,
) -> Result<(), ToDaeError> {
    for (field, system) in plan.field_systems(roles) {
        let size = checked_shape_size(&field.target, &flat.variables[&field.target])?;
        let count = match system {
            RecordFieldSystem::Continuous => &mut detail.continuous_equations,
            RecordFieldSystem::DiscreteReal => &mut detail.discrete_real_equations,
            RecordFieldSystem::DiscreteValue => &mut detail.discrete_value_definitions,
        };
        *count = count.checked_add(size).ok_or_else(|| {
            ToDaeError::unsupported_flat(
                "record equation shape",
                "record field scalar count overflowed",
                equation.span,
            )
        })?;
    }
    Ok(())
}

fn checked_shape_size(name: &VarName, variable: &flat::Variable) -> Result<usize, ToDaeError> {
    variable.shape_size().map_err(|_| {
        ToDaeError::unsupported_flat(
            "variable shape",
            format!("`{name}` has a shape whose scalar cardinality cannot be represented"),
            variable.source_span,
        )
    })
}

/// Count the definitions variable bindings contribute (MLS §4.4.1).
fn add_binding_counts(
    detail: &mut BalanceDetail,
    flat: &flat::Model,
    roles: &HashMap<VarName, PlannedRole>,
    record: &mut dyn FnMut(BalanceOwner<'_>),
) -> Result<(), ToDaeError> {
    for (name, variable) in &flat.variables {
        if variable.binding.is_none() {
            continue;
        }
        let scalar_count = checked_shape_size(name, variable)?;
        let before = detail.equations();
        match roles[name] {
            PlannedRole::UnusedExpandable
            | PlannedRole::Parameter
            | PlannedRole::Constant
            | PlannedRole::Input
            | PlannedRole::Clock => {}
            PlannedRole::State | PlannedRole::Algebraic | PlannedRole::Output => {
                detail.continuous_equations += scalar_count;
            }
            PlannedRole::DiscreteReal => detail.discrete_real_equations += scalar_count,
            PlannedRole::DiscreteValue => detail.discrete_value_definitions += scalar_count,
            PlannedRole::EnumerationLiteral | PlannedRole::Aggregate => {
                unreachable!("expression-only roles are not model variables")
            }
        }
        record(BalanceOwner::Definition(name, detail.equations() - before));
    }
    Ok(())
}

/// At most this many leftover owners are named; the rest are counted.
const LISTED_LEFTOVERS: usize = 8;

/// What a maximum matching of an unbalanced model's equations to its unknowns
/// leaves over (ED001): the unknowns without an equation when there are too
/// few, the equations without an unknown when there are too many.
///
/// The matching runs on the source owners the balance counts, one row per Flat
/// equation or variable definition and one column per unknown variable, with
/// their scalar counts as capacities. A row is incident to every unknown its
/// residual names (a definition to its own variable). That relaxes the scalar
/// incidence of arrays and of `der(x)`, so the leftover is never smaller than
/// the imbalance, but which owners it names is one maximum matching among
/// several; inputs of components that are neither connected nor bound are
/// matched last, so they are the ones named when they are the cause.
pub(in crate::construction) fn balance_leftover(
    flat: &flat::Model,
    analysis: &Analysis,
    sources: &SourceMap,
) -> Result<BalanceLeftover, ToDaeError> {
    let mut excluded_equation_rows = analysis.clock_equation_rows.clone();
    excluded_equation_rows.extend(&analysis.derived_parameter_rows);
    let mut unknowns = Vec::new();
    let mut rows = Vec::new();
    source_balance_owners(
        SourceBalanceInput {
            flat,
            roles: &analysis.roles,
            assigned_targets: &analysis.assigned_discrete_targets,
            excluded_equation_rows: &excluded_equation_rows,
            record_equations: &analysis.record_equations,
            multi_output_equations: &analysis.multi_output_equations,
            connection_ranks: &analysis.discrete_connection_ranks,
            aggregate_connections: &analysis.aggregate_discrete_connections,
        },
        &mut |owner| match owner {
            BalanceOwner::Unknown(name, count) => unknowns.push((name.clone(), count)),
            BalanceOwner::Equation(row, count) if count > 0 => rows.push((Ok(row), count)),
            BalanceOwner::Definition(name, count) if count > 0 => {
                rows.push((Err(name.clone()), count));
            }
            BalanceOwner::Equation(..) | BalanceOwner::Definition(..) => {}
        },
    )?;
    // Roles and `when` targets are hashed: put both sides in source order so
    // the matching, and so the message, is deterministic.
    unknowns.sort_by_key(|(name, _)| flat.variables.get_index_of(name));
    rows.sort_by_key(|(row, _)| {
        row.as_ref()
            .copied()
            .map_err(|name| flat.variables.get_index_of(name))
    });
    let columns = unknowns
        .iter()
        .enumerate()
        .map(|(column, (name, _))| (name.as_str(), column))
        .collect::<BTreeMap<_, _>>();
    let incidence = rows
        .iter()
        .map(|(row, _)| {
            let mut names = Vec::new();
            match row {
                Ok(row) => flat.equations[*row].residual.collect_var_refs(&mut names),
                Err(name) => names.push(name.clone()),
            }
            let mut incident = names
                .iter()
                .flat_map(|name| unknown_columns(&columns, name.as_str()))
                .collect::<Vec<_>>();
            incident.sort_unstable();
            incident.dedup();
            incident
        })
        .collect::<Vec<_>>();
    let unknown_counts = unknowns.iter().map(|(_, count)| *count).collect::<Vec<_>>();
    let row_counts = rows.iter().map(|(_, count)| *count).collect::<Vec<_>>();
    let mut leftover = Vec::new();
    let heading = if analysis.balance.balance() < 0 {
        let mut adjacent = vec![Vec::new(); unknowns.len()];
        for (row, incident) in incidence.iter().enumerate() {
            for &column in incident {
                adjacent[column].push(row);
            }
        }
        let dangling = |column: usize| is_dangling_input(&flat.variables[&unknowns[column].0]);
        let order = (0..unknowns.len())
            .filter(|&column| !dangling(column))
            .chain((0..unknowns.len()).filter(|&column| dangling(column)));
        let matched = b_matching(&adjacent, &unknown_counts, &row_counts, order);
        for (column, (name, count)) in unknowns.iter().enumerate() {
            if matched[column] < *count {
                let variable = &flat.variables[name];
                leftover.push((
                    unknown_text(name, variable, *count - matched[column], *count),
                    variable.source_span,
                ));
            }
        }
        "left without an equation"
    } else {
        let matched = b_matching(&incidence, &row_counts, &unknown_counts, 0..rows.len());
        for (index, (row, count)) in rows.iter().enumerate() {
            if matched[index] < *count {
                leftover.push(row_text(flat, sources, row));
            }
        }
        "equations left over"
    };
    let more = leftover.len().saturating_sub(LISTED_LEFTOVERS);
    leftover.truncate(LISTED_LEFTOVERS);
    Ok(BalanceLeftover {
        heading,
        listed: leftover,
        more,
    })
}

/// An input of a component that nothing connects and nothing binds: its
/// equation must come from outside, and none does (MLS §4.7).
fn is_dangling_input(variable: &flat::Variable) -> bool {
    matches!(variable.causality, Causality::Input(_))
        && variable.binding.is_none()
        && !variable.connected
}

fn unknown_text(name: &VarName, variable: &flat::Variable, missing: usize, count: usize) -> String {
    let mut text = match name.as_str().rsplit_once('.') {
        Some((component, _)) if is_dangling_input(variable) => {
            format!("`{name}` is an input of `{component}` with no connection and no binding")
        }
        _ => format!("`{name}`"),
    };
    if count > 1 {
        text.push_str(&format!(" ({missing} of {count} elements)"));
    }
    text
}

fn row_text(
    flat: &flat::Model,
    sources: &SourceMap,
    row: &Result<usize, VarName>,
) -> (String, Span) {
    match row {
        Ok(row) => {
            let equation = &flat.equations[*row];
            let text = sources
                .get_source(equation.span.source)
                .and_then(|(_, text)| text.get(equation.span.start.0..equation.span.end.0))
                .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|text| !text.is_empty())
                .map_or_else(String::new, |text| {
                    format!("`{}` ", text.chars().take(80).collect::<String>())
                });
            (format!("{text}({})", equation.origin), equation.span)
        }
        Err(name) => {
            let variable = &flat.variables[name];
            match variable.binding.as_ref().and_then(Expression::span) {
                Some(span) => (format!("binding of `{name}`"), span),
                None => (format!("assignment to `{name}`"), variable.source_span),
            }
        }
    }
}

/// The unknown columns a reference names: itself, the fields or elements of
/// the record or array it names whole, or the unknown it is a part of.
fn unknown_columns(columns: &BTreeMap<&str, usize>, name: &str) -> Vec<usize> {
    if let Some(&column) = columns.get(name) {
        return vec![column];
    }
    let parts = columns
        .range(name..)
        .take_while(|(candidate, _)| candidate.starts_with(name))
        .filter(|(candidate, _)| candidate[name.len()..].starts_with(['.', '[']))
        .map(|(_, &column)| column)
        .collect::<Vec<_>>();
    if !parts.is_empty() {
        return parts;
    }
    name.rmatch_indices(['.', '['])
        .find_map(|(at, _)| columns.get(&name[..at]).copied())
        .into_iter()
        .collect()
}

/// A maximum b-matching found greedily in `order`: each left node takes up to
/// its capacity in units along augmenting paths, and a node never gives a unit
/// back, so the nodes late in `order` are the ones left short. Returns the
/// units each left node got.
fn b_matching(
    adjacent: &[Vec<usize>],
    left: &[usize],
    right: &[usize],
    order: impl Iterator<Item = usize>,
) -> Vec<usize> {
    let mut held = vec![Vec::new(); right.len()];
    let mut matched = vec![0; left.len()];
    for node in order {
        while matched[node] < left[node] {
            let mut seen = vec![false; right.len()];
            if !augment(node, adjacent, right, &mut held, &mut seen) {
                break;
            }
            matched[node] += 1;
        }
    }
    matched
}

/// Give `node` one more unit, moving the holders of full right nodes along.
fn augment(
    node: usize,
    adjacent: &[Vec<usize>],
    right: &[usize],
    held: &mut [Vec<usize>],
    seen: &mut [bool],
) -> bool {
    for &target in &adjacent[node] {
        if std::mem::replace(&mut seen[target], true) {
            continue;
        }
        if held[target].len() < right[target] {
            held[target].push(node);
            return true;
        }
        for unit in 0..held[target].len() {
            if augment(held[target][unit], adjacent, right, held, seen) {
                held[target][unit] = node;
                return true;
            }
        }
    }
    false
}
