//! The model part canonical DAE construction is building, for ED020.
//!
//! A construction check fails with only the span of the rejected operation,
//! which often lies in a library class instantiated many times. Each owner
//! loop records the owner it is building here (a cheap copy), and clears it
//! when the loop is done, so a failure can name the flat variable, equation,
//! or function specialization it arose in. Construction runs on one thread,
//! so a thread-local slot needs no threading through the lowering signatures.

use std::cell::Cell;

use super::*;

#[derive(Clone, Copy)]
pub(super) enum Building {
    /// The declaration of the flat variable at this index.
    Variable(usize),
    /// The binding equation of the flat variable at this index.
    Binding(usize),
    /// A flat equation row, of the initial equations when `initial`.
    Equation { initial: bool, row: usize },
    /// The function specialization at this certificate index.
    Function(usize),
}

thread_local! {
    static BUILDING: Cell<Option<Building>> = const { Cell::new(None) };
}

/// Record the owner construction is building now; `None` once a loop is done.
pub(super) fn building(owner: Option<Building>) {
    BUILDING.set(owner);
}

/// The owner a failed construction was building, named in model terms with
/// its source span, and cleared for the next construction.
pub(super) fn failed_owner(flat: &flat::Model, analysis: &Analysis) -> Option<(String, Span)> {
    let owner = BUILDING.take()?;
    match owner {
        Building::Variable(index) => {
            let (name, variable) = flat.variables.get_index(index)?;
            Some((format!("the declaration of `{name}`"), variable.source_span))
        }
        Building::Binding(index) => {
            let (name, variable) = flat.variables.get_index(index)?;
            let span = variable.binding.as_ref().and_then(Expression::span);
            Some((
                format!("the binding equation of `{name}`"),
                span.unwrap_or(variable.source_span),
            ))
        }
        Building::Equation { initial, row } => {
            let (equations, prefix) = if initial {
                (&flat.initial_equations, "initial ")
            } else {
                (&flat.equations, "")
            };
            let equation = equations.get(row)?;
            Some((format!("{prefix}{}", equation.origin), equation.span))
        }
        Building::Function(index) => {
            let certificate = analysis.function_shapes.certificates().get(index)?;
            let key = &certificate.key;
            let inputs = key
                .inputs
                .iter()
                .map(|shape| match shape.as_slice() {
                    [] => "scalar".to_string(),
                    extents => format!("{extents:?}"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            let span = flat
                .functions
                .get(&key.function)
                .map(|function| function.span);
            Some((
                format!("function `{}` specialized for ({inputs})", key.function),
                span.unwrap_or(Span::DUMMY),
            ))
        }
    }
}
