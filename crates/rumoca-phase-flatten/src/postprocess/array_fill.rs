//! MLS 3.7 §10.3.3: `fill(s, n1, ..., nk)` of an array `s` is the
//! `n1 x ... x nk` array whose every element is `s`, of shape
//! `{n1, ..., nk, size(s)}`. The DAE `fill` takes a scalar value, so an array
//! value is spelled as the array constructor of its copies, which states that
//! same shape (`fill(C, 2)` becomes `{C, C}`).

use super::*;
use rumoca_core::ExpressionRewriter;

pub(crate) fn spell_array_valued_fills(flat: &mut flat::Model, ctx: &Context) {
    let arrays = flat
        .variables
        .iter()
        .filter(|(_, variable)| !variable.dims.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    let mut speller = ArrayFillSpeller {
        arrays,
        extents: ParamEvaluator::new(&ctx.param_eval_context(None)),
    };
    for equation in flat
        .equations
        .iter_mut()
        .chain(flat.initial_equations.iter_mut())
    {
        equation.residual = speller.rewrite_expression(&equation.residual);
    }
    for family in flat
        .structured_equations
        .iter_mut()
        .chain(flat.initial_structured_equations.iter_mut())
    {
        if let Some(template) = family.template.as_mut() {
            template.body = speller.rewrite_expressions(&template.body);
        }
    }
    for variable in flat.variables.values_mut() {
        for slot in [
            &mut variable.binding,
            &mut variable.start,
            &mut variable.min,
            &mut variable.max,
            &mut variable.nominal,
        ] {
            if let Some(expression) = slot.as_mut() {
                *expression = speller.rewrite_expression(expression);
            }
        }
    }
}

struct ArrayFillSpeller<'a> {
    arrays: rustc_hash::FxHashSet<rumoca_core::VarName>,
    /// MLS §10.1: an extent is evaluable at translation.
    extents: ParamEvaluator<'a>,
}

impl ArrayFillSpeller<'_> {
    fn is_array(&self, expression: &rumoca_core::Expression) -> bool {
        match expression {
            rumoca_core::Expression::Array { .. } => true,
            // A whole array variable, or one read through a slice (`X[1:2]`).
            rumoca_core::Expression::VarRef {
                name, subscripts, ..
            } => {
                self.arrays.contains(name.var_name())
                    && (subscripts.is_empty()
                        || subscripts.iter().any(|subscript| match subscript {
                            rumoca_core::Subscript::Colon { .. } => true,
                            rumoca_core::Subscript::Expr { expr, .. } => {
                                matches!(**expr, rumoca_core::Expression::Range { .. })
                            }
                            rumoca_core::Subscript::Index { .. } => false,
                        }))
            }
            _ => false,
        }
    }
}

impl ExpressionRewriter for ArrayFillSpeller<'_> {
    fn rewrite_expression(&mut self, expr: &rumoca_core::Expression) -> rumoca_core::Expression {
        let expr = self.walk_expression(expr);
        let rumoca_core::Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Fill,
            args,
            span,
        } = &expr
        else {
            return expr;
        };
        let Some((value, extents)) = args.split_first() else {
            return expr;
        };
        if !self.is_array(value) {
            return expr;
        }
        // ponytail: an empty extent keeps `fill`; its array value has no copy
        // to spell the element shape with.
        let Some(extents) = extents
            .iter()
            .map(|extent| {
                let extent = self.extents.eval_integer(extent, None)?;
                usize::try_from(extent).ok().filter(|extent| *extent > 0)
            })
            .collect::<Option<Vec<_>>>()
        else {
            return expr;
        };
        extents
            .iter()
            .rev()
            .fold(value.clone(), |element, &extent| {
                rumoca_core::Expression::Array {
                    elements: vec![element; extent],
                    kind: rumoca_core::ArrayConstructor::Array,
                    span: *span,
                }
            })
    }
}
