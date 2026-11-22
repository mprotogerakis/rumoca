//! A continuous-time model discretized as MLS §16.8.1 does a clocked partition with
//! `solverMethod = "ExplicitEuler"`, for code run once a period (the GALEC projection takes
//! sampled models only): its states one explicit Euler step a period, its events and its time
//! seen at the ticks. A flat model in, a flat model of discrete-time variables out:
//!
//! - each state `x` discrete, `x = pre(x) + period*'der.x'` at each tick, `'der.x'` an
//!   algebraic variable its derivative's equation now defines; everywhere else `x` read as
//!   `pre(x)`, the state at the tick's start;
//! - `time` a discrete variable, 0 at the first tick, `period` on at each;
//! - `initial()` true at the first tick; `edge(b)`, `change(v)`, and `pre(v)` of a variable no
//!   `when` sets, against a memory of it set at each tick;
//! - a `when` on another condition than a sample: at each tick, its equations taking effect
//!   where an element of its condition rose since the last tick (`v = if rose then e else
//!   pre(v)`); its `reinit` of a state that tick's value instead of the Euler step; a `when`
//!   on samples stays as it is;
//! - an initial equation `x = e` of a state, `e` of parameters and constants: its start value.
//!
//! Refused, said: `delay`, `terminal`, `terminate`; an algorithm over
//! a state or `time`; another initial equation, an initial algorithm; in a `when` on another
//! condition than a sample, an if-equation or a call with several outputs.

use rumoca_core::{
    BuiltinFunction, Causality, Expression, ExpressionVisitor, Literal, OpBinary, OpUnary, Reference, Span, Subscript, VarName,
    Variability,
};
use rumoca_ir_flat as flat;
use std::collections::{HashMap, HashSet};

use crate::ToDaeError;

thread_local! {
    /// Where what the discretization makes is said to come from: a source span of the model (a
    /// flat model's every node has one).
    static AT: std::cell::Cell<Span> = const { std::cell::Cell::new(Span::DUMMY) };
}
fn at() -> Span {
    AT.with(|a| a.get())
}

fn refuse(what: &str, detail: impl Into<String>, span: Span) -> ToDaeError {
    ToDaeError::unsupported_flat(what, detail.into(), span)
}

fn var(name: &str) -> Expression {
    Expression::VarRef { name: Reference::new(name), subscripts: vec![], span: at() }
}
fn literal(value: Literal) -> Expression {
    Expression::Literal { value, span: at() }
}
fn builtin(function: BuiltinFunction, args: Vec<Expression>) -> Expression {
    Expression::BuiltinCall { function, args, span: at() }
}
fn pre(e: Expression) -> Expression {
    builtin(BuiltinFunction::Pre, vec![e])
}
fn binary(op: OpBinary, lhs: Expression, rhs: Expression) -> Expression {
    Expression::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: at() }
}
fn not(e: Expression) -> Expression {
    Expression::Unary { op: OpUnary::Not, rhs: Box::new(e), span: at() }
}
fn select(branches: Vec<(Expression, Expression)>, otherwise: Expression) -> Expression {
    if branches.is_empty() {
        return otherwise;
    }
    Expression::If { branches, else_branch: Box::new(otherwise), span: at() }
}
fn any(terms: Vec<Expression>) -> Expression {
    terms.into_iter().reduce(|a, b| binary(OpBinary::Or, a, b)).unwrap_or(literal(Literal::Boolean(false)))
}

/// An expression rebuilt: `f` replaces a node (and nothing below it), else its children are.
fn rewrite(e: &mut Expression, f: &mut dyn FnMut(&Expression) -> Result<Option<Expression>, ToDaeError>) -> Result<(), ToDaeError> {
    if let Some(new) = f(e)? {
        *e = new;
        return Ok(());
    }
    match e {
        Expression::Binary { lhs, rhs, .. } => {
            rewrite(lhs, f)?;
            rewrite(rhs, f)?;
        }
        Expression::Unary { rhs, .. } => rewrite(rhs, f)?,
        Expression::BuiltinCall { args, .. } | Expression::FunctionCall { args, .. } => {
            for a in args {
                rewrite(a, f)?;
            }
        }
        Expression::StringConversion { value, .. } => rewrite(value, f)?,
        Expression::If { branches, else_branch, .. } => {
            for (c, v) in branches {
                rewrite(c, f)?;
                rewrite(v, f)?;
            }
            rewrite(else_branch, f)?;
        }
        Expression::Array { elements, .. } | Expression::Tuple { elements, .. } => {
            for x in elements {
                rewrite(x, f)?;
            }
        }
        Expression::Range { start, step, end, .. } => {
            rewrite(start, f)?;
            if let Some(s) = step {
                rewrite(s, f)?;
            }
            rewrite(end, f)?;
        }
        Expression::ArrayComprehension { expr, filter, .. } => {
            rewrite(expr, f)?;
            if let Some(x) = filter {
                rewrite(x, f)?;
            }
        }
        Expression::Index { base, .. } | Expression::FieldAccess { base, .. } => rewrite(base, f)?,
        Expression::VarRef { .. } | Expression::Literal { .. } | Expression::Empty { .. } => {}
    }
    Ok(())
}

/// What a discretization adds: its variables (name, kind, start) and the equations of its tick.
#[derive(Default)]
struct Added {
    vars: Vec<(String, Kind, Expression)>,
    tick: Vec<flat::WhenEquation>,
    taken: HashSet<String>,
    first: Option<String>,
    time: Option<String>,
    /// a discrete variable's value before the first tick (an initial equation `pre(v) = e`, else
    /// its start): its memory's start
    pre_starts: HashMap<String, Expression>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Real,
    Integer,
    Boolean,
}

impl Added {
    /// A name of its own: under a reserved word (`der`, `pre`, `initial`, `time`) no component
    /// can have, numbered on where taken.
    fn fresh(&mut self, want: &str) -> String {
        let mut name = want.to_string();
        let mut k = 1;
        while self.taken.contains(&name) {
            k += 1;
            name = format!("{want}{k}");
        }
        self.taken.insert(name.clone());
        name
    }

    fn assign(&mut self, target: &str, value: Expression, origin: &str) {
        self.tick.push(flat::WhenEquation::Assign { target: VarName::new(target), value, span: at(), origin: origin.to_string() });
    }

    /// A memory of `value` (its kind; its value before the first tick, else 0 or false), set at
    /// each tick: its name.
    fn memory(&mut self, what: &str, kind: Kind, value: Expression, before: Option<Expression>) -> String {
        let name = self.fresh(&format!("pre.{what}"));
        let start = before.unwrap_or_else(|| match kind {
            Kind::Real => literal(Literal::Real(0.0)),
            Kind::Integer => literal(Literal::Integer(0)),
            Kind::Boolean => literal(Literal::Boolean(false)),
        });
        self.vars.push((name.clone(), kind, start));
        self.assign(&name, value, "the memory of a discretized model's event");
        name
    }

    /// `pre(first)`: true at the first tick.
    fn initial(&mut self) -> Expression {
        let name = match &self.first {
            Some(n) => n.clone(),
            None => {
                let n = self.fresh("initial.first");
                self.vars.push((n.clone(), Kind::Boolean, literal(Literal::Boolean(true))));
                self.assign(&n, literal(Literal::Boolean(false)), "a discretized model's first tick");
                self.first = Some(n.clone());
                n
            }
        };
        pre(var(&name))
    }

    /// The time at this tick: 0 at the first, `period` on at each.
    fn time(&mut self, period: f64) -> Expression {
        let name = match &self.time {
            Some(n) => n.clone(),
            None => {
                let n = self.fresh("time.tick");
                self.vars.push((n.clone(), Kind::Real, literal(Literal::Real(-period))));
                self.assign(&n, binary(OpBinary::Add, pre(var(&n)), literal(Literal::Real(period))), "a discretized model's time");
                self.time = Some(n.clone());
                n
            }
        };
        var(&name)
    }
}

/// The variables an algorithm reads and sets.
#[derive(Default)]
struct Refs(Vec<VarName>);
impl ExpressionVisitor for Refs {
    fn visit_var_ref(&mut self, name: &Reference, subscripts: &[Subscript]) {
        self.0.push(name.var_name().clone());
        self.walk_var_ref(name, subscripts);
    }
}
impl flat::StatementVisitor for Refs {}

struct Scope<'a> {
    states: &'a HashSet<VarName>,
    when_set: &'a HashSet<String>,
    kinds: &'a HashMap<String, Kind>,
    period: f64,
}

/// An expression as the discretized model reads it at a tick (see the module's doc).
fn observe(e: &mut Expression, scope: &Scope, added: &mut Added) -> Result<(), ToDaeError> {
    rewrite(e, &mut |node| {
        Ok(match node {
            Expression::BuiltinCall { function: BuiltinFunction::Der, args, span } => match args.as_slice() {
                [Expression::VarRef { name, subscripts, .. }] if scope.states.contains(name.var_name()) => {
                    Some(Expression::VarRef { name: Reference::new(format!("der.{}", name.var_name().as_str())), subscripts: subscripts.clone(), span: *span })
                }
                _ => return Err(refuse("discretization", "der() of an expression: only der() of a state is discretized", *span)),
            },
            Expression::VarRef { name, subscripts, span } if scope.states.contains(name.var_name()) => {
                Some(pre(Expression::VarRef { name: name.clone(), subscripts: subscripts.clone(), span: *span }))
            }
            Expression::VarRef { name, subscripts, .. } if name.var_name().as_str() == "time" && subscripts.is_empty() => Some(added.time(scope.period)),
            Expression::BuiltinCall { function: BuiltinFunction::Initial, .. } => Some(added.initial()),
            Expression::BuiltinCall { function: f @ (BuiltinFunction::Delay | BuiltinFunction::Terminal), span, .. } => {
                return Err(refuse("discretization", format!("{}() in a model exported as code run once a period", f.name()), *span));
            }
            Expression::BuiltinCall { function: BuiltinFunction::Edge, args, span } => {
                let [b] = args.as_slice() else { return Err(refuse("discretization", "edge() of one argument", *span)) };
                let mut b = b.clone();
                observe(&mut b, scope, added)?;
                let m = added.memory("edge", Kind::Boolean, b.clone(), None);
                Some(binary(OpBinary::And, b, not(pre(var(&m)))))
            }
            Expression::BuiltinCall { function: BuiltinFunction::Change, args, span } => {
                let [v] = args.as_slice() else { return Err(refuse("discretization", "change() of one argument", *span)) };
                let kind = match v {
                    Expression::VarRef { name, subscripts, .. } if subscripts.is_empty() => scope.kinds.get(name.var_name().as_str()).copied(),
                    _ => None,
                }
                .ok_or_else(|| refuse("discretization", "change() of a scalar variable only", *span))?;
                let mut v = v.clone();
                observe(&mut v, scope, added)?;
                let m = added.memory("change", kind, v.clone(), None);
                Some(binary(OpBinary::Neq, v, pre(var(&m))))
            }
            // (pre of what no `when` sets, a Boolean an equation defines: against its memory)
            Expression::BuiltinCall { function: BuiltinFunction::Pre, args, span } => match args.as_slice() {
                [Expression::VarRef { name, subscripts, .. }] if subscripts.is_empty() && !scope.when_set.contains(name.var_name().as_str()) && !scope.states.contains(name.var_name()) => {
                    let n = name.var_name().as_str();
                    let kind = scope.kinds.get(n).copied().ok_or_else(|| refuse("discretization", format!("pre() of {n}"), *span))?;
                    let before = added.pre_starts.get(n).cloned();
                    let m = added.memory(&format!("of.{n}"), kind, var(n), before);
                    Some(pre(var(&m)))
                }
                _ => None,
            },
            _ => None,
        })
    })
}

/// Whether a `when` condition is a sample (or samples): such a `when` is clocked already.
fn is_sample(c: &Expression) -> bool {
    match c {
        Expression::BuiltinCall { function: BuiltinFunction::Sample, args, .. } => args.len() == 2,
        Expression::Array { elements, .. } => !elements.is_empty() && elements.iter().all(is_sample),
        _ => false,
    }
}

/// A `when` condition's rise at this tick: any of its elements true now, false at the last tick
/// (`initial()`: the first tick; a sample of the period: every tick).
fn rose(c: &Expression, scope: &Scope, added: &mut Added) -> Result<Expression, ToDaeError> {
    let elements = match c {
        Expression::Array { elements, .. } => elements.clone(),
        c => vec![c.clone()],
    };
    let mut terms = vec![];
    for el in elements {
        match &el {
            Expression::BuiltinCall { function: BuiltinFunction::Initial, .. } => terms.push(added.initial()),
            Expression::BuiltinCall { function: BuiltinFunction::Sample, args, span } => {
                let number = |e: &Expression| match e {
                    Expression::Literal { value: Literal::Real(v), .. } => Some(*v),
                    Expression::Literal { value: Literal::Integer(v), .. } => Some(*v as f64),
                    _ => None,
                };
                let every = matches!(args.as_slice(), [a, b] if number(a) == Some(0.0) && number(b).is_some_and(|p| (p - scope.period).abs() <= 1e-12 * scope.period.max(1.0)));
                if !every {
                    return Err(refuse("discretization", "a when on a sample of another period beside conditions of its own", *span));
                }
                terms.push(literal(Literal::Boolean(true)));
            }
            // (an edge already: that, not a rise of it)
            Expression::BuiltinCall { function: BuiltinFunction::Edge, .. } => {
                let mut b = el.clone();
                observe(&mut b, scope, added)?;
                terms.push(b);
            }
            _ => {
                let mut b = el.clone();
                observe(&mut b, scope, added)?;
                // (a condition true from the start no rise: as a when, which only a change fires)
                let m = added.memory("when", Kind::Boolean, b.clone(), None);
                let first = added.initial();
                terms.push(binary(OpBinary::And, binary(OpBinary::And, b, not(pre(var(&m)))), not(first)));
            }
        }
    }
    Ok(any(terms))
}

/// The flat model `model` discretized at `period` seconds by explicit Euler (see the module).
pub fn explicit_euler(model: &flat::Model, period: f64) -> Result<flat::Model, ToDaeError> {
    if !(period > 0.0 && period.is_finite()) {
        return Err(refuse("discretization", format!("a period of {period} s"), at()));
    }
    let mut m = model.clone();
    let anchor = m.equations.first().map(|e| e.span).or_else(|| m.variables.values().next().map(|v| v.source_span)).unwrap_or(Span::DUMMY);
    AT.with(|a| a.set(anchor));
    // the states, in their order
    let mut found: Vec<VarName> = vec![];
    for eq in &m.equations {
        eq.residual.collect_state_variables(&mut found);
    }
    let mut states: HashSet<VarName> = HashSet::new();
    let order: Vec<VarName> = found.into_iter().filter(|x| states.insert(x.clone())).collect();
    let when_set: HashSet<String> = m
        .when_chains
        .iter()
        .flat_map(|c| c.branches())
        .flat_map(|b| b.equations.iter())
        .filter_map(|e| match e {
            flat::WhenEquation::Assign { target, .. } => Some(target.as_str().to_string()),
            flat::WhenEquation::AssignElement { target, .. } => Some(target.var_name().as_str().to_string()),
            _ => None,
        })
        .collect();
    // (a variable's kind: its effective type's canonical one, a predefined scalar type)
    let p = m.predefined_types;
    let kind_of = |t: rumoca_core::TypeId| -> Option<Kind> {
        let c = m.effective_types.get(&t).map(|e| e.canonical_type())?;
        if c == p.real { Some(Kind::Real) } else if c == p.integer { Some(Kind::Integer) } else if c == p.boolean { Some(Kind::Boolean) } else { None }
    };
    let kinds: HashMap<String, Kind> = m.variables.iter().filter_map(|(n, v)| Some((n.as_str().to_string(), kind_of(v.type_id)?))).collect();
    let mut added = Added { taken: m.variables.keys().map(|n| n.as_str().to_string()).collect(), ..Default::default() };
    let scope = Scope { states: &states, when_set: &when_set, kinds: &kinds, period };

    // an algorithm over the states or time: not yet
    for alg in &m.algorithms {
        let mut c = Refs::default();
        for st in &alg.statements {
            flat::StatementVisitor::visit_statement(&mut c, st);
        }
        if let Some(n) = c.0.iter().find(|n| states.contains(*n) || n.as_str() == "time") {
            return Err(refuse("discretization", format!("an algorithm reading {} (a state or time): not exported as code yet", n.as_str()), alg.span));
        }
    }
    if let Some(a) = m.initial_algorithms.first() {
        return Err(refuse("discretization", "an initial algorithm: not exported as code", a.span));
    }
    // an initial equation x = e of a state, e of parameters: its start value
    let fixed_only = |e: &Expression| {
        let mut refs: Vec<VarName> = vec![];
        e.collect_var_refs(&mut refs);
        refs.iter().all(|n| m.variables.get(n).is_some_and(|v| matches!(v.variability, Variability::Parameter(_) | Variability::Constant(_))))
    };
    let mut starts: Vec<(VarName, Expression)> = vec![];
    let mut pre_starts: Vec<(VarName, Expression)> = vec![];
    for eq in &m.initial_equations {
        let pre_of = |e: &Expression| match e {
            Expression::BuiltinCall { function: BuiltinFunction::Pre, args, .. } => match args.as_slice() {
                [Expression::VarRef { name, subscripts, .. }] if subscripts.is_empty() => Some(name.var_name().clone()),
                _ => None,
            },
            _ => None,
        };
        let found = match &eq.residual {
            Expression::Binary { op: OpBinary::Sub, lhs, rhs, .. } => match (lhs.as_ref(), rhs.as_ref()) {
                (Expression::VarRef { name, subscripts, .. }, e) | (e, Expression::VarRef { name, subscripts, .. })
                    if subscripts.is_empty() && states.contains(name.var_name()) && fixed_only(e) => Some((name.var_name().clone(), e.clone(), false)),
                (p, e) | (e, p) if pre_of(p).is_some() && fixed_only(e) => Some((pre_of(p).unwrap(), e.clone(), true)),
                _ => None,
            },
            _ => None,
        };
        match found {
            Some((v, e, true)) => {
                added.pre_starts.insert(v.as_str().to_string(), e.clone());
                pre_starts.push((v, e));
            }
            Some((x, e, false)) => starts.push((x, e)),
            None => return Err(refuse("discretization", "an initial equation other than a state's or a pre-value's start of parameters (x = x0, pre(y) = y0): not exported as code; set the state's start, or initType = NoInit", eq.span)),
        }
    }
    m.initial_equations.clear();
    m.initial_structured_equations.clear();
    // (a state's start; a when-set variable's pre-value its start too, its value before the first tick)
    for (x, e) in starts.into_iter().chain(pre_starts.into_iter().filter(|(v, _)| when_set.contains(v.as_str()))) {
        if let Some(v) = m.variables.get_mut(&x) {
            v.start = Some(e);
            v.fixed = Some(vec![true]);
        }
    }

    // the equations: as read at a tick
    let mut equations = std::mem::take(&mut m.equations);
    for eq in &mut equations {
        observe(&mut eq.residual, &scope, &mut added)?;
    }
    m.equations = equations;
    m.structured_equations.clear();

    // the whens: on samples as they are (read at a tick); on anything else, at each tick where
    // their condition rose
    let mut kept = vec![];
    let mut on: Vec<(String, Vec<(Expression, Expression)>)> = vec![]; // (target, its branches)
    let mut on_element: Vec<(Reference, Vec<rumoca_core::Subscript>, Vec<(Expression, Expression)>)> = vec![];
    let mut reinits: HashMap<VarName, Vec<(Expression, Expression)>> = HashMap::new();
    let mut asserts: Vec<flat::WhenEquation> = vec![];
    for mut chain in std::mem::take(&mut m.when_chains) {
        if chain.branches().all(|b| is_sample(&b.condition)) {
            for b in chain.branches_mut() {
                for e in &mut b.equations {
                    match e {
                        flat::WhenEquation::Assign { value, .. } | flat::WhenEquation::AssignElement { value, .. } => observe(value, &scope, &mut added)?,
                        flat::WhenEquation::Reinit { span, .. } => return Err(refuse("discretization", "reinit in a when on samples: not exported as code", *span)),
                        _ => {}
                    }
                }
            }
            kept.push(chain);
            continue;
        }
        for b in chain.branches() {
            let r = rose(&b.condition, &scope, &mut added)?;
            for e in &b.equations {
                match e {
                    flat::WhenEquation::Assign { target, value, .. } => {
                        let mut v = value.clone();
                        observe(&mut v, &scope, &mut added)?;
                        match on.iter_mut().find(|(t, _)| t == target.as_str()) {
                            Some((_, bs)) => bs.push((r.clone(), v)),
                            None => on.push((target.as_str().to_string(), vec![(r.clone(), v)])),
                        }
                    }
                    flat::WhenEquation::AssignElement { target, subscripts, value, .. } => {
                        let mut v = value.clone();
                        observe(&mut v, &scope, &mut added)?;
                        match on_element.iter_mut().find(|(t, s, _)| t.var_name() == target.var_name() && format!("{s:?}") == format!("{subscripts:?}")) {
                            Some((_, _, bs)) => bs.push((r.clone(), v)),
                            None => on_element.push((target.clone(), subscripts.clone(), vec![(r.clone(), v)])),
                        }
                    }
                    flat::WhenEquation::Reinit { state, value, span, .. } => {
                        if !states.contains(state) {
                            return Err(refuse("discretization", format!("reinit of {}, no state", state.as_str()), *span));
                        }
                        let mut v = value.clone();
                        observe(&mut v, &scope, &mut added)?;
                        reinits.entry(state.clone()).or_default().push((r.clone(), v));
                    }
                    flat::WhenEquation::Assert { condition, message, level, span, origin } => {
                        let mut c = condition.clone();
                        observe(&mut c, &scope, &mut added)?;
                        asserts.push(flat::WhenEquation::Assert { condition: binary(OpBinary::Or, not(r.clone()), c), message: message.clone(), level: level.clone(), span: *span, origin: origin.clone() });
                    }
                    other => return Err(refuse("discretization", "an if-equation, a call of several outputs or terminate() in a when: not exported as code", other.span())),
                }
            }
        }
    }
    m.when_chains = kept;

    // the tick: each state's Euler step (or its reinit), each when's equations, the memories
    for x in &order {
        let n = x.as_str();
        let step = binary(OpBinary::Add, pre(var(n)), binary(OpBinary::Mul, literal(Literal::Real(period)), var(&format!("der.{n}"))));
        let value = select(reinits.remove(x).unwrap_or_default(), step);
        added.assign(n, value, "a discretized state's Euler step");
    }
    for (target, branches) in on {
        added.assign(&target, select(branches, pre(var(&target))), "a discretized when");
    }
    for (target, subscripts, branches) in on_element {
        let old = pre(Expression::VarRef { name: target.clone(), subscripts: subscripts.clone(), span: at() });
        added.tick.push(flat::WhenEquation::AssignElement { target, subscripts, value: select(branches, old), span: at(), origin: "a discretized when".into() });
    }
    added.tick.extend(asserts);

    // the variables: the states discrete; their derivatives, the memories, first and time new
    let template = m.variables.values().next().cloned();
    // (each new variable an occurrence of its own)
    let mut next_id = m.variables.values().map(|v| v.instance_id.0).chain(m.record_instances.values().map(|r| r.instance_id.0)).max().unwrap_or(0);
    let mut fresh_id = || {
        next_id += 1;
        rumoca_core::InstanceId::new(next_id)
    };
    for x in &order {
        let Some(v) = m.variables.get_mut(x) else { continue };
        v.variability = Variability::Discrete(Default::default());
        let mut d = v.clone();
        d.name = VarName::new(format!("der.{}", x.as_str()));
        d.variability = Variability::Continuous(Default::default());
        d.causality = Causality::Empty;
        d.start = None;
        d.fixed = None;
        d.binding = None;
        d.binding_from_modification = false;
        d.min = None;
        d.max = None;
        d.nominal = None;
        d.component_ref = None;
        d.description = Some(format!("The derivative of {}", x.as_str()));
        d.is_protected = true;
        d.instance_id = fresh_id();
        let type_name = m.variable_type_names.get(x).cloned();
        m.variables.insert(d.name.clone(), d.clone());
        if let Some(t) = type_name {
            m.variable_type_names.insert(d.name.clone(), t);
        }
    }
    if let Some(t) = template {
        for (name, kind, start) in &added.vars {
            let mut v = t.clone();
            v.name = VarName::new(name.clone());
            let (canonical, type_name) = match kind {
                Kind::Real => (m.predefined_types.real, "Real"),
                Kind::Integer => (m.predefined_types.integer, "Integer"),
                Kind::Boolean => (m.predefined_types.boolean, "Boolean"),
            };
            // (a scalar effective type of that kind: one the model has, else a new one)
            let found = m.effective_types.iter().find(|(_, e)| e.canonical_type() == canonical && e.nominal_type() == canonical && e.dimensions().is_empty()).map(|(k, _)| *k);
            let type_id = match found {
                Some(k) => k,
                None => {
                    let key = rumoca_core::TypeId::new(m.effective_types.keys().map(|k| k.0).filter(|k| *k != u32::MAX).max().unwrap_or(0) + 1);
                    let e = rumoca_core::EffectiveType::new(canonical, canonical, Vec::<i64>::new()).map_err(|e| refuse("discretization", format!("a {type_name}'s type: {e:?}"), at()))?;
                    m.effective_types.insert(key, e);
                    key
                }
            };
            v.type_id = type_id;
            v.is_discrete_type = *kind != Kind::Real;
            v.is_primitive = true;
            v.variability = Variability::Discrete(Default::default());
            v.causality = Causality::Empty;
            v.dims = vec![];
            v.start = Some(start.clone());
            v.fixed = Some(vec![true]);
            v.binding = None;
            v.binding_from_modification = false;
            v.min = None;
            v.max = None;
            v.nominal = None;
            v.unit = None;
            v.display_unit = None;
            v.quantity = None;
            v.component_ref = None;
            v.description = Some("Of the discretization".into());
            v.is_protected = true;
            v.instance_id = fresh_id();
            m.variables.insert(v.name.clone(), v);
            m.variable_type_names.insert(VarName::new(name.clone()), type_name.to_string());
        }
    }
    if !added.tick.is_empty() {
        let mut branch = flat::WhenBranch::new(builtin(BuiltinFunction::Sample, vec![literal(Literal::Integer(0)), literal(Literal::Real(period))]), at());
        branch.equations = added.tick;
        m.when_chains.push(flat::WhenChain::new(branch, at()));
    }
    Ok(m)
}
