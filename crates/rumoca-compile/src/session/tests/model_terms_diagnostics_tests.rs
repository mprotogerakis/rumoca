//! ToDae diagnostics that say what is wrong in model terms: the instance paths
//! an unbalanced model leaves over, with their source spans.

use super::*;

fn todae_failure(source: &str, model: &str) -> (String, Option<String>, Vec<CommonDiagnostic>) {
    let mut session = Session::default();
    session
        .add_document("Model.mo", source)
        .expect("source parses");
    match session.compile_model_phases(model).expect("phase result") {
        PhaseResult::Failed {
            phase: FailedPhase::ToDae,
            error,
            error_code,
            diagnostics,
        } => (error, error_code, diagnostics),
        other => panic!("expected a ToDae failure, got {other:?}"),
    }
}

fn labelled<'a>(source: &'a str, diagnostic: &CommonDiagnostic) -> Vec<(&'a str, String)> {
    diagnostic
        .labels
        .iter()
        .map(|label| {
            (
                &source[label.span.start.0..label.span.end.0],
                label.message.clone().unwrap_or_default(),
            )
        })
        .collect()
}

const GAIN: &str = "
block Gain
  input Real u;
  output Real y;
equation
  y = 2*u;
end Gain;
";

#[test]
fn unbalanced_model_names_the_unconnected_unbound_input_of_a_component() {
    let source = format!(
        "{GAIN}
model Open
  Gain g;
  Real x(start = 0, fixed = true);
equation
  der(x) = g.y;
end Open;
"
    );
    let (error, code, diagnostics) = todae_failure(&source, "Open");
    assert_eq!(code.as_deref(), Some("ED001"));
    assert!(
        error.starts_with("unbalanced model: 2 equations, 3 unknowns (balance = -1); "),
        "{error}"
    );
    let named = "`g.u` is an input of `g` with no connection and no binding";
    assert!(
        error.ends_with(&format!("; left without an equation: {named}")),
        "{error}"
    );
    let labels = labelled(&source, &diagnostics[0]);
    assert_eq!(labels.len(), 1, "{labels:?}");
    assert!(labels[0].0.contains("Real u"), "{labels:?}");
    assert_eq!(labels[0].1, named);
}

#[test]
fn unbalanced_model_names_a_plain_unknown_without_an_equation() {
    let source = "
model Short
  Real x;
  Real y;
equation
  x = 1;
end Short;
";
    let (error, _, diagnostics) = todae_failure(source, "Short");
    assert!(
        error.ends_with("; left without an equation: `y`"),
        "{error}"
    );
    assert_eq!(labelled(source, &diagnostics[0])[0].1, "`y`");
}

#[test]
fn overdetermined_model_names_the_equation_left_over() {
    let source = "
model Over
  Real x;
  Real y;
equation
  x = 1;
  y = 2;
  x + y = 3;
end Over;
";
    let (error, code, diagnostics) = todae_failure(source, "Over");
    assert_eq!(code.as_deref(), Some("ED001"));
    assert!(
        error.ends_with("; equations left over: `x + y = 3` (top-level model equation)"),
        "{error}"
    );
    assert_eq!(labelled(source, &diagnostics[0])[0].0, "x + y = 3");
}

#[test]
fn unbalanced_model_names_at_most_eight_leftovers_and_counts_the_rest() {
    let source = format!(
        "{GAIN}
model Many
  Gain g[10];
  Real x(start = 0, fixed = true);
equation
  der(x) = sum(g.y);
end Many;
"
    );
    let (error, _, diagnostics) = todae_failure(&source, "Many");
    assert!(error.contains("(balance = -10)"), "{error}");
    assert!(
        error.contains(
            "left without an equation: `g[1].u` is an input of `g[1]` with no connection and \
             no binding; `g[2].u` is an input of `g[2]`"
        ),
        "{error}"
    );
    assert!(error.ends_with("; and 2 more"), "{error}");
    assert_eq!(diagnostics[0].labels.len(), 8);
}
