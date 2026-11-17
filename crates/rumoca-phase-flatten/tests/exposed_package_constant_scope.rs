//! MLS §5.3, §7.3: `Medium.X_default` read in `col.cooling` is the constant of
//! the package `cooling` selects (`Water`, one substance). Its binding names
//! `reference_X`, which resolves in that package, not in `col`, whose own
//! `Medium` (`Mix`, three substances) modifies `reference_X`.

use rumoca_core::Expression;
use rumoca_ir_ast as ast;

const SOURCE_NAME: &str = "exposed-package-constant-scope.mo";
const SOURCE: &str = r#"
package Q
  partial package PartialMedium
    constant String substanceNames[:] = {"single"};
    final constant Integer nX = size(substanceNames, 1);
    constant Real reference_X[nX] = fill(1/nX, nX);
    final constant Real X_default[nX] = reference_X;
  end PartialMedium;

  package Mix
    extends PartialMedium(substanceNames = {"A", "B", "C"}, reference_X = {0.2, 0.3, 0.5});
  end Mix;

  package Water
    extends PartialMedium(substanceNames = {"water"});
  end Water;

  model Volume
    replaceable package Medium = PartialMedium;
    parameter Real X_start[Medium.nX] = Medium.X_default;
    Real x[Medium.nX](start = X_start, each fixed = true);
  equation
    der(x) = -x;
  end Volume;

  model Column
    replaceable package Medium = Mix constrainedby PartialMedium;
    replaceable package Medium_CW = Water constrainedby PartialMedium;
    Volume mix(redeclare package Medium = Medium);
    Volume cooling(redeclare package Medium = Medium_CW);
  end Column;

  model Plant
    Column col;
  end Plant;
end Q;
"#;

#[test]
fn package_constant_binding_resolves_in_the_exposing_package() {
    let model = "Q.Plant";
    let stored = rumoca_phase_parse::parse_to_ast(SOURCE, SOURCE_NAME).expect("fixture parses");
    let mut tree = ast::ClassTree::from_parsed(stored);
    tree.source_map.add(SOURCE_NAME, SOURCE);
    let resolved =
        rumoca_phase_resolve::resolve(ast::ParsedTree::new(tree)).expect("fixture resolves");
    let ast::InstancedTree { tree, mut overlay } =
        rumoca_phase_instantiate::instantiate(resolved, model).expect("fixture instantiates");
    rumoca_phase_typecheck::typecheck_instanced(&tree, &mut overlay, model)
        .expect("fixture typechecks");
    let flat = rumoca_phase_flatten::flatten_ref(&tree, &overlay, model).expect("fixture flattens");

    let binding = flat.variables[&rumoca_core::VarName::new("col.cooling.X_start")]
        .binding
        .as_ref()
        .expect("X_start keeps its declaration binding");
    assert!(
        !matches!(binding, Expression::Array { elements, .. } if elements.len() == 3),
        "col.cooling.X_start takes Mix's composition: {binding:?}"
    );
    rumoca_phase_dae::to_dae(&flat, tree.source_map.clone())
        .expect("each volume's composition matches its medium");
}
