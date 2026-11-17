//! MLS §10.3.3: `fill(s, n1, ..., nk)` of an array `s` has shape
//! `{n1, ..., nk, size(s)}`, as `ports.C_outflow = fill(C_in_internal, nPorts)`
//! in Modelica.Fluid's sources needs.

use rumoca_ir_ast as ast;

const SOURCE_NAME: &str = "array-valued-fill.mo";
const SOURCE: &str = r#"
package FT
  record R
    Real C[2];
  end R;

  model Equation
    parameter Integer n = 3;
    Real c[2] = {1 + time, 2};
    Real y[n, 2];
  equation
    y = fill(c, n);
  end Equation;

  model RecordField
    parameter Integer n = 2;
    Real c[2] = {1 + time, 2};
    R r[n];
  equation
    r.C = fill(c, n);
  end RecordField;

  model TwoExtents
    Real c[2] = {1 + time, 2};
    Real y[2, 2, 2] = fill(c, 2, 2);
  end TwoExtents;

  model ColonBinding
    Real c[2] = {1 + time, 2};
    Real y[:, :] = fill(c, 3);
  end ColonBinding;
end FT;
"#;

fn flatten_to_dae(model: &str) -> rumoca_ir_flat::Model {
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
    rumoca_phase_dae::to_dae(&flat, tree.source_map.clone())
        .unwrap_or_else(|error| panic!("{model} lowers to checked DAE: {error:?}"));
    flat
}

#[test]
fn fill_of_an_array_keeps_the_array_axes() {
    for model in ["FT.Equation", "FT.RecordField", "FT.TwoExtents"] {
        flatten_to_dae(model);
    }
    let flat = flatten_to_dae("FT.ColonBinding");
    assert_eq!(flat.variables[&rumoca_core::VarName::new("y")].dims, [3, 2]);
}
