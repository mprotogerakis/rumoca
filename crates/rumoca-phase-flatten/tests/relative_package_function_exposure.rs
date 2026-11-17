//! MLS §5.3: `Medium.density` names `density` through whichever `Medium` the
//! calling class sees. `Machine` and the `PartialTwoPort` that its curve
//! extends each declare their own `Medium`, so their calls spell one name for
//! two exposures, and only `Machine.Medium` exposes `Gas.ThermodynamicState`.
//! Either call may be met first, so both orders are checked.

use rumoca_ir_ast as ast;

const SOURCE_NAME: &str = "relative-package-function-exposure.mo";
const SOURCE: &str = r#"
package P
  partial package PartialMedium
    replaceable record ThermodynamicState
    end ThermodynamicState;
    replaceable partial function density
      input ThermodynamicState state;
      output Real d;
    end density;
  end PartialMedium;

  package Gas
    extends PartialMedium;
    redeclare record extends ThermodynamicState
      Real p;
      Real T;
    end ThermodynamicState;
    redeclare function extends density
    algorithm
      d := state.p/(287*state.T);
    end density;
  end Gas;

  partial model PartialTwoPort
    replaceable package Medium = PartialMedium;
    Medium.ThermodynamicState state(p = 1e5 + time, T = 300);
  end PartialTwoPort;

  model Curve
    extends PartialTwoPort(redeclare replaceable package Medium = Gas);
    Real d;
  equation
    d = Medium.density(state);
  end Curve;

  model BoundCurve
    extends PartialTwoPort(redeclare replaceable package Medium = Gas);
    Real d = Medium.density(state);
  end BoundCurve;

  model Machine
    replaceable package Medium = Gas constrainedby PartialMedium;
    Real d = Medium.density(curve.state);
    Curve curve(redeclare package Medium = Medium);
  end Machine;

  model EquationMachine
    replaceable package Medium = Gas constrainedby PartialMedium;
    Real d;
    BoundCurve curve(redeclare package Medium = Medium);
  equation
    d = Medium.density(curve.state);
  end EquationMachine;
end P;
"#;

fn assert_each_exposure_keeps_its_state(model: &str) {
    let stored = rumoca_phase_parse::parse_to_ast(SOURCE, SOURCE_NAME).expect("fixture parses");
    let mut tree = ast::ClassTree::from_parsed(stored);
    tree.source_map.add(SOURCE_NAME, SOURCE);
    let resolved =
        rumoca_phase_resolve::resolve(ast::ParsedTree::new(tree)).expect("fixture resolves");
    let ast::InstancedTree { tree, mut overlay } =
        rumoca_phase_instantiate::instantiate(resolved, model).expect("fixture instantiates");
    rumoca_phase_typecheck::typecheck_instanced(&tree, &mut overlay, model)
        .expect("fixture typechecks");
    let source_map = tree.source_map.clone();
    let flat = rumoca_phase_flatten::flatten_ref(&tree, &overlay, model).expect("fixture flattens");

    for function in flat.functions.values() {
        if function.name.as_str().ends_with("density") {
            let inputs: Vec<_> = function
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect();
            assert_eq!(
                inputs,
                ["state_p", "state_T"],
                "{} takes Gas's state",
                function.name
            );
        }
    }
    rumoca_phase_dae::to_dae(&flat, source_map).expect("both exposures lower to checked DAE");
}

#[test]
fn package_function_called_through_two_owners_keeps_each_exposure() {
    assert_each_exposure_keeps_its_state("P.Machine");
    assert_each_exposure_keeps_its_state("P.EquationMachine");
}
