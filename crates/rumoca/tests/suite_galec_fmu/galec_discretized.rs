//! A continuous-time model discretized for the GALEC targets (Compiler::compile_str_discretized).
use rumoca::{Compiler, render_target_files};

const MODEL: &str = r#"
model Plant "a lag, a PI, an event, a start, the time"
  input Real w;
  input Real y;
  output Real u;
  output Boolean alarm;
  output Integer trips(start = 0);
  parameter Real T = 2;
  parameter Real k = 3;
  parameter Real x0 = 1;
  Real lag;
  Real integral(start = 0);
  Real seen;
  discrete Real t0(start = 0);
  output Boolean h "a hysteresis, as the MSL's";
  parameter Boolean h0 = true;
initial equation
  lag = x0;
  pre(h) = h0;
equation
  der(lag) = (y - lag)/T;
  der(integral) = w - lag;
  u = k*(w - lag) + integral;
  alarm = lag > 5 or time > 7;
  when alarm then
    trips = pre(trips) + 1;
  end when;
  when {initial(), edge(alarm)} then
    t0 = time;
  end when;
  seen = if initial() then 0 else 1;
  h = not pre(h) and y > 6 or pre(h) and y >= 4;
end Plant;
"#;

#[test]
fn a_continuous_model_discretized_goes_through_the_galec_targets() {
    let c = Compiler::new().model("Plant").compile_str_discretized(MODEL, "plant.mo", 0.1).unwrap_or_else(|e| panic!("{e}"));
    for target in ["galec", "embedded-c-galec"] {
        let files = render_target_files(&c, "Plant", target, None).unwrap_or_else(|e| panic!("{target}: {e:#}"));
        assert!(!files.is_empty(), "{target}");
        if target == "galec" {
            let alg = &files.iter().find(|f| f.path.ends_with(".alg")).unwrap().content;
            // (each state one Euler step from its value at the tick's start; its start of parameters
            // from its initial equation; the events as rises since the last tick; time 0 at the first)
            for want in [
                "self.lag := self.'previous(lag)' + (0.1 * ((self.y - self.'previous(lag)') / self.T));",
                "self.integral := self.'previous(integral)' + (0.1 * (self.w - self.'previous(lag)'));",
                "self.lag := 1.0;",
                "self.'time.tick' := -0.1;",
                "self.'time.tick' := self.'previous(time.tick)' + 0.1;",
                ":= self.'previous(trips)' + 1;",
                "self.'pre.when' := self.alarm;",
                "self.'pre.of.h' := true;",
                "self.'pre.of.h' := self.h;",
            ] {
                assert!(alg.contains(want), "{want}\n{alg}");
            }
        }
    }
}
