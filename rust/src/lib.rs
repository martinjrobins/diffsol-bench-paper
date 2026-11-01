//! Timing harness shared by the benchmark binaries; mirrors `common.run` in Python.

use serde_json::{json, Value};
use std::time::Instant;

pub struct Args {
    pub size: usize,
    pub phase: String,
    pub grad: String,
    pub wrt: String,
    pub jac: String,
    pub problem: Value,
    pub out: String,
}

pub fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str, default: &str| -> String {
        argv.iter()
            .position(|a| a == name)
            .map(|i| argv[i + 1].clone())
            .unwrap_or_else(|| default.to_string())
    };
    let problem = std::fs::read_to_string(get("--problem", "")).expect("--problem");
    Args {
        size: get("--size", "1").parse().unwrap(),
        phase: get("--phase", "grad"),
        grad: get("--grad", "fwd"),
        wrt: get("--wrt", "params"),
        jac: get("--jac", "manual"),
        problem: serde_json::from_str(&problem).unwrap(),
        out: get("--out", ""),
    }
}

pub fn vec_f64(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

pub fn mat_f64(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(vec_f64).collect()
}

fn timed<R>(f: &mut impl FnMut() -> R) -> (f64, R) {
    let t0 = Instant::now();
    let r = f();
    (t0.elapsed().as_secs_f64(), r)
}

fn median<R>(f: &mut impl FnMut() -> R, t_first: f64) -> f64 {
    let nrep = ((1.0 / t_first.max(1e-9)) as usize).clamp(3, 10);
    let mut ts: Vec<f64> = (0..nrep).map(|_| timed(f).0).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[nrep / 2]
}

/// `make` builds the problem and returns (solve, grad) closures giving y at tobs and the
/// gradient, both in the canonical layout used by `common.py`.
pub fn run<S, G>(args: &Args, make: impl FnOnce() -> (S, G))
where
    S: FnMut() -> Vec<f64>,
    G: FnMut() -> Result<Vec<f64>, String>,
{
    let mut res = json!({});
    if args.phase != "import" {
        let t0 = Instant::now();
        let (mut solve, mut grad) = make();
        let (t_first, y) = timed(&mut solve);
        let t_build = t0.elapsed().as_secs_f64() - t_first;
        let (t_warm, _) = timed(&mut solve);
        let solve_s = median(&mut solve, t_warm);
        res["solve_s"] = json!(solve_s);
        res["setup_s"] = json!(t_build + t_first - solve_s);
        res["y_at_tobs"] = json!(y);
        if args.phase == "grad" {
            let (t_first, g) = timed(&mut grad);
            let g = g.unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let grad_s = median(&mut grad, t_first);
            res["grad_s"] = json!(grad_s);
            res["grad_setup_s"] = json!(t_first - grad_s);
            res["grad"] = json!(g);
        }
    }
    std::fs::write(&args.out, res.to_string()).unwrap();
}
