//! Batched Robertson with diffsol's native batching (nalgebra, one lane per ODE) and
//! hand-written Jacobian, adjoint and parameter-derivative closures.
//!
//! Parameters per lane: wrt=params -> [log k1, log k2, log k3, eps] with y0 = (1-eps, eps, 0);
//! wrt=ic -> y0 (3 values), with k fixed.
// ponytail: eps rides along as a 4th parameter because `init` closures can't see the lane index,
// so forward sensitivities solve 4 equations per lane instead of 3; seed the state directly if this matters.

use std::rc::Rc;

use bench::{mat_f64, parse_args, run, vec_f64};
use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, Matrix, MatrixCommon, NalgebraContext, NalgebraLU,
    NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod, OdeSolverState,
    SensitivitiesOdeSolverMethod, StepVjp, Vector, VectorView, VectorViewMut,
};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type LS = NalgebraLU<f64>;

fn rates(p: &[f64], kc: [f64; 3]) -> [f64; 3] {
    if p.len() == 4 {
        [p[0].exp(), p[1].exp(), p[2].exp()]
    } else {
        kc
    }
}

fn rhs(x: &[f64], k: [f64; 3], f: &mut [f64]) {
    f[0] = -k[0] * x[0] + k[2] * x[1] * x[2];
    f[1] = k[0] * x[0] - k[1] * x[1] * x[1] - k[2] * x[1] * x[2];
    f[2] = k[1] * x[1] * x[1];
}

fn jac(x: &[f64], k: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [-k[0], k[2] * x[2], k[2] * x[1]],
        [k[0], -2.0 * k[1] * x[1] - k[2] * x[2], -k[2] * x[1]],
        [0.0, 2.0 * k[1] * x[1], 0.0],
    ]
}

/// columns are k_j * df/dk_j, i.e. df/dlog(k_j)
fn dfdlogk(x: &[f64], k: [f64; 3]) -> [[f64; 3]; 3] {
    let (yy, yz) = (x[1] * x[1], x[1] * x[2]);
    [
        [-k[0] * x[0], 0.0, k[2] * yz],
        [k[0] * x[0], -k[1] * yy, -k[2] * yz],
        [0.0, k[1] * yy, 0.0],
    ]
}

fn mul(a: &[[f64; 3]; 3], v: &[f64], y: &mut [f64], sign: f64) {
    for i in 0..3 {
        y[i] = sign * (a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2]);
    }
}

fn mul_t(a: &[[f64; 3]; 3], v: &[f64], y: &mut [f64], sign: f64) {
    for j in 0..3 {
        y[j] = sign * (a[0][j] * v[0] + a[1][j] * v[1] + a[2][j] * v[2]);
    }
}

fn init(p: &[f64], y: &mut [f64]) {
    if p.len() == 4 {
        y.copy_from_slice(&[1.0 - p[3], p[3], 0.0]);
    } else {
        y.copy_from_slice(&p[..3]);
    }
}

macro_rules! builder {
    ($ctx:expr, $p:expr, $a:expr) => {
        OdeBuilder::<M>::new()
            .context($ctx)
            .rtol($a.rtol)
            .atol([$a.atol])
            .sens_rtol($a.rtol)
            .sens_atol([$a.atol])
            .p($p.clone())
    };
}

struct Prob {
    n: usize,
    tobs: Vec<f64>,
    data: Vec<Vec<f64>>, // stacked (3n, nt)
    rtol: f64,
    atol: f64,
    delta: f64,
}

impl Prob {
    fn dl(&self, y: f64, d: f64) -> f64 {
        let ym = (y + self.delta).max(1e-30);
        2.0 * (ym.ln() - (d + self.delta).ln()) / ym
    }
    fn l(&self, y: f64, d: f64) -> f64 {
        let r = (y + self.delta).max(1e-30).ln() - (d + self.delta).ln();
        r * r
    }
    /// lane-major batched column -> stacked index helper
    fn d(&self, b: usize, s: usize, j: usize) -> f64 {
        self.data[s * self.n + b][j]
    }
    /// dense solution (columns are batched vectors) -> stacked row-major (3n, nt)
    fn stacked(&self, soln: &M) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; self.tobs.len()]; 3 * self.n];
        for j in 0..soln.ncols() {
            let col = soln.column(j).into_owned().clone_as_vec();
            for b in 0..self.n {
                for s in 0..3 {
                    out[s * self.n + b][j] = col[b * 3 + s];
                }
            }
        }
        out
    }
}

fn main() {
    let args = parse_args();
    let pj = &args.problem;
    let a = Prob {
        n: args.size,
        tobs: vec_f64(&pj["tobs"]),
        data: mat_f64(&pj["data"]),
        rtol: pj["rtol"].as_f64().unwrap(),
        atol: pj["atol"].as_f64().unwrap(),
        delta: pj["delta"].as_f64().unwrap(),
    };
    let n = a.n;
    let logk = vec_f64(&pj["logk"]);
    let y0 = vec_f64(&pj["y0"]);
    let kc = [logk[0].exp(), logk[1].exp(), logk[2].exp()];
    let params = args.wrt == "params";
    let p: Vec<f64> = (0..n)
        .flat_map(|b| {
            if params {
                vec![logk[0], logk[1], logk[2], y0[n + b]]
            } else {
                vec![y0[b], y0[n + b], y0[2 * n + b]]
            }
        })
        .collect();
    let ctx = NalgebraContext::with_nbatch(n);
    let a = &a;
    let grad_method = args.grad.clone();

    run(&args, || {
        let sens_problem = builder!(ctx, p, a)
            .rhs_sens_implicit(
                move |x, p, _t, y| rhs(x, rates(p, kc), y),
                move |x, p, _t, v, y| mul(&jac(x, rates(p, kc)), v, y, 1.0),
                move |x, p, _t, v, y| {
                    if p.len() == 4 {
                        mul(&dfdlogk(x, rates(p, kc)), v, y, 1.0)
                    } else {
                        y.fill(0.0)
                    }
                },
            )
            .init_sens(
                |p, _t, y| init(p, y),
                |p, _t, v, y| {
                    if p.len() == 4 {
                        y.copy_from_slice(&[-v[3], v[3], 0.0]);
                    } else {
                        y.copy_from_slice(&v[..3]);
                    }
                },
                3,
            )
            .build()
            .unwrap();
        // adjoint closures return minus J^T v, f_p^T v and (dy0/dp)^T v
        let adj_problem = builder!(ctx, p, a)
            .rhs_adjoint_implicit(
                move |x, p, _t, y| rhs(x, rates(p, kc), y),
                move |x, p, _t, v, y| mul(&jac(x, rates(p, kc)), v, y, 1.0),
                move |x, p, _t, v, y| mul_t(&jac(x, rates(p, kc)), v, y, -1.0),
                move |x, p, _t, v, y| {
                    if p.len() == 4 {
                        mul_t(&dfdlogk(x, rates(p, kc)), v, &mut y[..3], -1.0);
                        y[3] = 0.0;
                    } else {
                        y.fill(0.0)
                    }
                },
            )
            .init_adjoint(
                |p, _t, y| init(p, y),
                |p, _t, v, y| {
                    if p.len() == 4 {
                        y.copy_from_slice(&[0.0, 0.0, 0.0, v[0] - v[1]]);
                    } else {
                        y.iter_mut().zip(v).for_each(|(y, v)| *y = -v);
                    }
                },
                3,
            )
            .build()
            .unwrap();

        let (sens_problem, adj_problem) = (Rc::new(sens_problem), Rc::new(adj_problem));
        let solve_problem = sens_problem.clone();
        let solve = move || {
            let (soln, _) = solve_problem.bdf::<LS>().unwrap().solve_dense(&a.tobs).unwrap();
            a.stacked(&soln).concat()
        };

        // sum lanes of per-lane param gradients into the canonical layout
        let canon = move |lanes: Vec<Vec<f64>>| -> Vec<f64> {
            if params {
                (0..3).map(|j| lanes.iter().map(|l| l[j]).sum()).collect()
            } else {
                (0..3).flat_map(|s| lanes.iter().map(move |l| l[s])).collect()
            }
        };
        let lanes_of = move |v: &V| -> Vec<Vec<f64>> {
            (0..n).map(|b| v.get_batch(b).into_owned().clone_as_vec()).collect()
        };
        let dgdu_col = move |soln: &Vec<Vec<f64>>, j: usize| -> Vec<f64> {
            (0..n)
                .flat_map(|b| (0..3).map(move |s| a.dl(soln[s * n + b][j], a.d(b, s, j))))
                .collect()
        };

        let grad = move || -> Result<Vec<f64>, String> {
            let e = |e: diffsol::error::DiffsolError| e.to_string();
            match grad_method.as_str() {
                "fwd" => {
                    let mut solver = sens_problem.bdf_sens::<LS>().map_err(e)?;
                    let (soln, sens, _) = solver.solve_dense_sensitivities(&a.tobs).map_err(e)?;
                    let y = a.stacked(&soln);
                    let lanes: Vec<Vec<f64>> = (0..n)
                        .map(|b| {
                            sens.iter()
                                .map(|sj| {
                                    let sj = a.stacked(sj);
                                    (0..a.tobs.len())
                                        .map(|t| {
                                            (0..3)
                                                .map(|s| {
                                                    a.dl(y[s * n + b][t], a.d(b, s, t))
                                                        * sj[s * n + b][t]
                                                })
                                                .sum::<f64>()
                                        })
                                        .sum()
                                })
                                .collect()
                        })
                        .collect();
                    Ok(canon(lanes))
                }
                "adj" => {
                    let mut solver = adj_problem.bdf::<LS>().map_err(e)?;
                    let (cp, soln, _) =
                        solver.solve_dense_with_checkpointing(&a.tobs, None).map_err(e)?;
                    let y = a.stacked(&soln);
                    let mut dgdu = M::zeros(3, a.tobs.len(), ctx);
                    for j in 0..a.tobs.len() {
                        dgdu.column_mut(j).copy_from(&V::from_vec(dgdu_col(&y, j), ctx));
                    }
                    let adj = adj_problem
                        .bdf_solver_adjoint::<LS, _>(cp, Some(solver), Some(1))
                        .map_err(e)?;
                    let (state, _) =
                        adj.solve_adjoint_backwards_pass(&a.tobs, &[&dgdu]).map_err(e)?;
                    Ok(canon(lanes_of(&state.into_common().sg)))
                }
                "rev" => {
                    let mut solver = adj_problem.bdf::<LS>().map_err(e)?;
                    let g = |i: usize, _t: f64, u: &V, out: &mut V| {
                        out.for_each_batch_host([u], |o, [u], b| {
                            o[0] += (0..3).map(|s| a.l(u[s], a.d(b, s, i))).sum::<f64>()
                        });
                    };
                    let dg = |i: usize, _t: f64, u: &V, d: &mut V| {
                        d.for_each_batch_host([u], |d, [u], b| {
                            for s in 0..3 {
                                d[s] = a.dl(u[s], a.d(b, s, i));
                            }
                        });
                    };
                    let (_, dgdp) = solver.solve_dense_vjp(&a.tobs, 1, g, dg).map_err(e)?;
                    Ok(canon(lanes_of(&dgdp)))
                }
                _ => Err("unsupported grad".into()),
            }
        };
        (solve, grad)
    });
}
