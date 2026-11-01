"""Problem definitions, loss functions and the timing harness shared by all Python solver scripts."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

RTOL, ATOL = 1e-6, 1e-8
REF_RTOL, REF_ATOL = 1e-10, 1e-12
RESULTS = Path(__file__).parent / "results"
TINY = 1e-30
# log-misfit floor: values below ~atol are solver noise, so they shouldn't dominate the loss
DELTA = 10 * ATOL

# --- Robertson -------------------------------------------------------------
# state layout everywhere: stacked [x_0..x_{n-1}, y_0.., z_0..]

LOGK_TRUE = np.log([0.04, 3e7, 1e4])


def robertson_diffsl(n, wrt):
    """Stacked batched Robertson in DiffSL; inputs are log k (wrt=params) or y0 (wrt=ic)."""
    if wrt == "params":
        inputs = "in_i { lk1 = 0, lk2 = 0, lk3 = 0 }"
        ks = "k1 { exp(lk1) } k2 { exp(lk2) } k3 { exp(lk3) }"
        y0 = robertson_y0(n)
        u = "y0_i { " + ", ".join(repr(float(v)) for v in y0) + " }\n"
    else:
        inputs = f"in_i {{ (0:{3 * n}): y0 = 1 }}"
        lk = robertson_logk()
        ks = f"k1 {{ {float(np.exp(lk[0]))!r} }} k2 {{ {float(np.exp(lk[1]))!r} }} k3 {{ {float(np.exp(lk[2]))!r} }}"
        u = ""
    u += f"u_i {{ (0:{n}): x = y0_i[0:{n}], ({n}:{2 * n}): y = y0_i[{n}:{2 * n}], ({2 * n}:{3 * n}): z = y0_i[{2 * n}:{3 * n}] }}"
    return f"""
{inputs}
{ks}
{u}
F_i {{
    -k1 * x_i + k3 * y_i * z_i,
    k1 * x_i - k2 * y_i * y_i - k3 * y_i * z_i,
    k2 * y_i * y_i,
}}
out_i {{ x_i, y_i, z_i }}
"""


def robertson_y0(n):
    eps = 1e-3 * np.arange(n) / n
    return np.concatenate([1 - eps, eps, np.zeros(n)])


def robertson_logk():
    return LOGK_TRUE + 0.1


def robertson_problem(n):
    import pydiffsol as ds

    tobs = np.logspace(-2, 5, 40)
    ode = ds.Ode(robertson_diffsl(n, "params"), matrix_type=_matrix(n), linear_solver=ds.lu)
    ode.rtol, ode.atol = REF_RTOL, REF_ATOL
    data = ode.solve_dense(LOGK_TRUE, tobs).ys
    return dict(
        n=n,
        tobs=tobs.tolist(),
        logk=robertson_logk().tolist(),
        y0=robertson_y0(n).tolist(),
        data=data.tolist(),
        rtol=RTOL,
        atol=ATOL,
        delta=DELTA,
    )


def robertson_loss(y, data):
    r = np.log(np.maximum(y + DELTA, TINY)) - np.log(data + DELTA)
    return float(np.sum(r**2))


def robertson_dloss(y, data):
    """dL/dy, same shape as y."""
    ym = np.maximum(y + DELTA, TINY)
    return 2 * (np.log(ym) - np.log(data + DELTA)) / ym


def _matrix(n):
    import pydiffsol as ds

    return ds.nalgebra_dense if n < 20 else ds.faer_sparse


def robertson_reference(prob, wrt):
    """Tight-tolerance solution and gradient (pydiffsol forward sensitivities)."""
    import pydiffsol as ds

    n, tobs, data = prob["n"], np.array(prob["tobs"]), np.array(prob["data"])
    p = np.array(prob["logk"] if wrt == "params" else prob["y0"])
    ode = ds.Ode(robertson_diffsl(n, wrt), matrix_type=_matrix(n), linear_solver=ds.lu)
    ode.rtol, ode.atol = REF_RTOL, REF_ATOL
    ode.sens_rtol, ode.sens_atol = REF_RTOL, REF_ATOL
    sol = ode.solve_fwd_sens(p, tobs)
    dl = robertson_dloss(sol.ys, data)
    return sol.ys, np.array([np.sum(dl * s) for s in sol.sens])


# --- Brusselator (SciML advanced ODE tutorial) -------------------------------
# state layout: [U(:); V(:)], column-major over (i, j) with x = xyd[i], y = xyd[j]

BRUSS_P0 = np.array([3.4, 1.0, 10.0])  # (A, B, alpha), tutorial naming


def bruss_grid(N):
    xyd = np.linspace(0, 1, N)
    x, y = np.meshgrid(xyd, xyd, indexing="ij")
    return xyd, x.ravel(order="F"), y.ravel(order="F")


def bruss_laplacian(N):
    """Unscaled periodic 5-point Laplacian on the N x N grid, as (rows, cols, vals)."""
    idx = lambda i, j: (i % N) + (j % N) * N
    rows, cols, vals = [], [], []
    for j in range(N):
        for i in range(N):
            r = idx(i, j)
            for c, v in [(r, -4.0), (idx(i + 1, j), 1.0), (idx(i - 1, j), 1.0), (idx(i, j + 1), 1.0), (idx(i, j - 1), 1.0)]:
                rows.append(r), cols.append(c), vals.append(v)
    return rows, cols, vals


def bruss_mask(N):
    _, x, y = bruss_grid(N)
    return (((x - 0.3) ** 2 + (y - 0.6) ** 2) <= 0.1**2).astype(float)


def bruss_y0(N):
    _, x, y = bruss_grid(N)
    return np.concatenate([22 * (y * (1 - y)) ** 1.5, 27 * (x * (1 - x)) ** 1.5])


def bruss_p():
    return BRUSS_P0 * 1.05


def _vec(v):
    return ", ".join(repr(float(x)) for x in v)


def brusselator_diffsl(N, wrt):
    N2 = N * N
    dx = 1 / (N - 1)
    lap = ", ".join(f"({r}, {c}): {v!r}" for r, c, v in zip(*bruss_laplacian(N)))
    if wrt == "params":
        inputs = "in_i { A = 1, B = 1, alpha = 1 }"
        consts = f"y0_i {{ {_vec(bruss_y0(N))} }}"
    else:
        A, B, alpha = map(float, bruss_p())
        inputs = f"in_i {{ (0:{2 * N2}): y0 = 1 }}"
        consts = f"A {{ {A!r} }} B {{ {B!r} }} alpha {{ {alpha!r} }}"
    return f"""
{inputs}
{consts}
s {{ alpha / {dx * dx!r} }}
lap_ij {{ {lap} }}
m_i {{ {_vec(bruss_mask(N))} }}
u_i {{ (0:{N2}): uu = y0_i[0:{N2}], ({N2}:{2 * N2}): vv = y0_i[{N2}:{2 * N2}] }}
lu_i {{ lap_ij * uu_j }}
lv_i {{ lap_ij * vv_j }}
F_i {{
    s * lu_i + B + uu_i * uu_i * vv_i - (A + 1) * uu_i + 5 * heaviside(t - 1.1) * m_i,
    s * lv_i + A * uu_i - uu_i * uu_i * vv_i,
}}
out_i {{ uu_i, vv_i }}
"""


def brusselator_problem(N):
    import pydiffsol as ds

    tobs = np.linspace(1.0, 11.5, 10)
    ode = ds.Ode(brusselator_diffsl(N, "params"), matrix_type=ds.faer_sparse, linear_solver=ds.lu)
    ode.rtol, ode.atol = REF_RTOL, REF_ATOL
    data = ode.solve_dense(BRUSS_P0, tobs).ys
    return dict(n=N, tobs=tobs.tolist(), p=bruss_p().tolist(), y0=bruss_y0(N).tolist(),
                data=data.tolist(), rtol=RTOL, atol=ATOL)


def brusselator_dloss(y, data):
    return 2 * (y - data)


def brusselator_reference(prob, wrt):
    import pydiffsol as ds

    N, tobs, data = prob["n"], np.array(prob["tobs"]), np.array(prob["data"])
    p = np.array(prob["p"] if wrt == "params" else prob["y0"])
    ode = ds.Ode(brusselator_diffsl(N, wrt), matrix_type=ds.faer_sparse, linear_solver=ds.lu)
    ode.rtol, ode.atol = REF_RTOL, REF_ATOL
    ode.sens_rtol, ode.sens_atol = REF_RTOL, REF_ATOL
    sol = ode.solve_fwd_sens(p, tobs)
    dl = brusselator_dloss(sol.ys, data)
    return sol.ys, np.array([np.sum(dl * s) for s in sol.sens])


# --- DFN (dfn/export.py writes dfn/dfn.ds and dfn/dfn.json) ----------------

DFN_DIR = Path(__file__).parent / "dfn"


def dfn_ode():
    import pydiffsol as ds

    ode = ds.Ode((DFN_DIR / "dfn.ds").read_text(), matrix_type=ds.faer_sparse, linear_solver=ds.lu, ode_solver=ds.bdf)
    ode.ic_options.armijo_constant = 1e-1
    return ode


def dfn_problem(size):
    d = json.loads((DFN_DIR / "dfn.json").read_text())
    return dict(n=size, tobs=d["tobs"], p=d["p"], data=d["data"], rtol=RTOL, atol=ATOL)


def dfn_reference(prob, wrt):
    ode = dfn_ode()
    ode.rtol, ode.atol = REF_RTOL, REF_ATOL
    ode.sens_rtol, ode.sens_atol = REF_RTOL, REF_ATOL
    sol = ode.solve_fwd_sens(np.array(prob["p"]), np.array(prob["tobs"]))
    dl = brusselator_dloss(sol.ys, np.array(prob["data"]))
    return sol.ys, np.array([np.sum(dl * s) for s in sol.sens])


PROBLEMS = {"robertson": robertson_problem, "brusselator": brusselator_problem, "dfn": dfn_problem}
REFERENCES = {"robertson": robertson_reference, "brusselator": brusselator_reference, "dfn": dfn_reference}


def problem_path(bench, size):
    return RESULTS / f"problem_{bench}_{size}.json"


def write_problem(bench, size):
    path = problem_path(bench, size)
    if not path.exists():
        RESULTS.mkdir(exist_ok=True)
        path.write_text(json.dumps(PROBLEMS[bench](size)))
    return path


# --- harness ---------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--phase", choices=["import", "setup", "solve", "grad"], default="grad")
    ap.add_argument("--grad", choices=["fwd", "adj", "rev"], default="fwd")
    ap.add_argument("--wrt", choices=["params", "ic"], default="params")
    ap.add_argument("--jac", choices=["auto", "manual"], default="auto")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alg", default="")
    return ap.parse_args()


def _timed(f):
    t0 = time.perf_counter()
    r = f()
    return time.perf_counter() - t0, r


def _median(f, t_first):
    nrep = min(10, max(3, int(1.0 / max(t_first, 1e-9))))
    return statistics.median(_timed(f)[0] for _ in range(nrep))


def run(make):
    """make(problem, args) -> (solve, grad); solve() -> y at tobs, grad() -> gradient (both canonical layout)."""
    args = parse_args()
    prob = json.loads(Path(args.problem).read_text())
    res = {}
    if args.phase != "import":
        t_build, (solve, grad) = _timed(lambda: make(prob, args))
        t_first, y = _timed(solve)
        t_warm, _ = _timed(solve)
        res["solve_s"] = _median(solve, t_warm)
        res["setup_s"] = t_build + t_first - res["solve_s"]
        res["y_at_tobs"] = np.asarray(y).ravel().tolist()
        if args.phase == "grad":
            t_first, g = _timed(grad)
            res["grad_s"] = _median(grad, t_first)
            res["grad_setup_s"] = t_first - res["grad_s"]
            res["grad"] = np.asarray(g).ravel().tolist()
    Path(args.out).write_text(json.dumps(res))
