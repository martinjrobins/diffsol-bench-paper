"""Benchmark driver.

uv run run.py check robertson --size 10      # consistency of all solvers vs reference
uv run run.py sweep robertson [--quick]      # timings + memory -> results/robertson.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

import common

ROOT = Path(__file__).parent
PY = [sys.executable]
JL = ["julia", "--project=" + str(ROOT / "julia")]

# name -> (command, jac variants, grad methods)
SOLVERS = {
    "robertson": {
        "diffsol": ([str(ROOT / "rust/target/release/robertson")], ["manual"], ["fwd", "adj", "rev"]),
        "pydiffsol": (PY + ["robertson/pydiffsol_.py"], ["auto"], ["fwd", "adj"]),
        "diffrax": (PY + ["robertson/diffrax_.py"], ["auto"], ["fwd", "adj", "rev"]),
        "casadi": (PY + ["robertson/casadi_.py"], ["auto"], ["fwd", "adj"]),
        "julia": (JL + ["julia/robertson.jl"], ["auto", "manual"], ["fwd", "adj"]),  # rev: bugs/README.md
    },
    "brusselator": {
        "pydiffsol": (PY + ["brusselator/pydiffsol_.py"], ["auto"], ["fwd", "adj"]),
        "casadi": (PY + ["brusselator/casadi_.py"], ["auto"], ["fwd", "adj"]),
        # fwd / all Sundials gradients fail in this Julia stack: bugs/README.md
        "julia": (JL + ["julia/brusselator.jl", "--alg", "fbdf"], ["auto"], ["adj"]),
        "julia-sundials": (JL + ["julia/brusselator.jl", "--alg", "cvode"], ["auto"], []),
    },
    "dfn": {
        "pydiffsol": (PY + ["dfn/pydiffsol_.py"], ["auto"], ["fwd", "adj"]),
        "julia": (JL + ["julia/dfn.jl"], ["auto"], []),
    },
}
SIZES = {"robertson": [1, 3, 10, 30, 100, 300, 1000], "brusselator": [8, 16, 32, 64, 128], "dfn": [1]}
# (bench, grad, wrt) -> largest size run; forward sensitivities w.r.t. ICs scale with the state size
CAPS = {("robertson", "fwd", "ic"): 100, ("brusselator", "fwd", "ic"): 16}
WRT = {"robertson": ["params", "ic"], "brusselator": ["params", "ic"], "dfn": ["params"]}
NTHREADS = [1, 15]
TIMEOUT = 1800
FIELDS = [
    "bench", "solver", "jac", "size", "nthreads", "wrt", "grad", "status",
    "setup_s", "solve_s", "grad_setup_s", "grad_s",
    "mem_import", "mem_setup", "mem_solve", "mem_grad",
]


def run_one(cmd, size, nthreads, phase, grad, wrt, jac, problem):
    """Run one solver process; returns (result dict or None, peak RSS bytes, error message)."""
    if cmd[0] == "julia":
        cmd = cmd[:1] + ["-t", str(nthreads)] + cmd[1:]
    cpus = "0" if nthreads == 1 else f"0-{nthreads - 1}"
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        args = ["taskset", "-c", cpus, *cmd, "--size", str(size), "--phase", phase, "--grad", grad,
                "--wrt", wrt, "--jac", jac, "--problem", str(problem), "--out", out.name]
        n = str(nthreads)
        env = os.environ | dict(OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n,
                                RAYON_NUM_THREADS=n, JULIA_NUM_THREADS=n, JAX_PLATFORMS="cpu")
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        timer = threading.Timer(TIMEOUT, p.kill)
        timer.start()
        stderr = p.stderr.read().decode()
        _, status, rusage = os.wait4(p.pid, 0)
        timer.cancel()
        rss = rusage.ru_maxrss * 1024
        if status != 0:
            return None, rss, _error_line(stderr) or f"exit {status}"
        return json.loads(Path(out.name).read_text()), rss, ""


def _error_line(stderr):
    """Most informative line: Julia's 'ERROR:' line, else the last Python 'XxxError: ...' line."""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    julia = [l for l in lines if l.startswith("ERROR:")]
    py = [l for l in lines if "Error: " in l and not l.startswith("File ")]
    return (julia or py[-1:] or lines[-1:] or [""])[0][:300]


def configs(bench, sizes, nthreads):
    for solver, (cmd, jacs, grads) in SOLVERS[bench].items():
        for size in sizes:
            for nt in nthreads:
                for wrt in WRT[bench]:
                    for jac in jacs:
                        g = [g for g in grads if size <= CAPS.get((bench, g, wrt), size)]
                        yield solver, cmd, size, nt, wrt, jac, g


def sweep(bench, quick):
    sizes = SIZES[bench][:2] if quick else SIZES[bench]
    path = ROOT / "results" / f"{bench}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for solver, cmd, size, nt, wrt, jac, grads in configs(bench, sizes, [1] if quick else NTHREADS):
            problem = common.write_problem(bench, size)
            base = dict(bench=bench, solver=solver, jac=jac, size=size, nthreads=nt, wrt=wrt)
            mem = {}
            for phase in ["import", "setup", "solve"]:
                _, mem[phase], _ = run_one(cmd, size, nt, phase, "fwd", wrt, jac, problem)
            for grad in grads or [None]:  # no gradients: one solve-only row
                phase = "grad" if grad else "solve"
                res, mem_grad, err = run_one(cmd, size, nt, phase, grad or "fwd", wrt, jac, problem)
                row = base | dict(grad=grad or "none", status=err or "ok", mem_grad=mem_grad if grad else "",
                                  **{f"mem_{k}": v for k, v in mem.items()})
                row |= {k: v for k, v in (res or {}).items() if k.endswith("_s")}
                print(row["solver"], jac, size, nt, wrt, grad, row["status"], flush=True)
                w.writerow(row)
                f.flush()


def check(bench, size):
    problem = common.write_problem(bench, size)
    prob = json.loads(problem.read_text())
    ok = True
    refs = {wrt: common.REFERENCES[bench](prob, wrt) for wrt in WRT[bench]}
    for solver, cmd, _, _, wrt, jac, grads in configs(bench, [size], [1]):
        y_ref, g_ref = refs[wrt]
        for grad in grads or [None]:
            res, _, err = run_one(cmd, size, 1, "grad" if grad else "solve", grad or "fwd", wrt, jac, problem)
            if res is None:
                print(f"FAIL {solver} {jac} {wrt} {grad}: {err}")
                ok = False
                continue
            ey = _relerr(res["y_at_tobs"], y_ref)
            eg = _relerr(res["grad"], g_ref) if grad else 0.0
            good = ey < 1e-3 and eg < 1e-2
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {solver} {jac} {wrt} {grad}: y err {ey:.1e}, grad err {eg:.1e}")
    sys.exit(0 if ok else 1)


def _relerr(a, ref):
    a, ref = np.ravel(a), np.ravel(ref)
    return np.inf if a.shape != ref.shape else float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "sweep"])
    ap.add_argument("bench", choices=list(SOLVERS))
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--solvers", nargs="*", help="restrict to these solvers")
    a = ap.parse_args()
    if a.solvers:
        SOLVERS[a.bench] = {k: v for k, v in SOLVERS[a.bench].items() if k in a.solvers}
    check(a.bench, a.size) if a.cmd == "check" else sweep(a.bench, a.quick)
