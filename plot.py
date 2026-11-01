"""uv run plot.py robertson -> results/robertson_<wrt>_t<nthreads>.png and results/robertson_speedup.png"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS

COLS = [("setup_s", "mem_setup", "setup (incl. JIT)"), ("solve_s", "mem_solve", "solve"), ("grad_s", "mem_grad", "gradient")]
STYLE = {"fwd": "-", "adj": "--", "rev": ":"}
MARKER = {"auto": "o", "manual": "s"}


def load(bench):
    with open(RESULTS / f"{bench}.csv") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def series(rows, key, mem_base=None):
    """(solver, jac, grad) -> sorted [(size, value)]"""
    out = defaultdict(list)
    for r in rows:
        if not r[key] or (mem_base and not r[mem_base]):
            continue
        v = float(r[key])
        if mem_base:
            v = (v - float(r[mem_base])) / 2**20  # MB above the previous phase
        out[r["solver"], r["jac"], r["grad"]].append((int(r["size"]), v))
    return {k: sorted(v) for k, v in out.items()}


def panel(ax, data, solvers, setup_like, mem=False):
    for (solver, jac, grad), pts in data.items():
        # setup/solve don't depend on grad method: draw them once per (solver, jac)
        if setup_like and grad != min(g for s, j, g in data if (s, j) == (solver, jac)):
            continue
        xs, ys = zip(*pts)
        label = f"{solver} {jac}" if setup_like else f"{solver} {jac} {grad}"
        ax.plot(xs, ys, color=f"C{solvers.index(solver)}", ls="-" if setup_like else STYLE[grad],
                marker=MARKER[jac], ms=3, label=label)
    ax.set_xscale("log")
    # memory deltas can be <= 0
    ax.set_yscale("symlog", linthresh=1) if mem else ax.set_yscale("log")


def main(bench):
    rows = load(bench)
    solvers = sorted({r["solver"] for r in rows})
    prev = {"mem_setup": "mem_import", "mem_solve": "mem_setup", "mem_grad": "mem_solve"}
    for wrt in ["params", "ic"]:
        for nt in sorted({r["nthreads"] for r in rows}):
            sub = [r for r in rows if r["wrt"] == wrt and r["nthreads"] == nt]
            if not sub:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for c, (tkey, mkey, title) in enumerate(COLS):
                setup_like = c < 2
                panel(axes[0, c], series(sub, tkey), solvers, setup_like)
                panel(axes[1, c], series(sub, mkey, prev[mkey]), solvers, setup_like, mem=True)
                axes[0, c].set_title(title)
                axes[1, c].set_xlabel("size")
                axes[0, c].legend(fontsize=6)
            axes[0, 0].set_ylabel("time (s)")
            axes[1, 0].set_ylabel("peak RSS increase (MB)")
            fig.suptitle(f"{bench}: gradient w.r.t. {wrt}, nthreads={nt}")
            fig.savefig(RESULTS / f"{bench}_{wrt}_t{nt}.png", dpi=120)
            plt.close(fig)
    speedup(bench, rows)


def speedup(bench, rows):
    t = {(r["solver"], r["jac"], r["wrt"], r["grad"], r["size"], r["nthreads"]): float(r["grad_s"]) for r in rows if r["grad_s"]}
    pts = defaultdict(list)
    for (s, j, w, g, n, nt), v in t.items():
        if nt == "15" and (s, j, w, g, n, "1") in t:
            pts[s, j, w, g].append((int(n), t[s, j, w, g, n, "1"] / v))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for (s, j, w, g), p in sorted(pts.items()):
        ax.plot(*zip(*sorted(p)), ls=STYLE[g], marker=MARKER[j], ms=3, label=f"{s} {j} {w} {g}")
    ax.set_xscale("log")
    ax.set_xlabel("size")
    ax.set_ylabel("gradient speed-up, 15 vs 1 threads")
    ax.legend(fontsize=5)
    fig.savefig(RESULTS / f"{bench}_speedup.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1])
