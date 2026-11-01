import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import casadi as ca
import numpy as np

import common


def make(prob, args):
    N, tobs, data = prob["n"], prob["tobs"], np.array(prob["data"])
    N2, dx = N * N, 1 / (N - 1)
    params = args.wrt == "params"
    w = ca.SX.sym("w", 2 * N2)
    u, v = w[:N2], w[N2:]
    p = ca.SX.sym("p", 3 if params else 0)
    A, B, alpha = (p[0], p[1], p[2]) if params else prob["p"]
    t = ca.SX.sym("t")
    rows, cols, vals = common.bruss_laplacian(N)
    lap = ca.DM.triplet(rows, cols, vals, N2, N2)
    s = alpha / dx**2
    forcing = 5 * ca.if_else(t >= 1.1, 1, 0) * common.bruss_mask(N)
    rhs = ca.vertcat(
        s * ca.mtimes(lap, u) + B + u**2 * v - (A + 1) * u + forcing,
        s * ca.mtimes(lap, v) + A * u - u**2 * v,
    )
    opts = dict(abstol=prob["atol"], reltol=prob["rtol"], max_num_steps=100_000, linear_solver="csparse")
    F = ca.integrator("F", "cvodes", {"x": w, "p": p, "t": t, "ode": rhs}, 0.0, tobs, opts)

    y0, p0 = np.array(prob["y0"]), np.array(prob["p"])
    P = ca.MX.sym("P", 3 if params else 2 * N2)
    xf = F(x0=y0, p=P)["xf"] if params else F(x0=P, p=[])["xf"]
    # ad_weight: 0 forces forward sensitivities, 1 forces the adjoint
    L = ca.Function("L", [P], [ca.sumsqr(xf - data)], {"ad_weight": 0 if args.grad == "fwd" else 1})
    J = L.jacobian()
    x0 = p0 if params else y0
    if args.grad not in ("fwd", "adj"):
        raise NotImplementedError

    def solve():
        return np.array(F(x0=y0, p=p0 if params else [])["xf"])

    def grad():
        return np.array(J(x0, 0)).ravel()

    return solve, grad


if __name__ == "__main__":
    common.run(make)
