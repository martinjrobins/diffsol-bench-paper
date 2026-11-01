import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import casadi as ca
import numpy as np

import common


def make(prob, args):
    n, tobs, data = prob["n"], prob["tobs"], np.array(prob["data"])
    delta = prob["delta"]
    params = args.wrt == "params"
    u = ca.SX.sym("u", 3 * n)
    x, y, z = u[:n], u[n : 2 * n], u[2 * n :]
    p = ca.SX.sym("p", 3 if params else 0)
    k = ca.exp(p) if params else np.exp(prob["logk"])
    rhs = ca.vertcat(
        -k[0] * x + k[2] * y * z,
        k[0] * x - k[1] * y**2 - k[2] * y * z,
        k[1] * y**2,
    )
    opts = dict(abstol=prob["atol"], reltol=prob["rtol"], max_num_steps=100_000, linear_solver="csparse")
    F = ca.integrator("F", "cvodes", {"x": u, "p": p, "ode": rhs}, 0.0, tobs, opts)

    y0, logk = np.array(prob["y0"]), np.array(prob["logk"])
    P = ca.MX.sym("P", 3 if params else 3 * n)
    xf = F(x0=y0, p=P)["xf"] if params else F(x0=P, p=[])["xf"]
    r = ca.log(ca.fmax(xf + delta, common.TINY)) - np.log(data + delta)
    # ad_weight: 0 forces forward sensitivities, 1 forces the adjoint
    L = ca.Function("L", [P], [ca.sumsqr(r)], {"ad_weight": 0 if args.grad == "fwd" else 1})
    J = L.jacobian()
    x0 = logk if params else y0
    if args.grad not in ("fwd", "adj"):
        raise NotImplementedError

    def solve():
        return np.array(F(x0=y0, p=logk if params else [])["xf"])

    def grad():
        return np.array(J(x0, 0)).ravel()

    return solve, grad


if __name__ == "__main__":
    common.run(make)
