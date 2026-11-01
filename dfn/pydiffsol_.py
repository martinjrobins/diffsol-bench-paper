import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np

import common


def make(prob, args):
    tobs, data, p = np.array(prob["tobs"]), np.array(prob["data"]), np.array(prob["p"])
    ode = common.dfn_ode()
    ode.rtol, ode.atol = prob["rtol"], prob["atol"]
    ode.sens_rtol, ode.sens_atol = prob["rtol"], prob["atol"]

    def solve():
        return ode.solve_dense(p, tobs).ys

    def grad():
        if args.grad == "fwd":
            sol = ode.solve_fwd_sens(p, tobs)
            dl = common.brusselator_dloss(sol.ys, data)
            return np.array([np.sum(dl * s) for s in sol.sens])
        if args.grad == "adj":
            sol, ckpt = ode.solve_adjoint_fwd(p, tobs)
            return ode.solve_adjoint_bkwd(sol, ckpt, common.brusselator_dloss(sol.ys, data)).ravel()
        raise NotImplementedError

    return solve, grad


if __name__ == "__main__":
    common.run(make)
