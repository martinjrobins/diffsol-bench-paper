import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import jax

jax.config.update("jax_enable_x64", True)
import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

import common


def rober(t, y, k):
    x, yy, z = y
    return jnp.stack([
        -k[0] * x + k[2] * yy * z,
        k[0] * x - k[1] * yy**2 - k[2] * yy * z,
        k[1] * yy**2,
    ])


def make(prob, args):
    n, tobs = prob["n"], jnp.array(prob["tobs"])
    rtol, atol, delta = prob["rtol"], prob["atol"], prob["delta"]
    # canonical stacked (3n, nt) <-> diffrax (n, nt, 3)
    data = jnp.array(prob["data"]).reshape(3, n, -1).transpose(1, 2, 0)
    y0 = jnp.array(prob["y0"]).reshape(3, n).T
    logk = jnp.array(prob["logk"])
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint = {
        "rev": diffrax.RecursiveCheckpointAdjoint(),
        "adj": diffrax.BacksolveAdjoint(solver=diffrax.Kvaerno5(), stepsize_controller=controller),
        "fwd": diffrax.ForwardMode(),
    }[args.grad]

    def one(y0_lane, logk, adjoint):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rober), diffrax.Kvaerno5(), t0=0.0, t1=tobs[-1], dt0=1e-6,
            y0=y0_lane, args=jnp.exp(logk), saveat=diffrax.SaveAt(ts=tobs),
            stepsize_controller=controller, max_steps=100_000, adjoint=adjoint,
        )
        return sol.ys

    def batch(y0, logk, adjoint=diffrax.RecursiveCheckpointAdjoint()):
        return jax.vmap(one, in_axes=(0, None, None))(y0, logk, adjoint)

    def loss(x):
        ys = batch(x, logk, adjoint) if args.wrt == "ic" else batch(y0, x, adjoint)
        r = jnp.log(jnp.maximum(ys + delta, common.TINY)) - jnp.log(data + delta)
        return jnp.sum(r**2)

    solve_j = jax.jit(batch)
    x0 = y0 if args.wrt == "ic" else logk
    grad_j = eqx.filter_jit(jax.jacfwd(loss) if args.grad == "fwd" else jax.grad(loss))

    def solve():
        return np.asarray(solve_j(y0, logk)).transpose(2, 0, 1).reshape(3 * n, -1)

    def grad():
        g = np.asarray(grad_j(x0))
        return g.T.ravel() if args.wrt == "ic" else g

    return solve, grad


if __name__ == "__main__":
    common.run(make)
