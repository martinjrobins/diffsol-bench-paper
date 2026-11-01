"""diffsol continuous adjoint is fragile on Robertson with a data-misfit loss.

The backward pass fails ("Exceeded maximum number of error test failures" or "... nonlinear
solver failures") depending on the magnitude of dgdu and on rtol/atol, although the adjoint is
linear in dgdu. Seen with pydiffsol 0.6.2 and with Rust diffsol main @ 40d3848
(rust/src/bin/robertson.rs --grad adj). Forward sensitivities and solve_dense_vjp succeed.
"""

import numpy as np
import pydiffsol as ds

code = """
in_i { lk1 = 0, lk2 = 0, lk3 = 0 }
k1 { exp(lk1) } k2 { exp(lk2) } k3 { exp(lk3) }
u_i { x = 1, y = 0, z = 0 }
F_i { -k1 * x + k3 * y * z, k1 * x - k2 * y * y - k3 * y * z, k2 * y * y }
out_i { x, y, z }
"""
p = np.log([0.04, 3e7, 1e4])
t = np.logspace(-1, 5, 20)
for rtol, atol in [(1e-6, 1e-8), (1e-8, 1e-10)]:
    ode = ds.Ode(code, matrix_type=ds.nalgebra_dense)
    ode.rtol, ode.atol = rtol, atol
    for scale in [1e-6, 1.0, 1e4, 1e8]:
        sol, ckpt = ode.solve_adjoint_fwd(p, t)
        dgdu = np.zeros_like(sol.ys)
        dgdu[1] = scale  # x+y+z is conserved, so weight y only
        try:
            g = ode.solve_adjoint_bkwd(sol, ckpt, dgdu)
            print(f"rtol {rtol:g} scale {scale:g}: ok, grad/scale = {g.ravel() / scale}")
        except RuntimeError as e:
            print(f"rtol {rtol:g} scale {scale:g}: {e}")
