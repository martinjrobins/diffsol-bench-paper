"""Check the translated Julia DFN against pydiffsol at a perturbed state: uv run dfn/test_ds2jl.py"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pydiffsol as ds

ROOT = Path(__file__).parents[1]
prob = json.loads((ROOT / "dfn/dfn.json").read_text())
ode = ds.Ode((ROOT / "dfn/dfn.ds").read_text(), matrix_type=ds.faer_sparse, linear_solver=ds.lu)
p = np.array(prob["p"])
y0 = ode.y0(p)
y = y0 * (1 + 1e-3 * np.random.default_rng(0).standard_normal(y0.size))
f = ode.rhs(p, 10.0, y)
with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
    json.dump(dict(p=p.tolist(), y=y.tolist()), tmp)
    tmp.flush()
    jl = f"""
    include("{ROOT}/julia/dfn_gen.jl"); using JSON3
    d = JSON3.read(read("{tmp.name}", String))
    p, y = collect(Float64, d.p), collect(Float64, d.y)
    println(JSON3.write(Dict("f" => dfn_eval(y, p, 10.0)[1], "u0" => dfn_u0(p))))
    """
    res = json.loads(subprocess.run(["julia", f"--project={ROOT}/julia", "-e", jl], capture_output=True, text=True, check=True).stdout.splitlines()[-1])
err_f = np.max(np.abs(np.array(res["f"]) - f) / (np.abs(f) + 1e-8))
err_u0 = np.max(np.abs(np.array(res["u0"]) - y0))
print(f"max rel err rhs {err_f:.1e}, max abs err u0 {err_u0:.1e}")
assert err_f < 1e-10 and err_u0 < 1e-12
