"""Export the PyBaMM DFN (30 scalar inputs, voltage output) to dfn/dfn.ds and validate it.

uv run dfn/export.py
"""

import json
from pathlib import Path

import numpy as np
import pybamm
import pydiffsol as ds

HERE = Path(__file__).parent
T_END = 2500.0  # s; stays above the 3.105 V cut-off at 1C with inputs scaled by 1.05

# scalar parameters made inputs (None: use the parameter set's value). Parameters that enter the
# initial conditions (max/initial concentrations, porosities, initial temperature) are excluded
# because DiffSLExport rejects input-dependent initial conditions; geometry ones can't be inputs.
INPUTS = {
    "Negative electrode conductivity [S.m-1]": None,
    "Positive electrode conductivity [S.m-1]": None,
    "Negative electrode active material volume fraction": None,
    "Positive electrode active material volume fraction": None,
    "Negative electrode Bruggeman coefficient (electrolyte)": None,
    "Positive electrode Bruggeman coefficient (electrolyte)": None,
    "Separator Bruggeman coefficient (electrolyte)": None,
    "Negative electrode Bruggeman coefficient (electrode)": None,
    "Positive electrode Bruggeman coefficient (electrode)": None,
    "Negative electrode charge transfer coefficient": None,
    "Positive electrode charge transfer coefficient": None,
    "Initial concentration in electrolyte [mol.m-3]": None,
    "Cation transference number": None,
    "Thermodynamic factor": None,
    "Current function [A]": None,
    "Contact resistance [Ohm]": None,
    "Ambient temperature [K]": None,
    "Electrode height [m]": None,
    "Electrode width [m]": None,
    "Number of electrodes connected in parallel to make a cell": None,
    # only used by the stop events, so their voltage gradient is zero
    "Lower voltage cut-off [V]": None,
    "Upper voltage cut-off [V]": None,
}
# function-valued parameters: input "<name> scale" multiplies, "<name> shift" adds
# (OCP shifts would be natural but enter the initial conditions too)
FUNCS = {
    "Negative particle diffusivity [m2.s-1]": "scale",
    "Positive particle diffusivity [m2.s-1]": "scale",
    "Electrolyte diffusivity [m2.s-1]": "scale",
    "Electrolyte conductivity [S.m-1]": "scale",
    "Negative electrode exchange-current density [A.m-2]": "scale",
    "Positive electrode exchange-current density [A.m-2]": "scale",
}
OUTPUT = "Voltage [V]"


def main():
    model = pybamm.lithium_ion.DFN()
    pv = model.default_parameter_values.copy()
    nominal = {k: float(pv[k] if v is None else v) for k, v in INPUTS.items()}
    for k in INPUTS:
        pv[k] = "[input]"
    for k, kind in FUNCS.items():
        name, f = f"{k} {kind}", pv[k]
        p = pybamm.InputParameter(name)
        pv[k] = (lambda f, p: lambda *a: p * f(*a))(f, p) if kind == "scale" else (lambda f, p: lambda *a: f(*a) + p)(f, p)
        nominal[name] = 1.0 if kind == "scale" else 0.0
    print(f"{len(nominal)} inputs")
    geometry = model.default_geometry
    model_disc = pv.process_model(model, inplace=False)
    pv.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    pybamm.Discretisation(mesh, model.default_spatial_methods).process_model(model_disc, inplace=True)

    exporter = pybamm.DiffSLExport(model_disc)
    code = exporter.to_diffeq(outputs=[OUTPUT])
    p_nominal = exporter.map_inputs(nominal, outputs=[OUTPUT])
    p_eval = exporter.map_inputs({k: 1.05 * v for k, v in nominal.items()}, outputs=[OUTPUT])
    (HERE / "dfn.ds").write_text(code)

    tobs = np.linspace(0, T_END, 100)
    idaklu = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12, output_variables=[OUTPUT])
    v_pybamm = idaklu.solve(model_disc, [0, T_END], t_interp=tobs, inputs=nominal)[OUTPUT].data

    ode = ds.Ode(code, matrix_type=ds.faer_sparse, linear_solver=ds.lu, ode_solver=ds.bdf)
    ode.rtol, ode.atol = 1e-10, 1e-12
    ode.ic_options.armijo_constant = 1e-1
    v_diffsol = ode.solve_dense(p_nominal, tobs).ys[0]
    err = np.max(np.abs(v_diffsol - v_pybamm))
    print(f"max |V_pydiffsol - V_pybamm| = {err:.2e} V")
    assert err < 1e-4, err

    (HERE / "dfn.json").write_text(json.dumps(dict(
        inputs=list(nominal), nominal=list(nominal.values()),
        p_nominal=p_nominal.tolist(), p=p_eval.tolist(),
        tobs=tobs.tolist(), data=[v_pybamm.tolist()],
    )))


if __name__ == "__main__":
    main()
