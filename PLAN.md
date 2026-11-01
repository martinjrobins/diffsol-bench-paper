# Implementation plan: diffsol-bench-paper

## Context
The repo holds only `README.md` (the method) and `PLAN.md` (a high-level plan). This plan is based on reading the local sources:
- `~/git/diffsol` main @ `40d3848` (0.18.0)
- `~/git/diffsl` main (0.12.5)
- `~/git/pydiffsol` (0.6.x)
- PyBaMM, CasADi 3.6.5 and the installed Julia packages
- web documentation for diffrax and SciMLSensitivity

It turns the README into concrete per-library implementations. Agreed decisions:
- Robertson lanes share log k but have distinct ICs.
- Rust diffsol runs only the batched closure version with a manual Jacobian; pydiffsol covers DiffSL with the auto Jacobian.
- Setup includes JIT. Timings at nthreads 1 and 15. Params and IC gradient cases.

Step 0 of implementation: replace the repo's `PLAN.md` with this plan, and correct the README support table (diffrax forward is "forward-mode AD", CasADi forward sensitivities are "augmented states").

## Status (2026-11-01)
Implemented: harness (`common.py`, `run.py`, `rust/src/lib.rs`, `julia/common.jl`), `plot.py`, all Robertson, Brusselator and DFN solver scripts, `dfn/export.py`, `dfn/ds2jl.py` (+ `dfn/test_ds2jl.py`). Full sweeps not yet run. Upstream failures are in `bugs/README.md` and appear as failed rows in the CSVs.

Deviations from the plan below:
- Robertson tobs = logspace(-2, 5, 40), with the loss floored at δ = 10·atol: `log(y+δ)`. Observing values below atol made the IC gradient noise-dominated.
- pydiffsol uses `ds.lu` (faer sparse LU) everywhere; `ds.klu` segfaults.
- Rust diffsol params case carries eps (lane IC) as a 4th per-lane parameter.
- diffrax uses `dt0=1e-6`; `dt0=None` gave NaN gradients.
- Julia: fwd = `ForwardDiffSensitivity`, adj = `InterpolatingAdjoint(ReverseDiffVJP(true))`, no reverse AD. Brusselator is adjoint-only; Sundials is solve-only.
- DFN: 25 inputs, not 30. Inputs that enter the ICs or the geometry can't be exported, and DiffSLExport drops inputs the model doesn't use. Scale factors on function-valued parameters fill some of the gap.
- Brusselator Rust diffsol is skipped; pydiffsol runs the same DiffSL.

## Environment
- **Python:** `uv` with `pyproject.toml` (pixi isn't installed). Dependencies: `pydiffsol>=0.6.2` (its wheels include LLVM and KLU, which sensitivities need), `jax[cpu]`, `diffrax>=0.7.2`, `casadi`, `pybamm>=26.7` (for `DiffSLExport`), `matplotlib`, `numpy`.
- **Julia** (1.12, juliaup): `julia/Project.toml` + `Manifest.toml` with OrdinaryDiffEq (BDF), Sundials, SciMLSensitivity, Zygote, Enzyme, ReverseDiff, SparseConnectivityTracer, ADTypes, LinearSolve (KLU), JSON3.
  - SciMLSensitivity isn't installed yet, and its compatibility with OrdinaryDiffEq v7 / SciMLBase 3 is unverified. Let the resolver choose; if it refuses, pin OrdinaryDiffEq 6.
  - Run `Pkg.precompile()` before any timing.
- **Rust:** `rust/Cargo.toml` with `diffsol = { git, branch = "main" }` (for `solve_dense_vjp`), `nalgebra`, `serde_json`. One binary per benchmark.

## Shared harness
- **`run.py`** is the only orchestrator. For each (bench, solver, variant, size, nthreads, phase, grad, wrt) it:
  1. launches a fresh subprocess;
  2. sets the thread environment: `taskset -c 0` or `0-14`, plus `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=n`, `julia -t n`, `JAX_PLATFORMS=cpu`;
  3. reads peak RSS with `resource.getrusage(RUSAGE_CHILDREN)`;
  4. appends a row to `results/<bench>.csv`.
  - The PLAN's XLA thread flags aren't real XLA options, so `taskset` is the control for JAX.
- **Script contract:** every solver script takes the same CLI (`--size --phase {setup,solve,grad} --grad {fwd,adj,rev} --wrt {params,ic} --jac {auto,manual} --out`). It prints JSON `{setup_s, solve_s, grad_setup_s, grad_s, y_at_tobs, grad}`.
  - setup = build + first solve − median solve.
  - solve/grad = median of k repeats after warm-up.
  - grad_setup = first grad − median grad.
- **Memory:** run phase p and phase p−1 in separate processes; the phase's memory is the peak-RSS difference.
- **Consistency:** `run.py check` compares `y_at_tobs` and `grad` from every solver with the diffsol reference (rtol 1e-3 on the gradient), and compares fwd/adj/rev within each solver. It exits non-zero on any mismatch.
- **Shared problem data:** `common.py` holds the tolerances, `tobs`, the ICs and the reference generation. Julia and Rust read the same values from `results/problem_<bench>_<size>.json`, written by `run.py`, so nothing is duplicated across languages.
- **Same tolerances everywhere:** rtol=1e-6, atol=1e-8 (one value; tighten later if needed). Record the error against a tight-tolerance reference alongside each timing.

## Robertson
**Model:**
- k = exp(log k), with log k₀ = log(0.04, 3e7, 1e4) perturbed by +0.1.
- Lane b has y0 = (1 − ε_b, ε_b, 0), with ε_b = 1e-3·b/nbatch.
- tobs: 40 log-spaced points in [1e-5, 1e5]. Stopping at 1e5 avoids diffsol's adjoint failure beyond ~4e8 (`benches/ode_gradients.rs:147`).
- Loss: L = Σ_b,i,j (log y_bij − log ŷ_bij)², with data from the true log k₀. Clamp y₂ at 1e-20 inside the log.

**Parameterisation:** the ICs are parameters in every library, so one code path gives both the params case (∂/∂log k) and the ic case (∂/∂y0).
- Params case: only log k is an input; y0 is a constant.
- IC case: y0 is the input.

**Solver variants:**
| Solver | File | Implementation |
|---|---|---|
| diffsol (Rust) | `rust/src/bin/robertson.rs` | Copy `crates/diffsol/src/ode_equations/test_models/robertson_ode.rs:454` (batched closures: J·v, **minus** Jᵀv, f_p v), converting to log k by the chain rule (k·∂f/∂k). `NalgebraContext::with_nbatch(n)`; p per lane = log k (repeated) or y0. **fwd:** `bdf_sens` + `solve_dense_sensitivities`. **adj:** `solve_dense_with_checkpointing` → `bdf_solver_adjoint` → `solve_adjoint_backwards_pass(&t, &[&dgdu])`, following `benches/ode_gradients.rs:52-74`. **rev:** `solve_dense_vjp(&t, 1, g, dgdu)` (`step_ad.rs:142`). The per-lane gradient is summed over lanes for the params case. Manual Jacobian only. |
| pydiffsol | `robertson/pydiffsol_.py` | Stacked DiffSL generated in Python (start from `~/git/pydiffsol/bench/diffsol_robertson.py`), `in_i { lk1, lk2, lk3 }` or a vector `in_i { (0:3n): y0 }`. `nalgebra_dense` below 20 lanes, `faer_sparse` + `klu` from 20. **fwd:** `solve_fwd_sens`. **adj:** `solve_adjoint_fwd` + `solve_adjoint_bkwd(sol, ckpt, dL/dout)`. rev: N/A. Auto Jacobian only. |
| diffrax | `robertson/diffrax_.py` | Start from `~/git/pydiffsol/bench/diffrax_robertson.py`. Kvaerno5 + `PIDController`, `jax.vmap` over y0, parameters in `args`, `jax_enable_x64`. Setup via `jax.jit(f).lower().compile()`. **rev:** `RecursiveCheckpointAdjoint` + `jax.grad`. **adj:** `BacksolveAdjoint` + `jax.grad`. **fwd:** `ForwardMode` + `jax.jacfwd`. Auto Jacobian only. |
| CasADi | `robertson/casadi_.py` | Start from `~/git/pydiffsol/bench/casadi_robertson.py`. SX stacked system, `integrator('F','cvodes',dae,0,tobs,{linear_solver:'csparse', abstol, reltol, max_num_steps})`. Force the mode: fwd = `ad_weight=0` (or `L.forward`), adj = `ad_weight=1` (or `L.reverse(1)`). rev: N/A. Auto (symbolic) Jacobian only. |
| Julia | `julia/robertson.jl` | Start from `~/git/pydiffsol/bench/diffeq_robertson.jl`. Stacked in-place `f!`, FBDF. **Auto:** `jac_prototype` from `TracerSparsityDetector`, with `KLUFactorization` from 20 lanes. **Manual:** a hand-written sparse `jac!`. Gradients via SciMLSensitivity + `Zygote.gradient(loss, p)`: fwd = `ForwardSensitivity()`, adj = `GaussAdjoint(autojacvec=EnzymeVJP())`, rev = `ReverseDiffAdjoint()`. |

**Sweep:** nbatch ∈ {1, 3, 10, 30, 100, 300, 1000}. Forward sensitivities in the IC case scale with 3·nbatch; cap at 100 and record the cap.

## Brusselator (SciML tutorial, exact)
- **Model:** p = (A, B, α) = (3.4, 1.0, 10.0). Grid `xyd = range(0, 1, N)`, dx = 1/(N−1). Periodic 5-point Laplacian scaled by α/dx².
  - Forcing `5·[(x−0.3)²+(y−0.6)² ≤ 0.01]·[t ≥ 1.1]`.
  - ICs: u = 22(y(1−y))^1.5, v = 27(x(1−x))^1.5.
  - Time span (0, 11.5). State layout [U(:); V(:)], column-major in (i, j).
  - Keep the tutorial's A/B naming.
  - No `tstops` in any solver, so all handle the discontinuity the same way.
- **Loss:** a squared misfit of U and V at 10 times in [1, 11.5], against data from p₀, evaluated at p₀·1.05.
- **`models/brusselator.py`** generates the forcing mask, the Laplacian non-zeros and the ICs for size N. It emits the DiffSL source (sparse `L_ij { (r,c): v }`, `mask_i`, `heaviside(t-1.1)`) and the CasADi expressions from one source.
- **Solver variants:**
  - **diffsol (Rust):** skipped for Brusselator. pydiffsol already runs the same DiffSL; keep the matrix consistent with the Robertson decision.
  - **pydiffsol:** `faer_sparse` + `klu`, BDF. fwd and adj as for Robertson. For the IC case, `in_i { (0:2N²): y0 }`.
  - **CasADi:** SX + cvodes + csparse.
  - **Julia:** the tutorial `f!`, with
    1. `FBDF(linsolve=KLUFactorization())`, and
    2. `CVODE_BDF(linear_solver=:KLU)`, which needs `jac_prototype`. The Sundials variant gets fwd and adj only; `ReverseDiffAdjoint` can't trace C code. Same `sensealg` choices as for Robertson.
- **Sweep:** N ∈ {8, 16, 32, 64, 128}. The IC case has 2N² inputs; cap fwd at N=16.
- **Manual Jacobian:** none for Brusselator; the README asks for it only for Robertson.

## DFN
1. **`dfn/export.py`:** PyBaMM DFN, choose 30 parameters and set `pv[name] = "[input]"` for each, process and discretise, then `pybamm.DiffSLExport(model).to_diffeq(outputs=["Voltage [V]"])` and `map_inputs`. Recipe from `PyBaMM/tests/integration/test_solvers/test_diffsl.py:72-111`; no experiment, so no `reset`/`N`.
   - Validate the pydiffsol voltage against `pybamm.IDAKLUSolver` (max diff < 1e-4 V).
   - pydiffsol may need `ic_options.armijo_constant = 0.1` (see `diffsol/benches/pybamm_dfn.rs`).
2. **`dfn/ds2jl.py`:** a translator for the DiffSL subset PyBaMM emits:
   - `in_i` scalars;
   - range-run vectors `(a:b): v`;
   - sparse `(r,c): v` matrices, including `(0..n,0..n)` diagonals;
   - `varyingK_i` range concatenations;
   - mat-vecs `(K_ij * x_j)`;
   - slices `x_i[a:b]`;
   - element-wise ops and `pow/exp/log/sqrt/abs/tanh/sinh/max/min/copysign/heaviside`;
   - `u_i`/`dudt_i`/`M_i`/`F_i`/`out_i`.

   It emits a Julia module: constant `SparseMatrixCSC`s, an in-place `f!(du,u,p,t)`, a diagonal mass matrix (0 on algebraic rows), `u0(p)` and `voltage(u,p)`. Use a line and regex parser on the export's regular structure, rather than a full grammar. Translate anything outside the subset with an explicit error. Test the translator by comparing the Julia `f!` with pydiffsol's RHS at a random state (use the DiffSL `F_i` evaluated via pydiffsol, or a solve at t=small).
3. **Julia:** `ODEFunction(f!; mass_matrix=M, jac_prototype)`, `FBDF(linsolve=KLUFactorization())`, default DAE initialisation.
   - fwd = `ForwardSensitivity`, adj = `GaussAdjoint`, rev = `ReverseDiffAdjoint`; mark rev N/A if it fails.
   - Not CVODE: it rejects mass matrices.
4. **pydiffsol:** `faer_sparse` + `klu`, fwd and adj. Loss = voltage misfit at 100 times, against data from the nominal parameters, evaluated at the nominal parameters × 1.05.

## Plots
`plot.py` reads `results/*.csv`:
- One figure per (bench, wrt, nthreads), with 2×3 panels: time and memory against setup, solve and grad.
- Colour by solver, line style by Jacobian option or gradient method.
- An extra figure of nthreads 15/1 speed-up ratios.
- Unsupported combinations are simply missing.

## Threading expectations (to state in the paper)
Expected to be unchanged at 15 threads: Rust diffsol (nalgebra), pydiffsol (no thread API; faer `Par::Seq`), CasADi, Julia FBDF/KLU/Sundials. JAX and possibly Julia/Enzyme or BLAS paths may speed up. Measure; don't assume.

## Order of work
1. `pyproject.toml`, `julia/Project.toml`, `rust/Cargo.toml`, `common.py`, `run.py`.
2. Robertson pydiffsol end to end with consistency checks and a plot.
3. Robertson Rust diffsol (the reference), then diffrax, CasADi, Julia.
4. Brusselator generator + pydiffsol, CasADi, Julia.
5. DFN export + validation, then `ds2jl.py` + translator test, then Julia DFN.
6. Full sweeps on a quiet machine (32 cores; pin 0–14).

## Verification
- `uv run run.py check robertson --size 10` and the same for brusselator `--size 16` and dfn: every solver's solution and gradient must agree with the reference, and fwd/adj/rev must agree with each other.
- A finite-difference check of the reference gradient in `common.py` (central difference on 3 log k and on (A, B, α)).
- DFN: pydiffsol voltage vs PyBaMM IDAKLU; Julia `f!` vs DiffSL RHS; then Julia vs pydiffsol voltage.
- Smoke sweep: `uv run run.py sweep --quick` (smallest 2 sizes, 1 repeat), then `uv run plot.py`, then inspect the PNGs.
