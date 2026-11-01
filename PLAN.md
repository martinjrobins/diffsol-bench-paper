# Implementation plan

Implements the benchmarks described in `README.md`.

## 0. Decisions

- **Gradient targets.** Every benchmark has two gradient cases:
  - `params`: Robertson w.r.t. log k (3 values), Brusselator w.r.t. (A, B, α), DFN w.r.t. the 30 parameters.
  - `ic`: w.r.t. the initial conditions (Robertson: 3·nbatch, Brusselator: 2N²). Not run for DFN.
  - Each case is computed with forward sensitivities, the continuous adjoint and reverse-mode AD, so all three can be checked against each other.
  - Forward sensitivities w.r.t. the ICs scale with the state size. Cap the sweep where they become impractical and record the cap.
- **Threads.** Every timing is generated at `nthreads=1` and `nthreads=15`, so libraries that use multithreading get credit for it.
- **diffsol reverse AD.** Use `solve_dense_vjp` from the diffsol `main` branch (git dependency in `rust/Cargo.toml`).
- **Setup includes JIT.** Setup and JIT are a single phase, because in pydiffsol they can't be separated. Setup time = (build + first solve) − steady-state solve time, so the solve itself isn't counted twice.

### Method support table (check each cell; unsupported cells are N/A in plots)

| | fwd sens | cont. adjoint | reverse AD | manual jac | sparse |
|---|---|---|---|---|---|
| diffsol (Rust) | yes | yes | `solve_dense_vjp` | yes | yes |
| pydiffsol | yes | yes | no | no (diffsl auto) | yes |
| diffrax | `ForwardMode` / jacfwd | `BacksolveAdjoint` | `RecursiveCheckpointAdjoint` | no | no |
| CasADi | CVODES/IDAS fwd | CVODES/IDAS adj | ≈ adjoint (label as such) | auto only | yes |
| Julia | `ForwardSensitivity` / `ForwardDiffSensitivity` | `GaussAdjoint` / `InterpolatingAdjoint` | `ReverseDiffAdjoint` | yes | yes |

## 1. Repo layout

```
pixi.toml                 # python env: pydiffsol, diffrax/jax, casadi, pybamm, matplotlib
julia/Project.toml (+Manifest)
rust/Cargo.toml           # diffsol (git main), nalgebra, faer
models/robertson.ds, brusselator.ds   # diffsl, shared by Rust + pydiffsol
robertson/{diffsol.rs, pydiffsol.py, diffrax.py, casadi.py, julia.jl}
brusselator/{diffsol.rs, pydiffsol.py, casadi.py, julia.jl}
dfn/{export.py, ds2jl.py, pydiffsol.py, julia.jl}
run.py                    # driver
plot.py
results/*.csv
```

## 2. Harness (built first)

- **CLI contract** for every solver script:
  `--size N --nthreads {1,15} --phase {setup,solve,grad} --grad {fwd,adj,rev} --wrt {params,ic} --jac {auto,manual} --out file.json`
  Output JSON: `{setup_s, solve_s, grad_setup_s, grad_s, peak_rss_bytes, y_at_tobs, grad}`.
- **Threads.** `run.py` sets, per run:
  - `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` = nthreads
  - Julia: `julia -t nthreads`, plus `BLAS.set_num_threads(nthreads)`
  - JAX: at 1 thread `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`; at 15, pin the process to 15 cores (`taskset`)
  - Always pin to a fixed core set with `taskset` so 1 vs 15 is comparable.
- **Timing.**
  - setup: build the problem + first solve, minus the median steady-state solve time. This covers diffsl compile, Julia time-to-first-solve, and JAX lowering and compile.
  - solve and grad: median of k repeats after warm-up.
  - First gradient call: report its compile overhead as part of grad setup in the same way, (first grad − median grad), stored as `grad_setup_s`.
- **Memory.** Fresh subprocess per (solver, size, nthreads, phase); peak RSS via `resource.getrusage(RUSAGE_CHILDREN)`. Each phase's cost is the peak of a run up to that phase minus the peak of a run up to the previous phase.
- **Fairness.** Same rtol/atol everywhere, JAX on CPU with `jax_enable_x64`, and record the error against the reference alongside each timing.
- **Consistency.** `run.py` checks `y_at_tobs` and `grad` against the reference within a tolerance, and across the fwd/adj/rev methods, and fails loudly on mismatch.

## 3. Robertson (milestone 1: end to end)

1. **Reference:** diffsol at rtol=1e-10, observations at log-spaced t ∈ [1e-5, 1e5].
   - Loss: Σ (log yᵢ − log ŷᵢ)², parameterised by log k.
   - Evaluate gradients at a perturbed log k so they aren't zero.
2. **pydiffsol:** build it with the harness and a first plot to prove the pipeline.
3. **diffsol (Rust):** nalgebra dense with native batching, auto and manual Jacobian. Gradients use forward sensitivities, the adjoint and `solve_dense_vjp`.
4. **diffrax:** Kvaerno5 + `PIDController` (stiff ODE example), `vmap` over the batch.
5. **CasADi:** stack nbatch copies into one system, `integrator('cvodes', ...)` with a sparse linear solver.
6. **Julia:** FBDF on the stacked system.
   - Auto Jacobian: `jac_prototype` from SparseConnectivityTracer.
   - Manual Jacobian: a hand-written `jac!`.
   - Gradients: SciMLSensitivity, all three modes.
7. Sweep nbatch ∈ {1, 3, 10, 30, 100, 300, 1000} × nthreads ∈ {1, 15}.

## 4. Brusselator

1. Port the SciML tutorial model exactly: α, A, B, forcing, periodic BCs and initial conditions. The tutorial code is the Julia implementation and a correctness check.
2. Write the diffsl version with a sparse Laplacian, shared by pydiffsol and Rust diffsol (faer sparse backend).
3. **CasADi:** CVODES with a sparse direct solver.
4. **Julia:**
   - FBDF + KLU
   - `CVODE_BDF(linear_solver=:KLU)`, to compare Sundials via Julia against CasADi's Sundials
5. Sweep N ∈ {8, 16, 32, 64, 128} × nthreads ∈ {1, 15}. Cap N where needed and record the cap.

## 5. DFN

1. **`export.py`:** PyBaMM DFN with 30 parameters as inputs and voltage as output, exported to diffsl. Validate the pydiffsol solution against PyBaMM's own solver.
2. **`ds2jl.py`:** a mechanical diffsl → Julia translator producing a mass-matrix DAE (FBDF, sparse auto Jacobian).
   - Time-box it.
   - Fallback: hand-port only the operators that are hard to translate.
3. **Loss:** voltage misfit against synthetic data from the nominal parameters. Gradients w.r.t. the 30 parameters using fwd/adj/rev, pydiffsol vs Julia, nthreads ∈ {1, 15}.

## 6. Plots

`plot.py` makes one figure per benchmark and per gradient case (`params`, `ic`):
- Grid of 2×3 panels: rows are time and memory; columns are setup (incl. JIT), solve and grad.
- Colour by solver, line style by Jacobian option or gradient method, marker by nthreads.
- Log–log axes.

## Order

1. Harness + pydiffsol Robertson + plot
2. Remaining Robertson solvers, with consistency checks passing
3. Brusselator
4. DFN export, validated against PyBaMM
5. DFN Julia port
6. Final runs on a quiet machine with pinned cores
