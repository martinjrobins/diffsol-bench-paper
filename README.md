# diffsol-bench-paper

This repo compares the performance of diffsol, a differentiable ODE/DAE solver, with diffrax, DifferentialEquations.jl and CasADi.

Three benchmarks are included:

- Robertson ODE, a stiff 3-dimensional system that is batched so that `nbatch` independent Robertson ODEs are solved at once.
- A 2D Brusselator PDE, a reaction-diffusion system solved using the method of lines, with grid size `N` varied.
- A physics-based battery model, the Doyle-Fuller-Newman (DFN) model, which is a stiff system of ODEs and DAEs.

See `PLAN.md` for the implementation plan.

## Method

### Measurements (all benchmarks)

For each solver and problem size the following are measured:

- **setup**: building the problem plus the first solve (which includes any JIT compilation), minus the steady-state solve time so the solve itself is not counted. Setup and JIT are combined because they cannot be separated in pydiffsol.
- **solve**: median time of repeated solves after warm-up.
- **gradient**: median time of repeated gradient computations after warm-up.

Both wall time and peak memory are recorded for each phase. All timings are generated with `nthreads=1` and `nthreads=15`, so libraries that use multithreading can take advantage of it.

Gradients of a loss function are computed via forward sensitivity analysis, solution of the continuous adjoint equations, and reverse-mode automatic differentiation, where the library supports it. Two gradient cases are run:

- **params**: w.r.t. the model parameters (see each benchmark below).
- **ic**: w.r.t. the initial conditions (Robertson and Brusselator only).

All solutions and gradients are compared to each other and to a reference solution to ensure they are consistent.

Library support for the gradient methods:

| | forward sensitivities | continuous adjoint | reverse-mode AD |
|---|---|---|---|
| diffsol (Rust) | yes | yes | yes (`solve_dense_vjp`, diffsol `main` branch) |
| pydiffsol | yes | yes | no |
| diffrax | yes | yes | yes |
| CasADi | yes | yes | no (CasADi derivatives use the adjoint) |
| DifferentialEquations.jl | yes | yes | yes |

### Robertson ODE

A batched Robertson ODE model is implemented in Rust (diffsol), Python (pydiffsol, diffrax, CasADi) and Julia (DifferentialEquations.jl). The model is solved for `nbatch` independent ODEs, where `nbatch` is varied from 1 to 1000.

The params gradient is taken w.r.t. the log rate constants, log kᵢ, since the constants span many orders of magnitude. The ic gradient is taken w.r.t. the initial conditions of all `nbatch` ODEs. The loss uses data at log-spaced times with a log-scale misfit, because y₂ is around 10⁻⁵ and would otherwise contribute almost nothing to a plain least-squares loss.

diffsol and diffrax batch natively, so they implement the batched Robertson ODE with dense linear algebra (the nalgebra backend for diffsol). pydiffsol, CasADi and Julia instead extend the system to include `nbatch` independent copies of the ODE and solve the resulting system with sparse linear algebra. All libraries except diffrax have a BDF solver, so BDF is used in general. diffrax uses Kvaerno5, following its stiff ODE example: <https://docs.kidger.site/diffrax/examples/stiff_ode/>. Where a library supports both automatic and manual Jacobians, both are run.

Example plots:

- Time vs nbatch for setup, solve and gradient computations, one line for each of diffsol, pydiffsol, diffrax, CasADi and Julia.
- Memory vs nbatch for setup, solve and gradient computations, one line for each of diffsol, pydiffsol, diffrax, CasADi and Julia.

### 2D Brusselator PDE

The model is the Brusselator from the Julia documentation: <https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Definition-of-the-Brusselator-Equation>. diffrax does not have sparse linear algebra, so it is not used for this benchmark. The other solvers use BDF and sparse direct solvers. Julia is run with both its native BDF and the Sundials BDF, so that Sundials called via Julia can be compared with CasADi's implementation (which also uses Sundials).

The PDE is solved using the method of lines, with the grid size `N` varied. The params gradient is taken w.r.t. (A, B, α). The ic gradient is taken w.r.t. the initial conditions of the PDE.

### DFN battery model

The PyBaMM library defines a DFN model, which is written out to diffsl using PyBaMM's diffsl export. This diffsl is then ported to Julia so that pydiffsol and Julia can be compared; only these two are used for this benchmark. The Julia code uses automatic Jacobian computation. The loss function uses a voltage output and 30 of the model parameters, and its gradient w.r.t. the 30 parameters is computed using forward sensitivity analysis, the continuous adjoint equations and, in Julia, reverse-mode automatic differentiation. Measurements are the same as for the other benchmarks.
