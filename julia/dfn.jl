# DFN translated from PyBaMM's DiffSL export by dfn/ds2jl.py; mass-matrix DAE, p = the 25 inputs.
include("common.jl")
include("dfn_gen.jl")
using OrdinaryDiffEqBDF, SciMLSensitivity, Zygote, ReverseDiff
using OrdinaryDiffEqCore: BrownFullBasicInit
using ADTypes, SparseConnectivityTracer, LinearSolve, SparseArrays, LinearAlgebra

function make(prob, args)
    tobs, data, p0 = collect(Float64, prob.tobs), matrix(prob.data), collect(Float64, prob.p)
    u0 = dfn_u0(p0)
    jp = float.(ADTypes.jacobian_sparsity((du, u) -> dfn_rhs!(du, u, p0, 0.0), similar(u0), u0, TracerSparsityDetector()))
    f = ODEFunction{true,SciMLBase.FullSpecialize}(dfn_rhs!; mass_matrix=dfn_mass, jac_prototype=jp)
    prob_ = ODEProblem(f, u0, (0.0, tobs[end]), p0)
    alg = FBDF(linsolve=KLUFactorization())
    # v7 only checks the algebraic ICs by default; compute consistent ones like pydiffsol does
    kw = (saveat=tobs, abstol=prob.atol, reltol=prob.rtol, maxiters=1_000_000, initializealg=BrownFullBasicInit())
    sensealg = Dict(
        "fwd" => ForwardDiffSensitivity(),
        "adj" => InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
    )[args["--grad"]]
    voltage(sol, p) = permutedims(reduce(vcat, [dfn_out(u, p, 0.0) for u in sol.u]))
    loss(p) = sum(abs2, voltage(solve(prob_, alg; u0=dfn_u0(p), p, sensealg, kw...), p) .- data)

    solve_() = voltage(solve(prob_, alg; kw...), p0)
    grad() = Zygote.gradient(loss, p0)[1]
    solve_, grad
end

run(make)
