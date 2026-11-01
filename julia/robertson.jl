# Stacked batched Robertson: u = [x_1..x_n; y_1..y_n; z_1..z_n], p = log k.
include("common.jl")
using OrdinaryDiffEqBDF, SciMLSensitivity, Zygote, ReverseDiff
using ADTypes, SparseConnectivityTracer, LinearSolve, SparseArrays, LinearAlgebra

function rober!(du, u, p, t)
    n = length(u) ÷ 3
    k1, k2, k3 = exp(p[1]), exp(p[2]), exp(p[3])
    x, y, z = @view(u[1:n]), @view(u[n+1:2n]), @view(u[2n+1:3n])
    @. du[1:n] = -k1 * x + k3 * y * z
    @. du[n+1:2n] = k1 * x - k2 * y^2 - k3 * y * z
    @. du[2n+1:3n] = k2 * y^2
    nothing
end

function rober_jac!(J, u, p, t)
    n = length(u) ÷ 3
    k1, k2, k3 = exp(p[1]), exp(p[2]), exp(p[3])
    fill!(J, 0)
    for b in 1:n
        x, y, z = u[b], u[n+b], u[2n+b]
        ix, iy, iz = b, n + b, 2n + b
        J[ix, ix] = -k1; J[ix, iy] = k3 * z; J[ix, iz] = k3 * y
        J[iy, ix] = k1; J[iy, iy] = -2k2 * y - k3 * z; J[iy, iz] = -k3 * y
        J[iz, iy] = 2k2 * y
    end
    nothing
end

function make(prob, args)
    n, tobs = prob.n, collect(Float64, prob.tobs)
    data, δ = matrix(prob.data), prob.delta
    y0, logk = collect(Float64, prob.y0), collect(Float64, prob.logk)
    sparse_ = n >= 20
    jp = sparse_ ? float.(ADTypes.jacobian_sparsity((du, u) -> rober!(du, u, logk, 0.0), similar(y0), y0, TracerSparsityDetector())) : nothing
    jac = args["--jac"] == "manual" ? rober_jac! : nothing
    # FullSpecialize: the default FunctionWrappers specialisation rejects a user jac under ForwardDiff
    f = ODEFunction{true,SciMLBase.FullSpecialize}(rober!; jac, jac_prototype=jp)
    prob_ = ODEProblem(f, y0, (0.0, tobs[end]), logk)
    alg = sparse_ ? FBDF(linsolve=KLUFactorization()) : FBDF()
    kw = (saveat=tobs, abstol=prob.atol, reltol=prob.rtol, maxiters=1_000_000)
    # see bugs/README.md for why these and not ForwardSensitivity/GaussAdjoint/ReverseDiffAdjoint
    sensealg = Dict(
        "fwd" => ForwardDiffSensitivity(),
        "adj" => InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
    )[args["--grad"]]
    logd = log.(data .+ δ)
    loss(u0, p) = sum(abs2, log.(max.(Array(solve(prob_, alg; u0, p, sensealg, kw...)) .+ δ, 1e-30)) .- logd)

    solve_() = Array(solve(prob_, alg; kw...))
    grad() = args["--wrt"] == "ic" ? Zygote.gradient(u0 -> loss(u0, logk), y0)[1] : Zygote.gradient(p -> loss(y0, p), logk)[1]
    solve_, grad
end

run(make)
