# 2D Brusselator from the SciML advanced ODE tutorial; u = [U(:); V(:)], p = (A, B, alpha).
# --alg fbdf (OrdinaryDiffEq FBDF + KLU) or cvode (Sundials CVODE_BDF + KLU).
include("common.jl")
using OrdinaryDiffEqBDF, Sundials, SciMLSensitivity, Zygote, ReverseDiff
using ADTypes, SparseConnectivityTracer, LinearSolve, SparseArrays, LinearAlgebra
import DifferentiationInterface as DI, ForwardDiff
using SparseMatrixColorings: GreedyColoringAlgorithm

struct Bruss
    N::Int
    xyd::Vector{Float64}
    dx::Float64
end

brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function (b::Bruss)(du, u, p, t)
    N = b.N
    A, B, alpha = p[1], p[2], p[3] / b.dx^2
    U = reshape(@view(u[1:N*N]), N, N)
    V = reshape(@view(u[N*N+1:end]), N, N)
    dU = reshape(@view(du[1:N*N]), N, N)
    dV = reshape(@view(du[N*N+1:end]), N, N)
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = b.xyd[i], b.xyd[j]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N), limit(j - 1, N)
        dU[i, j] = alpha * (U[im1, j] + U[ip1, j] + U[i, jp1] + U[i, jm1] - 4U[i, j]) +
                   B + U[i, j]^2 * V[i, j] - (A + 1) * U[i, j] + brusselator_f(x, y, t)
        dV[i, j] = alpha * (V[im1, j] + V[ip1, j] + V[i, jp1] + V[i, jm1] - 4V[i, j]) +
                   A * U[i, j] - U[i, j]^2 * V[i, j]
    end
    nothing
end

function make(prob, args)
    N, tobs = prob.n, collect(Float64, prob.tobs)
    data = matrix(prob.data)
    y0, p0 = collect(Float64, prob.y0), collect(Float64, prob.p)
    f! = Bruss(N, collect(range(0, 1, length=N)), 1 / (N - 1))
    jp = float.(ADTypes.jacobian_sparsity((du, u) -> f!(du, u, p0, 0.0), similar(y0), y0, TracerSparsityDetector()))
    cvode = get(args, "--alg", "fbdf") == "cvode"
    # CVODE + KLU needs a jac function: coloured sparse ForwardDiff, same AD FBDF uses internally
    jac = nothing
    if cvode
        backend = AutoSparse(AutoForwardDiff(); sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm())
        du0 = similar(y0)
        prep = DI.prepare_jacobian(f!, du0, backend, y0, DI.Constant(p0), DI.Constant(0.0))
        jac = (J, u, p, t) -> DI.jacobian!(f!, du0, J, prep, backend, u, DI.Constant(p), DI.Constant(t))
    end
    prob_ = ODEProblem(ODEFunction{true,SciMLBase.FullSpecialize}(f!; jac, jac_prototype=jp), y0, (0.0, tobs[end]), p0)
    alg = cvode ? CVODE_BDF(linear_solver=:KLU) : FBDF(linsolve=KLUFactorization())
    kw = (saveat=tobs, abstol=prob.atol, reltol=prob.rtol, maxiters=1_000_000)
    # forward and all CVODE gradients fail in this stack (bugs/README.md)
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    loss(u0, p) = sum(abs2, Array(solve(prob_, alg; u0, p, sensealg, kw...)) .- data)

    solve_() = Array(solve(prob_, alg; kw...))
    grad() = args["--wrt"] == "ic" ? Zygote.gradient(u0 -> loss(u0, p0), y0)[1] : Zygote.gradient(p -> loss(y0, p), p0)[1]
    solve_, grad
end

run(make)
