# Timing harness shared by the Julia scripts; mirrors `common.run` in Python.
using JSON3, Statistics

function parse_args()
    a = Dict("--phase" => "grad", "--grad" => "fwd", "--wrt" => "params", "--jac" => "auto")
    for i in 1:2:length(ARGS)
        a[ARGS[i]] = ARGS[i+1]
    end
    a
end

function timed(f)
    t0 = time_ns()
    r = f()
    (time_ns() - t0) / 1e9, r
end

function median_time(f, t_first)
    nrep = clamp(floor(Int, 1.0 / max(t_first, 1e-9)), 3, 10)
    median([timed(f)[1] for _ in 1:nrep])
end

matrix(rows) = permutedims(reduce(hcat, [collect(Float64, r) for r in rows]))

"""make(prob, args) -> (solve, grad); solve() -> y at tobs as (nout, nt), grad() -> gradient vector."""
function run(make)
    args = parse_args()
    prob = JSON3.read(read(args["--problem"], String))
    res = Dict{String,Any}()
    if args["--phase"] != "import"
        t_make, (solve, grad) = timed(() -> make(prob, args))
        t_first, y = timed(solve)
        t_warm, _ = timed(solve)
        res["solve_s"] = median_time(solve, t_warm)
        res["setup_s"] = t_make + t_first - res["solve_s"]
        res["y_at_tobs"] = vec(permutedims(y))  # row-major, like numpy ravel
        if args["--phase"] == "grad"
            t_first, g = timed(grad)
            res["grad_s"] = median_time(grad, t_first)
            res["grad_setup_s"] = t_first - res["grad_s"]
            res["grad"] = collect(g)
        end
    end
    write(args["--out"], JSON3.write(res))
end
