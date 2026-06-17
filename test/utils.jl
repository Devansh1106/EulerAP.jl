module TestUtils

using Random
using EulerAP

export small_problem, set_seed!

function set_seed!(s::Integer)
    Random.seed!(s)
end

function small_problem(; size=(8, 8))
    ic = (x, y, t) -> (1.0, 0.0, 0.0)
    u0, coords, p, cache = build_problem(ic_func = ic, size = size, tspan = (0.0, 1.0))
    return u0, coords, p, cache
end

end # module
