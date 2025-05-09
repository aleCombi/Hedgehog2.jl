using StochasticDiffEq, DiffEqNoiseProcess, BenchmarkTools, Profile, Random, SciMLBase

struct FakeSolution{T,U}
    t::Vector{T}
    u::Vector{U}
end

# Alias .W to .u
function Base.getproperty(sol::FakeSolution, name::Symbol)
    if name === :W
        return getfield(sol, :u)
    else
        return getfield(sol, name)
    end
end

# Overload Base.iterate for FakeSolution
function Base.iterate(sol::FakeSolution)
    # Check if the solution's 'u' vector is empty
    if isempty(sol.u)
        return nothing # If empty, there's nothing to iterate over
    else
        # Return the first element (sol.u[1]) and set the state for the next iteration to 2
        return (sol.u[1], 2)
    end
end

function Base.iterate(sol::FakeSolution, state)
    # Check if the current 'state' (which acts as an index) is within the bounds of the 'u' vector
    if state <= length(sol.u)
        # Return the element at the current 'state' index and increment the state for the next iteration
        return (sol.u[state], state + 1)
    else
        # If the 'state' is out of bounds, iteration is complete
        return nothing
    end
end

# Alias .W to .u
function Base.getproperty(sol::FakeSolution, name::Symbol)
    if name === :W
        return getfield(sol, :u)
    else
        return getfield(sol, name)
    end
end

struct FakeProblem{P<:NoiseProblem, F<:FakeSolution}
    base_prob::P         # shared setup (no duplication)
    solutions::Vector{F}  # precomputed or simulated paths
end

Base.length(v::FakeProblem) = Base.length(v.solutions)
Base.getindex(fp::FakeProblem, i::Int) = fp.solutions[i]
Base.iterate(fp::FakeProblem) = iterate(fp.solutions)
Base.iterate(fp::FakeProblem, state) = iterate(fp.solutions, state)
Base.length(v::FakeSolution) = Base.length(v.u)
Base.iterate(fp::FakeSolution) = iterate(fp.u)

r, σ, t₀, S₀ = 0.02, 0.3, 0.0, 1.0
T = 1.0
tspan = (t₀, T)
noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
noise_problem = NoiseProblem(noise, tspan)

ΔW = [0.0, 0.0]
sol = [1.0, 1.0]
ΔT = 0.1
t_next = ΔT
rng = Xoshiro()
sol = FakeSolution([0.0], [[1.0], [1.0]])
# noise.dist(ΔW, sol, ΔT, sol, nothing, t_next, rng)

struct MyDist{T1, T2}
    μ::T1
    σ::T2
end

function (X::MyDist)(dW, W, dt, u, p, t, rng) #dist
    drift = X.μ - (1 / 2) * X.σ^2
    if dW isa AbstractArray
        rand_val = randn(rng, size(dW))
    else
        rand_val = randn(rng, typeof(dW))
    end

    new_val = exp.(drift * dt .+ X.σ * sqrt(dt) .* rand_val)
    return getindex.(W.W, i) .* (new_val .- 1)
end

function MyGeometricBrownianMotionProcess(μ, σ, t0, W0, Z0 = nothing; kwargs...)
    gbm = MyDist(μ, σ)
    NoiseProcess{false}(t0, W0, Z0, gbm,
        (dW, W, W0, Wh, q, h, u, p, t, rng) -> nothing; kwargs...)
end

gbm = MyDist(0.05, 0.2)
n = 5000
dW = zeros(n)
W = [[1.0] for _ in 1:n]
dt = 0.1
u, p, t, rng, i = W, nothing, 1.0, Xoshiro(), 1
# gbm(dW, W, dt, u, p, t, rng, i)


noise = MyGeometricBrownianMotionProcess(r, σ, t₀, S₀)
noise_problem = NoiseProblem(noise, tspan)

function simple_solve(noise_problem; steps, trajectories)
    N = trajectories
    rng = Xoshiro()
    T = noise_problem.tspan[end]
    gbm = noise_problem.noise
    ΔT = T / steps
    # Preallocate W and u as vectors of vectors
    W = [Vector{Float64}(undef, steps + 1) for _ in 1:N]

    # Set initial values
    for j in 1:N
        W[j][1] = gbm.u[1]
    end

    times = collect(range(0.0, 1.0; length=steps+1))
    sol = FakeSolution(times, W)
    ΔW = zeros(N)

    for i in 1:steps
        t_next = sol.t[end] + ΔT
        ΔW = gbm.dist(ΔW, sol, ΔT, sol, nothing, t_next, rng, i)
        for j in eachindex(ΔW)
            sol.W[j][i+1] = sol.W[j][i] + ΔW[j]
        end
    end

    return sol
end

s = simple_solve(noise_problem; steps=1, trajectories=5000)