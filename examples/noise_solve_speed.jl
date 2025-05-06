using StochasticDiffEq, DiffEqNoiseProcess, BenchmarkTools, Profile

r, σ, t₀, S₀ = 0.02, 0.2, 0.0, 1.0
T = 1.0
tspan = (t₀, T)
noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
noise_problem = NoiseProblem(noise, tspan)

ens_problem = EnsembleProblem(noise_problem)
ens_sol = StochasticDiffEq.solve(ens_problem; dt=T, trajectories=1000)

struct FakeSolution{T,U}
    t::Vector{T}
    u::Vector{U}
end

struct FakeProblem{P<:NoiseProblem, F<:FakeSolution}
    base_prob::P         # shared setup (no duplication)
    solutions::Vector{F}  # precomputed or simulated paths
end

using SciMLBase

function simple_solve(r, σ, S₀, T, N)
    ΔT = T / N
    times = collect(0:ΔT:T)
    drift_part = (r - 0.5 * σ^2) * ΔT
    diffusion_part = σ * sqrt(ΔT)
    Z = randn(N)
    log_returns = cumsum(drift_part .+ diffusion_part .* Z)
    log_path = vcat(0.0, log_returns)  # prepend initial log(S₀)
    S = S₀ .* exp.(log_path)

    return FakeSolution(times, S)
end

function generate_ensemble_solution(r, σ, S₀, T, t₀, Nsteps, Npaths)
    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    noise_problem = NoiseProblem(noise, (t₀, T))
    paths = Vector{FakeSolution{Float64, Float64}}(undef, Npaths)
    for i in 1:Npaths
        paths[i] = simple_solve(r, σ, S₀, T, Nsteps)
    end
    return SciMLBase.EnsembleSolution{FakeSolution{Float64, Float64}, 1, FakeProblem}(
        FakeProblem(noise_problem, paths), 1.0, true, nothing)
    end

Base.length(v::FakeProblem) = Base.length(v.solutions)
Base.getindex(fp::FakeProblem, i::Int) = fp.solutions[i]

@code_warntype generate_ensemble_solution(r, σ, S₀, T,t₀, 1, 10)
