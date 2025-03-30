using DifferentialEquations
using DiffEqNoiseProcess
using ForwardDiff
using Statistics
using Distributions

# --------------------------
# Custom Ensemble Framework
# --------------------------

struct CustomEnsembleProblem{P, F}
    base_problem::P
    seeds::Vector{Int64}
    modify::F  # (base_problem, seed, index) -> new_problem
end

struct CustomEnsembleSolution{S}
    solutions::Vector{S}
    seeds::Vector{Int64}
end

function solve_custom_ensemble(prob::CustomEnsembleProblem; dt, solver=EM(), trajectories=nothing)
    N = trajectories === nothing ? length(prob.seeds) : trajectories
    seeds = length(prob.seeds) == N ? prob.seeds : collect(1:N)

    sols = Vector{Any}(undef, N)

    Threads.@threads for i in 1:N
        seed = seeds[i]
        pmod = prob.modify(prob.base_problem, seed, i)
        sols[i] = DifferentialEquations.solve(pmod, solver; dt=dt)
    end

    return CustomEnsembleSolution(sols, seeds)
end

# --------------------------
# GBM with NoiseProcess
# --------------------------

function build_noise_gbm_ensemble(μ, σ, S0, tspan; seeds=nothing)
    T_ = typeof(σ)
    μ = convert(T_, μ)
    S0 = convert(T_, S0)
    t0, Tval = tspan
    tspan = (convert(T_, t0), convert(T_, Tval))

    base_proc = GeometricBrownianMotionProcess(μ, σ, tspan[1], S0)
    base_prob = NoiseProblem(base_proc, tspan)

    N = seeds === nothing ? 100 : length(seeds)
    seeds = seeds === nothing ? collect(1:N) : seeds
    modify = (prob, seed, i) -> remake(prob; seed=seed)
    return CustomEnsembleProblem(base_prob, collect(seeds), modify)
end

# --------------------------
# Monte Carlo Vega (FD)
# --------------------------

function mc_black_scholes_vega(; σ=0.2, S0=100.0, K=100.0, r=0.01, T=1.0, ε=1e-4, N=10_000, dt=1/250)
    tspan = (0.0, T)
    μ = r  # risk-neutral drift

    base_ens = build_noise_gbm_ensemble(μ, σ, S0, tspan; seeds=1:N)
    bump_ens = build_noise_gbm_ensemble(μ, σ + ε, S0, tspan; seeds=1:N)

    sol_base = solve_custom_ensemble(base_ens; dt=dt)
    sol_bump = solve_custom_ensemble(bump_ens; dt=dt)

    payoffs_base = max.(last.(map(s -> s.u, sol_base.solutions)) .- K, 0.0)
    payoffs_bump = max.(last.(map(s -> s.u, sol_bump.solutions)) .- K, 0.0)

    df = exp(-r * T)
    vega_est = mean(payoffs_bump .- payoffs_base) / ε * df

    return vega_est
end

# --------------------------
# Monte Carlo Vega (AD)
# --------------------------

function mc_black_scholes_vega_ad(; σ=0.2, S0=100.0, K=100.0, r=0.01, T=1.0, N=10_000, dt=1/250)
    seeds = 1:N
    payoff = x -> max(x - K, 0.0)
    df = exp(-r * T)

    f = σ_val -> begin
        T_ = typeof(σ_val)
        μ_ = convert(T_, r)  # risk-neutral drift
        S0_ = convert(T_, S0)
        tspan = (convert(T_, 0.0), convert(T_, T))

        base_ens = build_noise_gbm_ensemble(μ_, σ_val, S0_, tspan; seeds=seeds)
        sol = solve_custom_ensemble(base_ens; dt=dt)
        final_vals = map(s -> s.u[end], sol.solutions)
        mean(payoff.(final_vals)) * df
    end

    vega = ForwardDiff.derivative(f, σ)
    return vega
end

# --------------------------
# Analytic Black-Scholes Vega
# --------------------------

function bs_vega_analytic(S, K, r, σ, T)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return S * sqrt(T) * pdf(Normal(), d1)
end

# --------------------------
# Run Comparison
# --------------------------

S0 = 100.0
K = 100.0
r = 0.01
T = 1.0
σ = 0.2
N = 100000

vega_mc = mc_black_scholes_vega(; σ=σ, S0=S0, K=K, r=r, T=T, N=N)
vega_ad = mc_black_scholes_vega_ad(; σ=σ, S0=S0, K=K, r=r, T=T, N=N)
vega_an = bs_vega_analytic(S0, K, r, σ, T)

println("Monte Carlo Vega (FD): ", vega_mc)
println("Monte Carlo Vega (AD): ", vega_ad)
println("Analytic Vega:         ", vega_an)
println("Rel Error (FD):        ", abs(vega_mc - vega_an) / vega_an)
println("Rel Error (AD):        ", abs(vega_ad - vega_an) / vega_an)
