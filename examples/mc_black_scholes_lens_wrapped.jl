using DifferentialEquations
using DiffEqNoiseProcess
using ForwardDiff
using Statistics
using Distributions
using Accessors

# --------------------------
# Custom Ensemble Framework
# --------------------------

struct CustomEnsembleProblem{P,F}
    base_problem::P
    seeds::Vector{Int64}
    modify::F  # (base_problem, seed, index) -> new_problem
end

struct CustomEnsembleSolution{S}
    solutions::Vector{S}
    seeds::Vector{Int64}
end

function solve_custom_ensemble(
    prob::CustomEnsembleProblem;
    dt,
    solver = EM(),
    trajectories = nothing,
)
    N = trajectories === nothing ? length(prob.seeds) : trajectories
    seeds = length(prob.seeds) == N ? prob.seeds : collect(1:N)

    sols = Vector{Any}(undef, N)

    Threads.@threads for i = 1:N
        seed = seeds[i]
        pmod = prob.modify(prob.base_problem, seed, i)
        sols[i] = DifferentialEquations.solve(pmod, solver; dt = dt)
    end

    return CustomEnsembleSolution(sols, seeds)
end

# --------------------------
# GBM with NoiseProcess
# --------------------------

function build_noise_gbm_ensemble(μ, σ, S0, tspan; seeds = nothing)
    T_ = typeof(σ)
    μ = convert(T_, μ)
    S0 = convert(T_, S0)
    t0, Tval = tspan
    tspan = (convert(T_, t0), convert(T_, Tval))

    base_proc = GeometricBrownianMotionProcess(μ, σ, tspan[1], S0)
    base_prob = NoiseProblem(base_proc, tspan)

    N = seeds === nothing ? 100 : length(seeds)
    seeds = seeds === nothing ? collect(1:N) : seeds
    modify = (prob, seed, i) -> remake(prob; seed = seed)
    return CustomEnsembleProblem(base_prob, collect(seeds), modify)
end

# --------------------------
# Monte Carlo Param Wrapper & Lens
# --------------------------

struct BSMonteCarloSetup
    S0::Float64
    K::Float64
    r::Float64
    σ::Float64
    T::Float64
    N::Int
    dt::Float64
end

struct EnsembleProblemLens end

function (lens::EnsembleProblemLens)(p::BSMonteCarloSetup)
    tspan = (0.0, p.T)
    build_noise_gbm_ensemble(p.r, p.σ, p.S0, tspan; seeds = 1:p.N)
end

function Accessors.set(p::BSMonteCarloSetup, ::EnsembleProblemLens, new_σ)
    BSMonteCarloSetup(p.S0, p.K, p.r, new_σ, p.T, p.N, p.dt)
end


# --------------------------
# Analytic Black-Scholes Price & Vega
# --------------------------

function bs_call_price(S, K, r, σ, T)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end

function bs_vega_analytic(S, K, r, σ, T)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return S * sqrt(T) * pdf(Normal(), d1)
end

# --------------------------
# Usage Example with Lens
# --------------------------

params = BSMonteCarloSetup(100.0, 100.0, 0.01, 0.2, 1.0, 10_000, 1 / 250)
lens = EnsembleProblemLens()

# Get base ensemble problem
base_prob = lens(params)
sol = solve_custom_ensemble(base_prob; dt = params.dt)

# Get bumped ensemble by setting σ
params_bumped = Accessors.set(params, lens, 0.25)
bumped_prob = lens(params_bumped)
sol_bumped = solve_custom_ensemble(bumped_prob; dt = params_bumped.dt)

# Extract payoffs and compute price and vega
payoff = x -> max(x - params.K, 0.0)
payoffs_base = payoff.(map(s -> s.u[end], sol.solutions))
payoffs_bump = payoff.(map(s -> s.u[end], sol_bumped.solutions))

df = exp(-params.r * params.T)
price = mean(payoffs_base) * df
vega = mean(payoffs_bump .- payoffs_base) / (params_bumped.σ - params.σ) * df

# Compare to analytic
price_an = bs_call_price(params.S0, params.K, params.r, params.σ, params.T)
vega_an = bs_vega_analytic(params.S0, params.K, params.r, params.σ, params.T)

println("Monte Carlo Price: ", price)
println("Analytic Price:    ", price_an)
println("Rel Error (Price): ", abs(price - price_an) / price_an)

println("Monte Carlo Vega:  ", vega)
println("Analytic Vega:     ", vega_an)
println("Rel Error (Vega):  ", abs(vega - vega_an) / vega_an)
