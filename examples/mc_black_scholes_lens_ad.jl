using StochasticDiffEqs
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
        sols[i] = StochasticDiffEqs.solve(pmod, solver; dt = dt)
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
# Param Wrapper & Lens
# --------------------------

struct BSMonteCarloSetup{T}
    S0::T
    K::T
    r::T
    σ::T
    T::T
    N::Int
    dt::T
end

struct EnsembleProblemLens end

function (lens::EnsembleProblemLens)(p::BSMonteCarloSetup)
    tspan = (zero(p.T), p.T)
    build_noise_gbm_ensemble(p.r, p.σ, p.S0, tspan; seeds = 1:p.N)
end

function Accessors.set(p::BSMonteCarloSetup, ::EnsembleProblemLens, new_σ)
    T = typeof(new_σ)
    return BSMonteCarloSetup{T}(T(p.S0), T(p.K), T(p.r), new_σ, T(p.T), p.N, T(p.dt))
end

# --------------------------
# Analytic Black-Scholes
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
# AD Price & Vega via Lens
# --------------------------

function price_from_params(p::BSMonteCarloSetup, lens::EnsembleProblemLens)
    prob = lens(p)
    sol = solve_custom_ensemble(prob; dt = p.dt)
    payoffs = map(s -> max(s.u[end] - p.K, 0.0), sol.solutions)
    return mean(payoffs) * exp(-p.r * p.T)
end

function vega_from_ad(p::BSMonteCarloSetup, lens::EnsembleProblemLens)
    f = σ_val -> price_from_params(Accessors.set(p, lens, σ_val), lens)
    return ForwardDiff.derivative(f, p.σ)
end

# --------------------------
# Run Comparison
# --------------------------

params = BSMonteCarloSetup(100.0, 100.0, 0.01, 0.2, 1.0, 10_000, 1 / 250)
lens = EnsembleProblemLens()

price_mc = price_from_params(params, lens)
vega_ad = vega_from_ad(params, lens)
price_an = bs_call_price(params.S0, params.K, params.r, params.σ, params.T)
vega_an = bs_vega_analytic(params.S0, params.K, params.r, params.σ, params.T)

println("Monte Carlo Price: ", price_mc)
println("Analytic Price:    ", price_an)
println("Rel Error (Price): ", abs(price_mc - price_an) / price_an)

println("Monte Carlo Vega (AD): ", vega_ad)
println("Analytic Vega:         ", vega_an)
println("Rel Error (Vega):      ", abs(vega_ad - vega_an) / vega_an)
