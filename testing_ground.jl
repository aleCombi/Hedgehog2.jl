using Revise
using Hedgehog
using Dates
using Printf
using BenchmarkTools
using Random 
struct BSExactProblem{T,U,V,W,X}
    r::T
    σ::U
    t₀::V
    S₀::W
    T::X
end

struct BSExactProblemSimulated{T}
    S::T
end

function sde_problem2(
    problem::PricingProblem{Payoff, Inputs},
    ::LognormalDynamics,
    ::BlackScholesExact,
    ) where {Payoff, Inputs <: BlackScholesInputs}
    market = problem.market_inputs
    T = yearfrac(market.referenceDate, problem.payoff.expiry)

    r = zero_rate(market.rate, 0.0)
    σ = get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)

    r, σ, t₀, S₀ = promote(r, σ, t₀, S₀)

    return BSExactProblem(r, σ, t₀, S₀, T)
end

function simulate_paths(
    sde_prob::BSExactProblem,
    method::MonteCarlo,
    ::Hedgehog.NoVarianceReduction
)
    Δt = sde_prob.T - sde_prob.t₀
    σ = sde_prob.σ
    S₀ = sde_prob.S₀
    r = sde_prob.r
    N = method.config.trajectories

    drift_part = ((r - 0.5 * σ^2) * Δt)
    diffusion_part = σ * sqrt(Δt)
    S_out = S₀ .* exp.(drift_part .+ diffusion_part .* randn(N))
    return BSExactProblemSimulated(S_out)
end

function simulate_paths(
    sde_prob::BSExactProblem,
    method::MonteCarlo,
    ::Hedgehog.Antithetic
)
    Δt = sde_prob.T - sde_prob.t₀
    σ = sde_prob.σ
    S₀ = sde_prob.S₀
    r = sde_prob.r
    N = method.config.trajectories

    drift_part = ((r - 0.5 * σ^2) * Δt)
    diffusion_part = σ * sqrt(Δt)

    # We ensure that normal_sol is of the right type
    normal_sol = Vector{typeof(drift_part*diffusion_part)}(undef, N)
    randn!(normal_sol)

    antithetic_sol = S₀ .* exp.(drift_part .- diffusion_part .* normal_sol)
    normal_sol .= S₀ .* exp.(drift_part .+ diffusion_part .* normal_sol)
    return (BSExactProblemSimulated(normal_sol), BSExactProblemSimulated(antithetic_sol))
end

function reduce_payoffs(
    result::Tuple{BSExactProblemSimulated, BSExactProblemSimulated},
    payoff::F,
    ::Hedgehog.Antithetic,
    dynamics::Hedgehog.PriceDynamics,
    strategy::Hedgehog.SimulationStrategy,
) where {F}
    S1, S2 = result
    return (payoff(S1.S) + payoff(S2.S))/2
end

function solve2(
    prob::PricingProblem{VanillaOption{TS, TE, European, C, Spot}, I},
    method::MonteCarlo{D, S},
) where {TS, TE, C, I, D, S}
    strategy = method.strategy
    dynamics = method.dynamics
    config = method.config

    sde_prob = sde_problem2(prob, method.dynamics, method.strategy)
    ens = simulate_paths(sde_prob, method, config.variance_reduction)
    payoffs = reduce_payoffs(ens, prob.payoff, config.variance_reduction, dynamics, strategy)
    discount = df(prob.market_inputs.rate, prob.payoff.expiry)
    price = discount * mean(payoffs)

    return Hedgehog.MonteCarloSolution(prob, method, price, ens)
end

function test()
    spot = 100.0
    strike = 100.0
    rate = 0.05
    sigma = 0.20
    reference_date = Date(2023, 1, 1)
    expiry = reference_date + Year(1)

    # Create the payoff (European call option)
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

    # Create market inputs
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # Create pricing problem
    prob = PricingProblem(payoff, market_inputs)

    trajectories = 5_000
    mc_exact_method =
        MonteCarlo(LognormalDynamics(), BlackScholesExact(), SimulationConfig(trajectories))
    mc_exact_solution = solve(prob, mc_exact_method)
    display(mc_exact_solution.price)
    mc_exact_solution = solve2(prob, mc_exact_method)
    display(mc_exact_solution.price)
    display(@benchmark solve($prob, $mc_exact_method))
    display(@benchmark solve2($prob, $mc_exact_method))
end

test()