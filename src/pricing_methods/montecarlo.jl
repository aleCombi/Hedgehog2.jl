using DifferentialEquations
using DiffEqNoiseProcess
using Statistics
using Accessors

export MonteCarlo, HestonBroadieKaya, EulerMaruyama, BlackScholesExact, LognormalDynamics, HestonDynamics, solve

# ------------------ Price Dynamics ------------------

abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end
struct HestonDynamics <: PriceDynamics end

# ------------------ Simulation Strategies ------------------

abstract type SimulationStrategy end

struct HestonBroadieKaya <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
end

HestonBroadieKaya(trajectories; kwargs...) = HestonBroadieKaya(trajectories, 1, (; kwargs...))

struct EulerMaruyama <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
end

struct BlackScholesExact <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
    seeds::Union{Nothing, Vector{Int}}
end

EulerMaruyama(trajectories, steps; kwargs...) = EulerMaruyama(trajectories, steps, (; kwargs...))
BlackScholesExact(trajectories, steps=1; seeds=nothing, kwargs...) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    BlackScholesExact(trajectories, steps, (; kwargs...), seeds)
end

# ------------------ SDE Problem Builders ------------------

function sde_problem(::LognormalDynamics, s::BlackScholesExact, market::BlackScholesInputs, tspan)
    @assert is_flat(market.rate) "LognormalDynamics requires flat rate curve"

    # Promote all parameters to a common type (Dual or Float64)
    T = promote_type(typeof(zero_rate(market.rate, 0.0)), typeof(market.sigma), typeof(market.spot))

    r   = convert(T, zero_rate(market.rate, 0.0))
    σ   = convert(T, market.sigma)
    S₀  = convert(T, market.spot)
    t₀  = zero(T)

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    return NoiseProblem(noise, tspan; s.kwargs...)
end

function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    @assert is_flat(m.rate) "Heston simulation requires flat rate curve"
    rate = zero_rate(m.rate, 0.0)
    return HestonProblem(rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

function sde_problem(::HestonDynamics, strategy::HestonBroadieKaya, m::HestonInputs, tspan)
    @assert is_flat(m.rate) "Heston simulation requires flat rate curve"
    rate = zero_rate(m.rate, 0.0)
    noise = HestonNoise(rate, m.κ, m.θ, m.σ, m.ρ, 0.0, [log(m.spot), m.V0], Z0=nothing; strategy.kwargs...)
    return NoiseProblem(noise, tspan)
end

# ------------------ Marginal Laws (optional) ------------------

function marginal_law(::LognormalDynamics, m::BlackScholesInputs, t)
    rate = zero_rate(m.rate, t)
    α = yearfrac(m.rate.reference_date, t)
    return Normal(log(m.spot) + (rate - m.sigma^2 / 2) * √α, m.sigma * √α)
end

function marginal_law(::HestonDynamics, m::HestonInputs, t)
    α = yearfrac(m.rate.reference_date, t)
    return HestonDistribution(m.spot, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, α)
end

# ------------------ Ensemble Simulation Wrapper ------------------

function montecarlo_solution(problem::Union{NoiseProblem, SDEProblem}, strategy::S) where {S <: SimulationStrategy}
    dt = problem.tspan[end] / strategy.steps
    N = strategy.trajectories

    seeds = strategy isa BlackScholesExact && strategy.seeds !== nothing ? strategy.seeds :
            Base.rand(1_000_000_000:2_000_000_000, N)

    modify = (p, seed, _) -> remake(p; seed=seed)
    custom_prob = CustomEnsembleProblem(problem, collect(seeds), modify)
    return solve_custom_ensemble(custom_prob; dt=dt)
end

# ------------------ Terminal Value Extractors ------------------

get_terminal_value(path, ::HestonDynamics, ::HestonBroadieKaya) = exp(last(path)[1])
get_terminal_value(path, ::PriceDynamics, ::SimulationStrategy) = last(path) isa Number ? last(path) : last(path)[1]

# ------------------ Monte Carlo Method ------------------

struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy} <: AbstractPricingMethod
    dynamics::P
    strategy::S
end

function simulate_paths(method::MonteCarlo, market_inputs::I, T) where {I <: AbstractMarketInputs}
    return montecarlo_solution(
        sde_problem(method.dynamics, method.strategy, market_inputs, (0.0, T)),
        method.strategy
    )
end

function solve(prob::PricingProblem{VanillaOption{European, C, Spot}, I}, method::MonteCarlo) where {C, I <: AbstractMarketInputs}
    T = yearfrac(prob.market.referenceDate, prob.payoff.expiry)
    ens = simulate_paths(method, prob.market, T)
    paths = ens.solutions

    terminal_prices = [get_terminal_value(p, method.dynamics, method.strategy) for p in paths]
    payoffs = prob.payoff.(terminal_prices)
    price = df(prob.market.rate, prob.payoff.expiry) * mean(payoffs)

    return MonteCarloSolution(price, ens)
end
