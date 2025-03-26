export MonteCarlo, HestonBroadieKaya, EulerMaruyama, BlackScholesExact, LognormalDynamics, HestonDynamics

"""
    PriceDynamics

Abstract type representing the underlying asset dynamics in a pricing model.
"""
abstract type PriceDynamics end

"""
    LognormalDynamics <: PriceDynamics

Represents standard Black-Scholes lognormal dynamics.
"""
struct LognormalDynamics <: PriceDynamics end

"""
    HestonDynamics <: PriceDynamics

Represents Heston stochastic volatility model dynamics.
"""
struct HestonDynamics <: PriceDynamics end

"""
    SimulationStrategy

Abstract type for simulation strategies used in Monte Carlo pricing.
"""
abstract type SimulationStrategy end

"""
    HestonBroadieKaya <: SimulationStrategy

Simulation strategy for Heston dynamics using Broadie-Kaya exact sampling.

# Fields
- `trajectories`: Number of Monte Carlo paths.
- `steps`: Number of time steps (should be 1 for exact method).
- `kwargs`: Named tuple of optional arguments passed to noise process or solver.
"""
struct HestonBroadieKaya <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
end

"""
    HestonBroadieKaya(trajectories; kwargs...)

Convenience constructor for `HestonBroadieKaya` with default `steps = 1`.
"""
function HestonBroadieKaya(trajectories; kwargs...)
    return HestonBroadieKaya(trajectories, 1, (; kwargs...))
end

"""
    EulerMaruyama <: SimulationStrategy

Simulation strategy using Euler-Maruyama discretization.

# Fields
- `trajectories`: Number of Monte Carlo paths.
- `steps`: Number of time steps.
- `kwargs`: Named tuple of optional solver arguments.
"""
struct EulerMaruyama <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
end

"""
    BlackScholesExact <: SimulationStrategy

Exact simulation strategy for lognormal dynamics (Black-Scholes).

# Fields
- `trajectories`: Number of Monte Carlo paths.
- `steps`: Number of time steps (usually 1).
- `kwargs`: Named tuple of optional solver arguments.
"""
struct BlackScholesExact <: SimulationStrategy
    trajectories
    steps
    kwargs::NamedTuple
end

"""
    EulerMaruyama(trajectories, steps; kwargs...)

Convenience constructor for `EulerMaruyama` simulation strategy.
"""
function EulerMaruyama(trajectories, steps; kwargs...) 
    return EulerMaruyama(trajectories, steps, (; kwargs...))
end

"""
    BlackScholesExact(trajectories, steps=1; kwargs...)

Convenience constructor for `BlackScholesExact` simulation strategy.
"""
function BlackScholesExact(trajectories, steps=1; kwargs...)
    return BlackScholesExact(trajectories, steps, (; kwargs...))
end

"""
    sde_problem(::LognormalDynamics, s::BlackScholesExact, market_inputs, tspan)

Constructs a `NoiseProblem` for exact simulation of Black-Scholes model.
"""
function sde_problem(::LognormalDynamics, s::BlackScholesExact, market_inputs::BlackScholesInputs, tspan)
    noise = GeometricBrownianMotionProcess(market_inputs.rate, market_inputs.sigma, 0.0, market_inputs.spot)
    kwargs = s.kwargs
    return NoiseProblem(noise, tspan; kwargs...)
end

"""
    marginal_law(::LognormalDynamics, market_inputs, t)

Returns the lognormal distribution of the asset price at time `t`.
"""
function marginal_law(::LognormalDynamics, m::BlackScholesInputs, t)
    return Normal(log(m.spot) + (m.rate - m.sigma^2 / 2)t, m.sigma * √t)  
end

"""
    sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)

Constructs an `SDEProblem` for Heston dynamics using Euler-Maruyama.
"""
function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    return HestonProblem(m.rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

"""
    marginal_law(::HestonDynamics, market_inputs, t)

Returns the joint distribution of log(S_T) and V_T under the Heston model.
"""
function marginal_law(::HestonDynamics, m::HestonInputs, t)
    return HestonDistribution(m.spot, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)
end

"""
    sde_problem(::HestonDynamics, strategy::HestonBroadieKaya, m::HestonInputs, tspan)

Constructs a `NoiseProblem` for Heston dynamics using Broadie-Kaya sampling.
"""
function sde_problem(::HestonDynamics, strategy::HestonBroadieKaya, m::HestonInputs, tspan)
    noise = HestonNoise(m.rate, m.κ, m.θ, m.σ, m.ρ, 0.0, [log(m.spot), m.V0], Z0=nothing; strategy.kwargs...)
    return NoiseProblem(noise, tspan)
end

"""
    montecarlo_solution(problem, strategy)

Solves the SDE using ensemble simulation with the specified strategy.
"""
function montecarlo_solution(problem, strategy::S) where {S <: SimulationStrategy}
    return solve(
        EnsembleProblem(problem),
        EM();
        dt = problem.tspan[end] / strategy.steps,
        trajectories = strategy.trajectories
    )
end

"""
    MonteCarlo{P, S} <: AbstractPricingMethod

Pricing method for European options using Monte Carlo simulation.

# Fields
- `dynamics`: Price dynamics (e.g., Black-Scholes or Heston).
- `strategy`: Simulation strategy.
"""
struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy} <: AbstractPricingMethod
    dynamics::P
    strategy::S
end

"""
    get_terminal_value(path, ::HestonDynamics, ::HestonBroadieKaya)

Returns the terminal asset price from a path simulated with Broadie-Kaya.
"""
function get_terminal_value(path, ::HestonDynamics, strategy::HestonBroadieKaya)
    return exp(last(path)[1])
end

"""
    get_terminal_value(path, ::D, ::K)

Returns the terminal value for generic dynamics and strategy.
"""
function get_terminal_value(path, ::D, strategy::K) where {D <:PriceDynamics, K<:SimulationStrategy}
    return last(path) isa Number ? last(path) : last(path)[1]
end

"""
    simulate_paths(method::MonteCarlo, market_inputs, T)

Simulates asset paths over time horizon `T` using the specified Monte Carlo method.
"""
function simulate_paths(method::MonteCarlo, market_inputs::I, T) where {I <: AbstractMarketInputs}
    return montecarlo_solution(
        sde_problem(method.dynamics, method.strategy, market_inputs, (0.0, T)),
        method.strategy
    )
end

"""
    compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs, method::MonteCarlo) -> Float64

Prices a European vanilla option using Monte Carlo simulation.

# Arguments
- `payoff`: European-style vanilla option.
- `market_inputs`: Market data (spot, rate, reference date).
- `method`: Monte Carlo pricing method with defined dynamics and strategy.

# Returns
- Discounted expected payoff from simulated terminal values.
"""
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::MonteCarlo) where {C, I <: AbstractMarketInputs}
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    paths = simulate_paths(method, market_inputs, T).u
    prices = [get_terminal_value(p, method.dynamics, method.strategy) for p in paths]
    payoffs = payoff.(prices)
    return exp(-market_inputs.rate * T) * mean(payoffs)
end
