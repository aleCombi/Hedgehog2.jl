abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end
struct HestonDynamics <: PriceDynamics end

export MonteCarlo, HestonBroadieKaya, EulerMaruyama, BlackScholesExact, LognormalDynamics, HestonDynamics

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
end

EulerMaruyama(trajectories, steps; kwargs...) = EulerMaruyama(trajectories, steps, (; kwargs...))

BlackScholesExact(trajectories, steps=1; kwargs...) = BlackScholesExact(trajectories, steps, (; kwargs...))

function sde_problem(::LognormalDynamics, ::BlackScholesExact, market_inputs::BlackScholesInputs, tspan)
    noise = GeometricBrownianMotionProcess(market_inputs.rate, market_inputs.sigma, 0.0, market_inputs.spot)
    return NoiseProblem(noise, tspan)
end

function marginal_law(::LognormalDynamics, m::BlackScholesInputs, t)
    return Normal(log(m.spot) + (m.rate - m.sigma^2 / 2)t, m.sigma * √t)  
end

function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    return HestonProblem(m.rate, m.κ, m.θ, m.σ, m.ρ, [m.S0, m.V0], tspan)
end

function marginal_law(::HestonDynamics, m::HestonInputs, t)
    return HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)
end

function sde_problem(d::HestonDynamics, strategy::HestonBroadieKaya, m::HestonInputs, tspan)
    marginal_laws = t -> marginal_law(d, m, t)
    noise = HestonNoise(0.0, marginal_laws, Z0=nothing; strategy.kwargs...)
    return NoiseProblem(noise, tspan)
end

function montecarlo_solution(problem, strategy::S) where {S <: SimulationStrategy}
    println(strategy.kwargs)
    kwargs = strategy.kwargs
    return solve(
        EnsembleProblem(problem),
        EM();
        dt = problem.tspan[end] / strategy.steps,
        trajectories = strategy.trajectories,
        kwargs...
    )
end

# any method defining an SDEProblem and requiring a solver from DifferentialEquation.jl
struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy} <: AbstractPricingMethod
    dynamics::P
    strategy::S
end

get_terminal_value(path) = last(path) isa Number ? last(path) : last(path)[1]

function simulate_paths(method::MonteCarlo, market_inputs::I, T) where {I <: AbstractMarketInputs}
    return montecarlo_solution(
        sde_problem(method.dynamics, method.strategy, market_inputs, (0.0, T)),
        method.strategy
    )
end

simulate_terminal_prices(method, inputs, T) =
    get_terminal_value.(simulate_paths(method, inputs, T).u)

function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::MonteCarlo) where {C, I <: AbstractMarketInputs}
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    prices = simulate_terminal_prices(method, market_inputs, T)
    payoffs = payoff.(prices)
    # println("Standard error: ", sqrt(var(payoffs) / length(payoffs)))
    return exp(-market_inputs.rate * T) * mean(payoffs)
end
