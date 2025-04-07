# ------------------ Price Dynamics ------------------

"""
    abstract type PriceDynamics

Abstract supertype for all models describing the stochastic dynamics of asset prices.
"""
abstract type PriceDynamics end

"""
    struct LognormalDynamics <: PriceDynamics

Price process follows geometric Brownian motion (Black-Scholes model).
"""
struct LognormalDynamics <: PriceDynamics end

"""
    struct HestonDynamics <: PriceDynamics

Stochastic volatility model where the variance follows a CIR process.
"""
struct HestonDynamics <: PriceDynamics end

"""
    abstract type VarianceReductionStrategy

Abstract supertype for variance reduction strategies used in Monte Carlo simulations.
"""
abstract type VarianceReductionStrategy end

"""
    struct NoVarianceReduction <: VarianceReductionStrategy

Standard Monte Carlo simulation without variance reduction.
"""
struct NoVarianceReduction <: VarianceReductionStrategy end

"""
    struct Antithetic <: VarianceReductionStrategy

Antithetic variates method for variance reduction by simulating mirrored noise paths.
"""
struct Antithetic <: VarianceReductionStrategy end

# ------------------ Simulation Strategies ------------------

"""
    struct SimulationConfig{I, S, V<:VarianceReductionStrategy}

Configuration for Monte Carlo simulations.

# Fields
- `trajectories`: Number of Monte Carlo paths.
- `steps`: Number of time steps.
- `variance_reduction`: Strategy to reduce variance (e.g., `Antithetic`).
- `seeds`: RNG seeds used to control simulations.
"""
struct SimulationConfig{I, S, V<:VarianceReductionStrategy}
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{I}
end

"""
    SimulationConfig(trajectories; steps=1, seeds=nothing, variance_reduction=Antithetic())

Constructor for `SimulationConfig`, generating random seeds if not provided.
"""
SimulationConfig(trajectories; steps = 1, seeds = nothing, variance_reduction=Antithetic()) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    SimulationConfig(trajectories, steps, variance_reduction, seeds)
end

"""
    abstract type SimulationStrategy

Abstract supertype for numerical strategies to simulate SDEs.
"""
abstract type SimulationStrategy end

"""
    struct EulerMaruyama <: SimulationStrategy

Standard Euler-Maruyama discretization for SDEs.
"""
struct EulerMaruyama <: SimulationStrategy end

"""
    struct HestonBroadieKaya <: SimulationStrategy

Exact sampling scheme for the Heston model using Broadie-Kaya method.
"""
struct HestonBroadieKaya <: SimulationStrategy end

"""
    struct BlackScholesExact <: SimulationStrategy

Exact solution for geometric Brownian motion.
"""
struct BlackScholesExact <: SimulationStrategy end

# ------------------ SDE Problem Builders ------------------

# (docstrings omitted here but can be added similarly on request)

function sde_problem(
    ::LognormalDynamics,
    ::BlackScholesExact,
    market::BlackScholesInputs,
    tspan,
)
    @assert market.rate isa FlatRateCurve "LognormalDynamics requires flat rate curve"

    r = zero_rate(market.rate, 0.0)
    σ = get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    noise_problem = NoiseProblem(noise, tspan)
    return noise_problem
end

function sde_problem(::LognormalDynamics, ::EulerMaruyama, m::BlackScholesInputs, tspan)
    rate = zero_rate(m.rate, 0.0)
    σ = get_vol(m.sigma, nothing, nothing)
    return LogGBMProblem(rate, σ, log(m.spot), tspan)
end

function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    rate = zero_rate(m.rate, 0.0)
    return HestonProblem(rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

function sde_problem(::HestonDynamics, strategy::HestonBroadieKaya, m::HestonInputs, tspan)
    rate = zero_rate(m.rate, 0.0)
    noise = HestonNoise(
        rate,
        m.κ,
        m.θ,
        m.σ,
        m.ρ,
        0.0,
        [log(m.spot), m.V0],
        Z0 = nothing
    )
    return NoiseProblem(noise, tspan)
end

function get_antithetic_ensemble_problem(
    problem,
    ::PriceDynamics,
    ::EulerMaruyama,
    config::SimulationConfig,
    normal_sol::EnsembleSolution,
    market_inputs,
    tspan
)
    prob_func = function (_base_prob, _seed, i)
        flipped_noise = NoiseGrid(normal_sol.u[i].W.t, -normal_sol.u[i].W.W)
        return remake(_base_prob; noise = flipped_noise)
    end

    return EnsembleProblem(problem, prob_func=prob_func)
end

function get_antithetic_ensemble_problem(
    problem,
    ::LognormalDynamics,
    ::BlackScholesExact,
    config::SimulationConfig,
    normal_sol::EnsembleSolution,
    market::BlackScholesInputs,
    tspan
)
    @assert market.rate isa FlatRateCurve "LognormalDynamics requires flat rate curve"

    r = zero_rate(market.rate, 0.0)
    σ = -get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)

    noise_problem = NoiseProblem(noise, tspan)
    return get_ensemble_problem(noise_problem, s)
end

# ------------------ Marginal Laws (optional) ------------------

function marginal_law(::LognormalDynamics, m::BlackScholesInputs, t)
    rate = zero_rate(m.rate, t)
    α = yearfrac(m.rate.reference_date, t)
    return Normal(log(m.spot) + (rate - m.sigma^2 / 2) * √α, m.sigma * √α)
end

function marginal_law(::HestonDynamics, m::HestonInputs, t)
    α = yearfrac(m.rate.reference_date, t)
    rate = zero_rate(m.rate, t)
    return HestonDistribution(m.spot, m.V0, m.κ, m.θ, m.σ, m.ρ, rate, α)
end

# ------------------ Ensemble Simulation Wrapper ------------------

function get_ensemble_problem(prob, strategy_config::SimulationConfig)
    seeds = strategy_config.seeds
    let seeds = seeds
        prob_func = (prob, i, repeat) -> remake(prob; seed=seeds[i])
        EnsembleProblem(prob, prob_func=prob_func)
    end
end

# ------------------ Terminal Value Extractors ------------------

get_terminal_value(path, ::HestonDynamics, ::HestonBroadieKaya) = exp(last(path)[1])
get_terminal_value(path, ::LognormalDynamics, ::EulerMaruyama) = exp(last(path))
get_terminal_value(path, ::PriceDynamics, ::SimulationStrategy) =
    last(path) isa Number ? last(path) : last(path)[1]

# ------------------ Monte Carlo Method ------------------

"""
    struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy, C<:SimulationConfig}

Monte Carlo pricing method combining price dynamics, simulation strategy, and configuration.

# Fields
- `dynamics`: The model for asset price dynamics (e.g., `HestonDynamics`).
- `strategy`: Numerical discretization method (e.g., `EulerMaruyama`).
- `config`: Simulation parameters (number of paths, steps, variance reduction, etc.).
"""
struct MonteCarlo{P<:PriceDynamics,S<:SimulationStrategy, C<:SimulationConfig} <: AbstractPricingMethod
    dynamics::P
    strategy::S
    config::C
end

# (docstrings for simulate_paths and solve can be added next if needed)

function simulate_paths(
    prob::PricingProblem,
    method::M,
    ::NoVarianceReduction
) where {M<:MonteCarlo}
    strategy = method.strategy
    dynamics = method.dynamics
    config = method.config
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    tspan = (0.0, T)
    dt = T / config.steps

    normal_prob = sde_problem(dynamics, strategy, prob.market_inputs, tspan)
    ensemble_prob = get_ensemble_problem(normal_prob, config)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories)
    return normal_sol
end

function simulate_paths(
    prob::PricingProblem,
    method::M,
    ::Antithetic
) where {M<:MonteCarlo}
    strategy = method.strategy
    dynamics = method.dynamics
    config = method.config
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    tspan = (0.0, T)
    dt = T / config.steps

    normal_prob = sde_problem(dynamics, strategy, prob.market_inputs, tspan)
    ensemble_prob = get_ensemble_problem(normal_prob, config)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories, save_noise=true)

    antithetic_prob = get_antithetic_ensemble_problem(normal_prob, dynamics, strategy, config, normal_sol, prob.market_inputs, tspan)
    antithetic_sol = DifferentialEquations.solve(antithetic_prob, EM(); dt = dt, trajectories=config.trajectories)
    return (normal_sol, antithetic_sol)
end

function reduce_payoffs(
    result::EnsembleSolution,
    payoff::F,
    ::NoVarianceReduction,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    return [payoff(get_terminal_value(p, dynamics, strategy)) for p in result.u]
end

function reduce_payoffs(
    result::Tuple{EnsembleSolution, EnsembleSolution},
    payoff::F,
    ::Antithetic,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    paths₁, paths₂ = result[1].u, result[2].u
    return [
        payoff(get_terminal_value(p1, dynamics, strategy)) +
        payoff(get_terminal_value(p2, dynamics, strategy))
        for (p1, p2) in zip(paths₁, paths₂)
    ]
end

function solve(
    prob::PricingProblem{VanillaOption{TS, TE, European, C, Spot}, I},
    method::MonteCarlo{D, S},
) where {TS, TE, C, I<:AbstractMarketInputs, D<:PriceDynamics, S<:SimulationStrategy}
    strategy = method.strategy
    dynamics = method.dynamics
    config = method.config

    ens = simulate_paths(prob, method, config.variance_reduction)
    payoffs = reduce_payoffs(ens, prob.payoff, config.variance_reduction, dynamics, strategy)
    discount = df(prob.market_inputs.rate, prob.payoff.expiry)
    price = discount * mean(payoffs)

    return MonteCarloSolution(price, ens)
end
