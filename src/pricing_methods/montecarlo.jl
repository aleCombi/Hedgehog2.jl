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

Stochastic volatility model where the variance follows a Cox-Ingersoll-Ross (CIR) process.
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
    struct SimulationConfig{I, S, V<:VarianceReductionStrategy, TSeeds}

Configuration for Monte Carlo simulations.

# Fields
- `trajectories`: Number of Monte Carlo paths.
- `steps`: Number of time steps in each simulation.
- `variance_reduction`: Strategy to reduce variance (e.g., `Antithetic()`).
- `seeds`: Vector of RNG seeds used to initialize simulations for reproducibility.
"""
struct SimulationConfig{I, S, V<:VarianceReductionStrategy, TSeeds}
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{TSeeds}
end

"""
    SimulationConfig(trajectories; steps=1, seeds=nothing, variance_reduction=Antithetic())

Constructor for `SimulationConfig`. If `seeds` is not provided, random seeds are generated.
"""
SimulationConfig(trajectories; steps = 1, seeds = nothing, variance_reduction=Antithetic()) = begin
    seeds === nothing && (seeds = Base.rand(UInt64, trajectories))
    SimulationConfig(trajectories, steps, variance_reduction, seeds)
end

"""
    abstract type SimulationStrategy

Abstract supertype for numerical strategies to simulate SDEs.
"""
abstract type SimulationStrategy end

"""
    struct EulerMaruyama <: SimulationStrategy

Euler-Maruyama discretization for simulating SDEs.
"""
struct EulerMaruyama <: SimulationStrategy end

"""
    struct HestonBroadieKaya <: SimulationStrategy

Exact sampling scheme for the Heston model using the Broadie-Kaya method.
"""
struct HestonBroadieKaya <: SimulationStrategy end

"""
    struct BlackScholesExact <: SimulationStrategy

Exact sampling of the Black-Scholes model using closed-form solution.
"""
struct BlackScholesExact <: SimulationStrategy end

"""
    struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy, C<:SimulationConfig}

Monte Carlo pricing method combining price dynamics, simulation strategy, and configuration.

# Fields
- `dynamics`: The model for asset price dynamics.
- `strategy`: Numerical simulation method.
- `config`: Simulation configuration (paths, steps, seeds, etc.).
"""
struct MonteCarlo{P<:PriceDynamics, S<:SimulationStrategy, C<:SimulationConfig} <: AbstractPricingMethod
    dynamics::P
    strategy::S
    config::C
end

# ------------------ SDE Problem Builders ------------------

"""
    sde_problem(problem::PricingProblem, ::LognormalDynamics, ::BlackScholesExact)

Constructs a `NoiseProblem` for exact simulation of Black-Scholes dynamics.
"""
function sde_problem(
    problem::PricingProblem{Payoff, Inputs},
    ::LognormalDynamics,
    ::BlackScholesExact,
) where {Payoff, Inputs <: BlackScholesInputs}

    market = problem.market_inputs
    T = yearfrac(market.referenceDate, problem.payoff.expiry)
    tspan = (0.0, T)

    r = zero_rate(market.rate, 0.0)
    σ = get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)

    r, σ, t₀, S₀ = promote(r, σ, t₀, S₀)
    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    noise_problem = NoiseProblem(noise, tspan)
    return noise_problem
end

"""
    sde_problem(problem::PricingProblem, ::LognormalDynamics, ::EulerMaruyama)

Constructs a `LogGBMProblem` for Euler-Maruyama simulation of Black-Scholes dynamics.
"""
function sde_problem(
    problem::PricingProblem{Payoff, Inputs},
    ::LognormalDynamics,
    ::EulerMaruyama,
    ) where {Payoff, Inputs <: BlackScholesInputs}

    m = problem.market_inputs
    T = yearfrac(m.referenceDate, problem.payoff.expiry)
    tspan = (0.0, T)

    rate = zero_rate(m.rate, 0.0)
    σ = get_vol(m.sigma, nothing, nothing)
    S₀ = m.spot

    rate, σ, S₀ = promote(rate, σ, S₀)

    return LogGBMProblem(rate, σ, log(S₀), tspan)
end

"""
    sde_problem(problem::PricingProblem, ::HestonDynamics, ::EulerMaruyama)

Constructs a `HestonProblem` for Euler-Maruyama simulation of the Heston model.
"""
function sde_problem(
    problem::PricingProblem{Payoff, Inputs},
    ::HestonDynamics,
    ::EulerMaruyama,
    ) where {Payoff, Inputs <: HestonInputs}

    m = problem.market_inputs
    T = yearfrac(m.referenceDate, problem.payoff.expiry)
    tspan = (0.0, T)

    rate = zero_rate(m.rate, 0.0)
    return HestonProblem(rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

"""
    sde_problem(problem::PricingProblem, ::HestonDynamics, ::HestonBroadieKaya)

Constructs a `NoiseProblem` using Broadie-Kaya sampling for Heston dynamics.
"""
function sde_problem(
    problem::PricingProblem{Payoff, Inputs},
    ::HestonDynamics,
    strategy::HestonBroadieKaya,
    ) where {Payoff, Inputs <: HestonInputs}

    m = problem.market_inputs
    T = yearfrac(m.referenceDate, problem.payoff.expiry)
    tspan = (0.0, T)

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

"""
    sde_problem(problem, method)

Dispatches to the appropriate SDE constructor based on the method's dynamics and strategy.
"""
function sde_problem(
    problem::P,
    method::M
    ) where {P <: PricingProblem, M <: MonteCarlo}
    return sde_problem(problem, method.dynamics, method.strategy)
end

# ------------------ Antithetic Path Constructors ------------------

"""
    get_antithetic_ensemble_problem(problem, normal_sol, method)

Constructs an ensemble problem by mirroring noise from a base solution for Euler-Maruyama.
"""
function get_antithetic_ensemble_problem(
    problem,
    normal_sol::EnsembleSolution,
    method::MonteCarlo{P, EulerMaruyama, S}
) where {P<:PriceDynamics, S <: SimulationConfig}
    prob_func = function (prob, i, repeat)
        flipped_noise = NoiseGrid(normal_sol.u[i].W.t, -normal_sol.u[i].W.W)
        return remake(prob; noise = flipped_noise, seed=method.config.seeds[i])
    end

    return EnsembleProblem(problem, prob_func=prob_func)
end

"""
    get_antithetic_ensemble_problem(problem, normal_sol, method)

Constructs an ensemble problem by flipping volatility for exact Black-Scholes simulation.
"""
function get_antithetic_ensemble_problem(
    problem,
    normal_sol::EnsembleSolution,
    method::MonteCarlo{LognormalDynamics, BlackScholesExact, S}
    ) where {S <: SimulationConfig}
    r = problem.noise.dist.μ
    σ = -problem.noise.dist.σ
    S₀ = problem.noise.W[1]
    t₀ = problem.noise.t[1]

    r, σ, t₀, S₀ = promote(r, σ, t₀, S₀)
    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    noise_problem = NoiseProblem(noise, problem.tspan)
    return get_ensemble_problem(noise_problem, method.config)
end

# ------------------ Marginal Laws ------------------

"""
    marginal_law(problem::PricingProblem, ::LognormalDynamics, t)

Returns the marginal distribution of log-price under Black-Scholes dynamics at time `t`.
"""
function marginal_law(
    problem::PricingProblem{P, I},
    ::LognormalDynamics,
    t
) where {P, I <: BlackScholesInputs}
    m = problem.market_inputs
    rate = zero_rate(m.rate, t)
    σ = get_vol(m.sigma, nothing, nothing)
    α = yearfrac(m.rate.reference_date, t)
    return Normal(log(m.spot) + (rate - σ^2 / 2) * √α, σ * √α)
end

"""
    marginal_law(problem::PricingProblem, ::HestonDynamics, t)

Returns the marginal distribution of log-price under Heston dynamics at time `t`.
"""
function marginal_law(
    problem::PricingProblem{P, I},
    ::HestonDynamics,
    t
) where {P, I <: HestonInputs}

    m = problem.market_inputs
    α = yearfrac(m.rate.reference_date, t)
    rate = zero_rate(m.rate, t)
    return HestonDistribution(m.spot, m.V0, m.κ, m.θ, m.σ, m.ρ, rate, α)
end

# ------------------ Ensemble Simulation Wrapper ------------------

"""
    get_ensemble_problem(prob, config)

Wraps a simulation problem into an `EnsembleProblem` with seed control.
"""
function get_ensemble_problem(prob, strategy_config::SimulationConfig)
    seeds = strategy_config.seeds
    prob_func = (prob, i, repeat) -> remake(prob; seed=seeds[i])
    EnsembleProblem(prob, prob_func=prob_func)
end

# ------------------ Terminal Value Extractors ------------------

"""
    get_terminal_value(path, dynamics, strategy)

Extracts the terminal asset value from a simulated path, depending on model and strategy.
"""
get_terminal_value(path, ::HestonDynamics, ::HestonBroadieKaya) = exp(last(path)[1])
get_terminal_value(path, ::LognormalDynamics, ::EulerMaruyama) = exp(last(path))
get_terminal_value(path, ::PriceDynamics, ::SimulationStrategy) =
    last(path) isa Number ? last(path) : last(path)[1]

# ------------------ Monte Carlo Method ------------------

"""
    simulate_paths(sde_prob, method, ::NoVarianceReduction)

Simulates paths using standard Monte Carlo (no variance reduction).
"""
function simulate_paths(
    sde_prob,
    method::M,
    ::NoVarianceReduction
) where { M<:MonteCarlo}

    config = method.config
    dt = sde_prob.tspan[2] / config.steps
    ensemble_prob = get_ensemble_problem(sde_prob, config)
    normal_sol = StochasticDiffEq.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories)
    return normal_sol
end

"""
    simulate_paths(sde_prob, method, ::Antithetic)

Simulates paths using antithetic variates for variance reduction.
"""
function simulate_paths(
    sde_prob,
    method::M,
    ::Antithetic
    ) where {M<:MonteCarlo}

    config = method.config
    dt = sde_prob.tspan[2] / config.steps

    ensemble_prob = get_ensemble_problem(sde_prob, config)
    normal_sol = StochasticDiffEq.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories, save_noise=true)

    antithetic_prob = get_antithetic_ensemble_problem(sde_prob, normal_sol, method)
    antithetic_sol = StochasticDiffEq.solve(antithetic_prob, EM(); dt = dt, trajectories=config.trajectories)
    return (normal_sol, antithetic_sol)
end

"""
    reduce_payoffs(result, payoff, ::NoVarianceReduction, dynamics, strategy)

Computes the payoff from terminal values without applying variance reduction.
"""
function reduce_payoffs(
    result::EnsembleSolution,
    payoff::F,
    ::NoVarianceReduction,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    return [payoff(get_terminal_value(p, dynamics, strategy)) for p in result.u]
end

"""
    reduce_payoffs(result, payoff, ::Antithetic, dynamics, strategy)

Computes the average payoff over paired antithetic paths.
"""
function reduce_payoffs(
    result::Tuple{EnsembleSolution, EnsembleSolution},
    payoff::F,
    ::Antithetic,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    paths₁, paths₂ = result[1].u, result[2].u
    return [
        (payoff(get_terminal_value(p1, dynamics, strategy)) +
        payoff(get_terminal_value(p2, dynamics, strategy))) / 2
        for (p1, p2) in zip(paths₁, paths₂)
    ]
end

"""
    solve(prob::PricingProblem, method::MonteCarlo)

Solves the pricing problem using Monte Carlo simulation and returns the discounted expected payoff.
"""
function solve(
    prob::PricingProblem{VanillaOption{TS, TE, European, C, Spot}, I},
    method::MonteCarlo{D, S},
) where {TS, TE, C, I<:AbstractMarketInputs, D<:PriceDynamics, S<:SimulationStrategy}
    strategy = method.strategy
    dynamics = method.dynamics
    config = method.config

    sde_prob = sde_problem(prob, method.dynamics, method.strategy)
    ens = simulate_paths(sde_prob, method, config.variance_reduction)
    payoffs = reduce_payoffs(ens, prob.payoff, config.variance_reduction, dynamics, strategy)
    discount = df(prob.market_inputs.rate, prob.payoff.expiry)
    price = discount * mean(payoffs)

    return MonteCarloSolution(prob, method, price, ens)
end
