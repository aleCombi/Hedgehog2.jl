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
struct SimulationConfig{I, S, V<:VarianceReductionStrategy,TSeeds}
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{TSeeds}
end

"""
    SimulationConfig(trajectories; steps=1, seeds=nothing, variance_reduction=Antithetic())

Constructor for `SimulationConfig`, generating random seeds if not provided.
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

# ------------------ SDE Problem Builders ------------------

"""
    sde_problem(::LognormalDynamics, ::BlackScholesExact, market, tspan)

Constructs a `NoiseProblem` for Black-Scholes dynamics with exact solution.
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
    sde_problem(::LognormalDynamics, ::EulerMaruyama, market, tspan)

Constructs a `LogGBMProblem` using Euler-Maruyama for Black-Scholes dynamics.
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
    sde_problem(::HestonDynamics, ::EulerMaruyama, market, tspan)

Constructs a `HestonProblem` using Euler-Maruyama for Heston dynamics.
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
    sde_problem(::HestonDynamics, ::HestonBroadieKaya, market, tspan)

Constructs a `NoiseProblem` for Heston model using the Broadie-Kaya method.
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

function sde_problem(
    problem::P,
    method::M
    ) where {P <: PricingProblem, M <: MonteCarlo}
    return sde_problem(problem, method.dynamics, method.strategy)
end

"""
    get_antithetic_ensemble_problem(...)

Builds an antithetic ensemble problem by mirroring noise paths from a base solution.
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
    get_antithetic_ensemble_problem(...)

Constructs an antithetic ensemble problem for exact Black-Scholes dynamics.
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

# ------------------ Marginal Laws (optional) ------------------

"""
    marginal_law(::LognormalDynamics, market, t)

Returns the marginal distribution of log-price under Black-Scholes dynamics.
"""
function marginal_law(::LognormalDynamics, m::BlackScholesInputs, t)
    rate = zero_rate(m.rate, t)
    σ = get_vol(m.sigma, nothing, nothing)
    α = yearfrac(m.rate.reference_date, t)
    return Normal(log(m.spot) + (rate - σ^2 / 2) * √α, σ * √α)
end

"""
    marginal_law(::HestonDynamics, market, t)

Returns the marginal distribution of price under Heston dynamics.
"""
function marginal_law(::HestonDynamics, m::HestonInputs, t)
    α = yearfrac(m.rate.reference_date, t)
    rate = zero_rate(m.rate, t)
    return HestonDistribution(m.spot, m.V0, m.κ, m.θ, m.σ, m.ρ, rate, α)
end

# ------------------ Ensemble Simulation Wrapper ------------------

"""
    get_ensemble_problem(prob, config)

Constructs an `EnsembleProblem` with fixed seeds for reproducible simulations.
"""
function get_ensemble_problem(prob, strategy_config::SimulationConfig)
    seeds = strategy_config.seeds
    prob_func = (prob, i, repeat) -> remake(prob; seed=seeds[i])
    EnsembleProblem(prob, prob_func=prob_func)
end

# ------------------ Terminal Value Extractors ------------------

"""
    get_terminal_value(path, dynamics, strategy)

Extracts the terminal asset value from a simulated path.
"""
get_terminal_value(path, ::HestonDynamics, ::HestonBroadieKaya) = exp(last(path)[1])
get_terminal_value(path, ::LognormalDynamics, ::EulerMaruyama) = exp(last(path))
get_terminal_value(path, ::PriceDynamics, ::SimulationStrategy) =
    last(path) isa Number ? last(path) : last(path)[1]

# ------------------ Monte Carlo Method ------------------

function simulate_paths(
    sde_prob,    
    method::M,
    ::NoVarianceReduction
) where { M<:MonteCarlo}

    config = method.config
    dt = sde_prob.tspan[2] / config.steps
    ensemble_prob = get_ensemble_problem(sde_prob, config)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories)
    return normal_sol
end


function simulate_paths(
    sde_prob,    
    method::M,
    ::Antithetic
    ) where {M<:MonteCarlo}

    config = method.config
    dt = sde_prob.tspan[2] / config.steps

    ensemble_prob = get_ensemble_problem(sde_prob, config)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=config.trajectories, save_noise=true)

    antithetic_prob = get_antithetic_ensemble_problem(sde_prob, normal_sol, method)
    antithetic_sol = DifferentialEquations.solve(antithetic_prob, EM(); dt = dt, trajectories=config.trajectories)
    return (normal_sol, antithetic_sol)
end

"""
    reduce_payoffs(result, payoff, ::NoVarianceReduction, dynamics, strategy)

Computes payoffs from simulated terminal values without variance reduction.
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

Computes payoffs from paired antithetic simulations and averages them.
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
    solve(prob, method)

Solves a `PricingProblem` using the specified Monte Carlo method.
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
