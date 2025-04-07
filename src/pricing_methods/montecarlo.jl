# ------------------ Price Dynamics ------------------

abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end
struct HestonDynamics <: PriceDynamics end

abstract type VarianceReductionStrategy end

struct NoVarianceReduction <: VarianceReductionStrategy end
struct Antithetic <: VarianceReductionStrategy end

# ------------------ Simulation Strategies ------------------

abstract type SimulationStrategy end

struct HestonBroadieKaya{I <: Integer, S <: Integer, V <: VarianceReductionStrategy} <: SimulationStrategy
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{I}
end

struct EulerMaruyama{I <: Integer, S <: Integer, V <: VarianceReductionStrategy} <: SimulationStrategy
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{I}
end

struct BlackScholesExact{I <: Integer, S <: Integer, V <: VarianceReductionStrategy} <: SimulationStrategy
    trajectories::I
    steps::S
    variance_reduction::V
    seeds::Vector{I}
end

HestonBroadieKaya(trajectories; steps = 1, seeds = nothing, variance_reduction=Antithetic()               # just antithetic
) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    HestonBroadieKaya(trajectories, steps, variance_reduction, seeds)
end
EulerMaruyama(trajectories; steps = 1, seeds = nothing, variance_reduction=Antithetic()               # just antithetic
) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    EulerMaruyama(trajectories, steps, variance_reduction, seeds)
end

BlackScholesExact(trajectories; steps = 1, seeds = nothing, variance_reduction=Antithetic()               # just antithetic
) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    BlackScholesExact(trajectories, steps, variance_reduction, seeds)
end

# ------------------ SDE Problem Builders ------------------

function sde_problem(
    ::LognormalDynamics,
    s::BlackScholesExact,
    market::BlackScholesInputs,
    tspan,
)
    @assert market.rate isa FlatRateCurve "LognormalDynamics requires flat rate curve"

    r = zero_rate(market.rate, 0.0)
    σ = get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)  # ensure same type as S₀

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    noise_problem = NoiseProblem(noise, tspan)
    return noise_problem
end


function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    rate = zero_rate(m.rate, 0.0)
    return HestonProblem(rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

function sde_problem(::LognormalDynamics, ::EulerMaruyama, m::BlackScholesInputs, tspan)
    rate = zero_rate(m.rate, 0.0)
    σ = get_vol(m.sigma, nothing, nothing)
    return LogGBMProblem(rate, σ, log(m.spot), tspan)
end

function get_antithetic_ensemble_problem(
    problem,
    ::PriceDynamics,
    ::EulerMaruyama,
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
    s::BlackScholesExact,
    normal_sol::EnsembleSolution,
    market::BlackScholesInputs,
    tspan
)
    @assert market.rate isa FlatRateCurve "LognormalDynamics requires flat rate curve"

    r = zero_rate(market.rate, 0.0)
    σ = -get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    t₀ = zero(S₀)  # ensure same type as S₀

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)

    noise_problem = NoiseProblem(noise, tspan)
    return get_ensemble_problem(noise_problem, s)
end

function get_antithetic_ensemble_problem(
    d::HestonDynamics,
    s::HestonBroadieKaya,
    normal_sol::CustomEnsembleSolution,
    m::HestonInputs,
)
    tspan = (normal_sol.solutions[1].t[1], normal_sol.solutions[1].t[end])
    # Assume s is a simulation strategy that has .seeds and .kwargs fields
    # and normal_sol is a solution from a previous simulation

    # First, copy the seeds from normal_sol
    s_flipped = @set s.seeds = normal_sol.seeds

    # Then, modify a specific keyword argument in s.kwargs
    s_flipped = @set s_flipped.kwargs = merge(s_flipped.kwargs, (antithetic = false,))
    flipped_problem = sde_problem(d, s_flipped, m, tspan)

    antithetic_modify = function (_base_prob, _seed, i)
        return _base_prob
    end

    return CustomEnsembleProblem(flipped_problem, normal_sol.seeds, antithetic_modify)
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

function get_ensemble_problem(prob, strategy)
    seeds = strategy.seeds
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

struct MonteCarlo{P<:PriceDynamics,S<:SimulationStrategy} <: AbstractPricingMethod
    dynamics::P
    strategy::S
end

function simulate_paths(
    method::M,
    market_inputs::I,
    T,
    ::NoVarianceReduction
) where {M<:MonteCarlo, I<:AbstractMarketInputs}
    strategy = method.strategy
    dynamics = method.dynamics
    tspan = (0.0, T)
    dt = T / strategy.steps
    normal_prob = sde_problem(dynamics, strategy, market_inputs, tspan)
    ensemble_prob = get_ensemble_problem(normal_prob, strategy)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=strategy.trajectories)
    return MonteCarloSol(normal_sol, nothing)
end

function simulate_paths(
    method::M,
    market_inputs::I,
    T,
    ::Antithetic
) where {M<:MonteCarlo, I<:AbstractMarketInputs}

    strategy = method.strategy
    dynamics = method.dynamics
    tspan = (0.0, T)
    dt = T / strategy.steps
    normal_prob = sde_problem(dynamics, strategy, market_inputs, tspan)
    ensemble_prob = get_ensemble_problem(normal_prob, strategy)
    normal_sol = DifferentialEquations.solve(ensemble_prob, EM(); dt = dt, trajectories=strategy.trajectories, save_noise=true)
    
    antithetic_prob = get_antithetic_ensemble_problem(normal_prob,dynamics, strategy, normal_sol, market_inputs, tspan)
    antithetic_sol = DifferentialEquations.solve(antithetic_prob, EM(); dt = dt, trajectories=strategy.trajectories)
    return MonteCarloSol(normal_sol, antithetic_sol)
end

struct MonteCarloSol{S}
    solution::EnsembleSolution
    antithetic_sol::S
end

function solve(
    prob::PricingProblem{VanillaOption{TS, TE, European, C, Spot}, I},
    method::MonteCarlo{D, S},
) where {TS, TE, C, I<:AbstractMarketInputs, D<:PriceDynamics, S<:SimulationStrategy}
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    strategy = method.strategy
    dynamics = method.dynamics

    ens = simulate_paths(method, prob.market_inputs, T, strategy.variance_reduction)  # returns EnsembleSolution
    # Assume state is 1D and we extract final value from each trajectory
    
    payoffs = reduce_payoffs(ens, prob.payoff, strategy.variance_reduction, dynamics, strategy)
    discount = df(prob.market_inputs.rate, prob.payoff.expiry)
    price = discount * mean(payoffs)

    return MonteCarloSolution(price, ens)
end

function reduce_payoffs(
    result::MonteCarloSol{Nothing},
    payoff::F,
    ::NoVarianceReduction,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    paths = result.solution.u
    return [payoff(get_terminal_value(p, dynamics, strategy)) for p in paths]
end

function reduce_payoffs(
    result::MonteCarloSol{<:EnsembleSolution},
    payoff::F,
    ::Antithetic,
    dynamics::PriceDynamics,
    strategy::SimulationStrategy,
) where {F}
    paths₁ = result.solution.u
    paths₂ = result.antithetic_sol.u
    return [
        (
            payoff(get_terminal_value(p1, dynamics, strategy)) +
            payoff(get_terminal_value(p2, dynamics, strategy))
        )
        for (p1, p2) in zip(paths₁, paths₂)
    ]
end



