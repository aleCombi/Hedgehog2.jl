# ------------------ Price Dynamics ------------------

abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end
struct HestonDynamics <: PriceDynamics end

# ------------------ Simulation Strategies ------------------

abstract type SimulationStrategy end

struct HestonBroadieKaya <: SimulationStrategy
    trajectories::Any
    steps::Any
    kwargs::NamedTuple
    seeds::Union{Nothing,Vector{Int}}
end

struct EulerMaruyama <: SimulationStrategy
    trajectories::Any
    steps::Any
    kwargs::NamedTuple
    seeds::Union{Nothing,Vector{Int}}
end

struct BlackScholesExact <: SimulationStrategy
    trajectories::Any
    steps::Any
    kwargs::NamedTuple
    seeds::Union{Nothing,Vector{Int}}
end

HestonBroadieKaya(trajectories; steps = 1, seeds = nothing, kwargs...) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    HestonBroadieKaya(trajectories, steps, (; kwargs...), seeds)
end

EulerMaruyama(trajectories; steps = 1, seeds = nothing, kwargs...) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    EulerMaruyama(trajectories, steps, (; kwargs...), seeds)
end

BlackScholesExact(trajectories; steps = 1, seeds = nothing, kwargs...) = begin
    seeds === nothing && (seeds = Base.rand(1_000_000_000:2_000_000_000, trajectories))
    BlackScholesExact(trajectories, steps, (; kwargs...), seeds)
end

# ------------------ SDE Problem Builders ------------------

function sde_problem(
    ::LognormalDynamics,
    s::BlackScholesExact,
    market::BlackScholesInputs,
    tspan,
)
    @assert is_flat(market.rate) "LognormalDynamics requires flat rate curve"

    # Promote all parameters to a common type (Dual or Float64)
    T = promote_type(
        typeof(zero_rate(market.rate, 0.0)),
        typeof(market.sigma),
        typeof(market.spot),
    )

    r = convert(T, zero_rate(market.rate, 0.0))
    σ = convert(T, market.sigma)
    S₀ = convert(T, market.spot)
    t₀ = zero(T)

    noise = GeometricBrownianMotionProcess(r, σ, t₀, S₀)
    return NoiseProblem(noise, tspan; s.kwargs...)
end

function sde_problem(::HestonDynamics, ::EulerMaruyama, m::HestonInputs, tspan)
    @assert is_flat(m.rate) "Heston simulation requires flat rate curve"
    rate = zero_rate(m.rate, 0.0)
    return HestonProblem(rate, m.κ, m.θ, m.σ, m.ρ, [m.spot, m.V0], tspan)
end

function sde_problem(::LognormalDynamics, ::EulerMaruyama, m::BlackScholesInputs, tspan)
    @assert is_flat(m.rate) "Heston simulation requires flat rate curve"
    rate = zero_rate(m.rate, 0.0)
    return LogGBMProblem(rate, m.sigma, log(m.spot), tspan)
end

function get_antithetic_ensemble_problem(
    ::PriceDynamics,
    ::EulerMaruyama,
    normal_sol::CustomEnsembleSolution,
    market_inputs,
)
    base_prob = normal_sol.solutions[1].prob

    antithetic_modify = function (_base_prob, _seed, i)
        sol = normal_sol.solutions[i]
        flipped_noise = NoiseGrid(sol.W.t, -sol.W.W)
        return remake(_base_prob; noise = flipped_noise)
    end

    return CustomEnsembleProblem(base_prob, normal_sol.seeds, antithetic_modify)
end

function get_antithetic_ensemble_problem(
    d::LognormalDynamics,
    s::BlackScholesExact,
    normal_sol::CustomEnsembleSolution,
    m::BlackScholesInputs,
)
    tspan = (normal_sol.solutions[1].t[1], normal_sol.solutions[1].t[end])
    s_flipped = @set s.seeds = normal_sol.seeds
    m_flipped = @set m.sigma = -m.sigma
    flipped_problem = sde_problem(d, s_flipped, m_flipped, tspan)

    antithetic_modify = function (_base_prob, _seed, i)
        return _base_prob
    end

    return CustomEnsembleProblem(flipped_problem, normal_sol.seeds, antithetic_modify)
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
    @assert is_flat(m.rate) "Heston simulation requires flat rate curve"
    rate = zero_rate(m.rate, 0.0)
    noise = HestonNoise(
        rate,
        m.κ,
        m.θ,
        m.σ,
        m.ρ,
        0.0,
        [log(m.spot), m.V0],
        Z0 = nothing;
        strategy.kwargs...,
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

function get_ensemble_problem(
    problem::Union{NoiseProblem,SDEProblem},
    strategy::S,
) where {S<:SimulationStrategy}
    N = strategy.trajectories

    seeds =
        strategy isa BlackScholesExact && strategy.seeds !== nothing ? strategy.seeds :
        Base.rand(1_000_000_000:2_000_000_000, N)

    modify = (p, seed, _) -> remake(p; seed = seed)
    custom_prob = CustomEnsembleProblem(problem, collect(seeds), modify)
    return custom_prob
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
    method::MonteCarlo,
    market_inputs::I,
    T,
) where {I<:AbstractMarketInputs}
    strategy = method.strategy
    dynamics = method.dynamics
    tspan = (0.0, T)
    dt = T / strategy.steps

    antithetic = get(strategy.kwargs, :antithetic, false)

    # Step 1: simulate original paths
    normal_prob = sde_problem(dynamics, strategy, market_inputs, tspan)
    ensemble_prob = get_ensemble_problem(normal_prob, strategy)
    normal_sol = solve_custom_ensemble(ensemble_prob; dt = dt)

    if !antithetic
        return normal_sol
    end

    # Step 2: simulate antithetic paths using same seeds, flipped sigma
    antithetic_ensemble_prob =
        get_antithetic_ensemble_problem(dynamics, strategy, normal_sol, market_inputs)
    antithetic_sol = solve_custom_ensemble(antithetic_ensemble_prob; dt = dt)

    # Step 3: combine both solutions
    combined_paths = vcat(normal_sol.solutions, antithetic_sol.solutions)

    # Create new EnsembleSolution with merged paths
    return CustomEnsembleSolution(combined_paths, normal_sol.seeds)
end

function solve(
    prob::PricingProblem{VanillaOption{European,C,Spot},I},
    method::MonteCarlo,
) where {C,I<:AbstractMarketInputs}
    T = yearfrac(prob.market.referenceDate, prob.payoff.expiry)
    strategy = method.strategy
    dynamics = method.dynamics

    ens = simulate_paths(method, prob.market, T)
    paths = ens.solutions

    is_antithetic = get(strategy.kwargs, :antithetic, false)

    if is_antithetic
        N = length(paths) ÷ 2
        terminal_1 = [get_terminal_value(p, dynamics, strategy) for p in paths[1:N]]
        terminal_2 = [get_terminal_value(p, dynamics, strategy) for p in paths[N+1:end]]
        payoffs = [
            0.5 * (prob.payoff(x) + prob.payoff(y)) for
            (x, y) in zip(terminal_1, terminal_2)
        ]
    else
        terminal_prices = [get_terminal_value(p, dynamics, strategy) for p in paths]
        payoffs = prob.payoff.(terminal_prices)
    end

    price = df(prob.market.rate, prob.payoff.expiry) * mean(payoffs)
    return MonteCarloSolution(price, ens)
end
