"""
    LSM <: AbstractPricingMethod

Least Squares Monte Carlo (LSM) pricing method for American options.

Uses regression to estimate continuation values and determine optimal stopping times.

# Fields
- `mc_method`: A `MonteCarlo` method specifying dynamics and simulation strategy.
- `degree`: Degree of the polynomial basis for regression.
"""
struct LSM{M<:MonteCarlo} <: AbstractPricingMethod
    mc_method::M
    degree::Int  # degree of polynomial basis
end

"""
    LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, degree::Int; kwargs...)

Constructs an `LSM` pricing method with a given degree polynomial regression and Monte Carlo simulation backend.

# Arguments
- `dynamics`: The price dynamics.
- `strategy`: The simulation strategy.
- `degree`: Degree of polynomial basis.
- `kwargs...`: Additional arguments passed to the `MonteCarlo` constructor.
"""
function LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, config::SimulationConfig, degree::Int)
    mc = MonteCarlo(dynamics, strategy, config)
    return LSM(mc, degree)
end

"""
    extract_spot_grid(sol::EnsembleSolution)

Extracts the simulated spot paths from an `EnsembleSolution`. Returns a matrix of size `(nsteps, npaths)`,
where each column corresponds to one simulation path, and each row to a time step.

Assumes that `sol.u[i].u[j][1]` contains the spot at time `t[j]` for trajectory `i`.
"""
function extract_spot_grid(sol::EnsembleSolution)
    npaths = length(sol.u)
    nsteps = length(sol.u[1].t)
    spot_grid = Matrix{eltype(sol.u[1].u[1][1])}(undef, nsteps, npaths)

    @inbounds for j in 1:npaths
        @views spot_grid[:, j] = getindex.(sol.u[j].u, 1)
    end

    return spot_grid
end

function extract_spot_grid(sol_anti::Tuple{EnsembleSolution,EnsembleSolution})
    sol, antithetic = sol_anti
    npaths = length(sol.u)
    nsteps = length(sol.u[1].t)
    spot_grid = Matrix{eltype(sol.u[1].u[1][1])}(undef, nsteps, 2*npaths)

    @inbounds for j in 1:npaths
        @views spot_grid[:, j] = getindex.(sol.u[j].u, 1)
    end

    @inbounds for k in (npaths+1):(2*npaths)
        @views spot_grid[:, k] = getindex.(antithetic.u[k-npaths].u, 1)
    end

    return spot_grid
end


function solve(
    prob::PricingProblem{VanillaOption{TS,TE,American,C,S},I},
    method::L,
) where {TS,TE,I<:AbstractMarketInputs,C,S, L<:LSM}

    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    sol = Hedgehog2.simulate_paths(prob, method.mc_method, method.mc_method.config.variance_reduction)
    spot_grid = Hedgehog2.extract_spot_grid(sol) ./ prob.market_inputs.spot  # Normalize paths
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    # (time_index, value) for each path
    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = prob.market_inputs.spot * mean(discounted_values)

    return LSMSolution(price, stopping_info, spot_grid)
end

"""
    update_stopping_info!(
        stopping_info::Vector{Tuple{Int, Float64}},
        paths::Vector{Int},
        cont_value::Vector{Float64},
        payoff_t::Vector{Float64},
        t::Int
    )

Updates the stopping times and payoffs based on exercise decision.
Replaces values in `stopping_info` if immediate exercise is better than continuation.
"""
function update_stopping_info!(
    stopping_info::Vector{Tuple{Int,Float64}},
    paths::Vector{Int},
    cont_value::Vector{Float64},
    payoff_t::Vector{Float64},
    t::Int,
)
    exercise = payoff_t[paths] .> cont_value
    stopping_info[paths[exercise]] .= [(t, payoff_t[p]) for p in paths[exercise]]
end
