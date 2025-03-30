using Polynomials
export LSM

"""
    LSM <: AbstractPricingMethod

Least Squares Monte Carlo (LSM) pricing method for American options.

Uses regression to estimate continuation values and determine optimal stopping times.

# Fields
- `mc_method`: A `MonteCarlo` method specifying dynamics and simulation strategy.
- `degree`: Degree of the polynomial basis for regression.
"""
struct LSM <: AbstractPricingMethod
    mc_method::MonteCarlo
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
function LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, degree::Int; kwargs...)
    mc = MonteCarlo(dynamics, strategy; kwargs...)
    return LSM(mc, degree)
end

"""
    extract_spot_grid(sol)

Extracts the simulated spot paths from a `Vector` of state vectors. Returns a matrix of size (nsteps, npaths).
Each column corresponds to a single simulation path.
"""
function extract_spot_grid(sol::CustomEnsembleSolution)
    # Each s is a solution in sol.solutions, where s.u is a vector of states
    return hcat([getindex.(s.u, 1) for s in sol.solutions]...)  # size: (nsteps, npaths)
end


function solve(
    prob::PricingProblem{VanillaOption{American, C, Spot}, I},
    method::LSM
) where {I <: AbstractMarketInputs, C}

    if !is_flat(prob.market.rate)
        throw(ArgumentError("LSM pricing only supports flat rate curves."))
    end

    T = yearfrac(prob.market.referenceDate, prob.payoff.expiry)
    sol = simulate_paths(method.mc_method, prob.market, T)
    spot_grid = extract_spot_grid(sol) ./ prob.market.spot  # Normalize paths

    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market.rate, add_yearfrac(prob.market.referenceDate, T / nsteps))

    # (time_index, value) for each path
    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps + 1, p])) for p in 1:npaths]

    for i in (ntimes - 1):-1:2
        t = i - 1

        continuation = [
            discount^(stopping_info[p][1] - t) * stopping_info[p][2]
            for p in 1:npaths
        ]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = poly.(x)

        update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = prob.market.spot * mean(discounted_values)

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
    stopping_info::Vector{Tuple{Int, Float64}},
    paths::Vector{Int},
    cont_value::Vector{Float64},
    payoff_t::Vector{Float64},
    t::Int
)
    exercise = payoff_t[paths] .> cont_value
    stopping_info[paths[exercise]] .= [(t, payoff_t[p]) for p in paths[exercise]]
end
