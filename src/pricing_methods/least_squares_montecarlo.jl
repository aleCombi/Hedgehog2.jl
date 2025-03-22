using Polynomials
export LSM

struct LSM <: AbstractPricingMethod
    mc_method::MonteCarlo
    degree::Int  # degree of polynomial basis
end

LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, degree::Int) =
    LSM(MonteCarlo(dynamics, strategy), degree)
    
function extract_spot_grid(sol)
    # Each path is a Vector of state vectors; we extract first component at each time step
    return hcat([getindex.(s.u, 1) for s in sol.u]...)  # size: (nsteps, npaths)
end

function compute_price(
    payoff::VanillaOption{American, C, Spot},
    market_inputs::I,
    method::LSM
) where {I <: AbstractMarketInputs, C}

    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    sol = simulate_paths(method.mc_method, market_inputs, T)
    spot_grid = extract_spot_grid(sol) ./ market_inputs.spot

    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = exp(-market_inputs.rate * T / nsteps)

    # (time_index, payoff_value) per path
    stopping_info = [(nsteps, payoff(spot_grid[nsteps + 1, p])) for p in 1:npaths]

    for i in (ntimes - 1):-1:2
        t = i - 1 #the matrix indices are 1-based, but times are 0-based

        continuation = [
            discount^(stopping_info[p][1] - t) * stopping_info[p][2]
            for p in 1:npaths
        ]

        payoff_t = payoff.(spot_grid[i, :])

        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = poly.(x)

        update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    return market_inputs.spot * mean(discounted_values)
end

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

