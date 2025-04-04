using Accessors
using Zygote
"""
    CalibrationProblem{P, M}

Wraps a `BasketPricingProblem` and defines which market parameters
should be calibrated using Accessors.jl access paths.

# Fields
- `pricing_problem`: The basket of payoffs with shared market input.
- `accessors`: A vector of accessor functions specifying which fields to calibrate.
"""
struct CalibrationProblem{P<:AbstractPayoff, M<:AbstractMarketInputs, A<:AbstractPricingMethod, Accessor}
    pricing_problem::BasketPricingProblem{P, M}
    pricing_method::A
    accessors::Vector{Accessor}
    quotes
end

using Optimization  # Or other backend

function solve(calib::CalibrationProblem, initial_guess; kwargs...)
    function objective(x, p)
        basket_prob = calib.pricing_problem

        # Apply each accessor to update the corresponding parameter
        # Dynamically apply all accessors to update basket_prob
        updated_problem = foldl(
            (prob, (lens, val)) -> set(prob, lens, val),
            zip(calib.accessors, x),
            init = basket_prob
        )

        # Solve updated problem
        basket_solution = solve(updated_problem, calib.pricing_method)

        # Compute squared pricing error
        sol_prices = map(sol -> sol.price, basket_solution.solutions)
        errors = sol_prices .- calib.quotes
        return sum(abs2, errors)
    end

    optf = OptimizationFunction(objective, AutoFiniteDiff())
    prob = OptimizationProblem(optf, initial_guess, nothing)
    result = Optimization.solve(prob, Optimization.LBFGS(); kwargs...)
    return result
end
