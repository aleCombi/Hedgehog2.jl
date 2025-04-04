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
struct CalibrationProblem{
    P<:AbstractPayoff,
    M<:AbstractMarketInputs,
    A<:AbstractPricingMethod,
    Accessor,
}
    pricing_problem::BasketPricingProblem{P,M}
    pricing_method::A
    accessors::Vector{Accessor}
    quotes::Any
    initial_guess::Any
end

abstract type CalibrationAlgo end
struct OptimizerAlgo <: CalibrationAlgo
    diff::Any # AutoForwardDiff()
    optim_algo::Any #Optimization.LBFGS()
end

function OptimizerAlgo()
    return OptimizerAlgo(AutoForwardDiff(), Optimization.LBFGS())
end

using Optimization  # Or other backend

function solve(calib::CalibrationProblem, calib_algo::CalibrationAlgo; kwargs...)
    function objective(x, p)
        basket_prob = calib.pricing_problem

        # Apply each accessor to update the corresponding parameter
        # Dynamically apply all accessors to update basket_prob
        updated_problem = foldl(
            (prob, (lens, val)) -> set(prob, lens, val),
            zip(calib.accessors, x),
            init = basket_prob,
        )

        # Solve updated problem
        basket_solution = solve(updated_problem, calib.pricing_method)

        # Compute squared pricing error
        sol_prices = map(sol -> sol.price, basket_solution.solutions)
        errors = sol_prices .- calib.quotes
        return sum(abs2, errors)
    end

    optf = OptimizationFunction(objective, calib_algo.diff)
    opt_prob = OptimizationProblem(optf, calib.initial_guess, nothing)
    result = Optimization.solve(opt_prob, calib_algo.optim_algo; kwargs...)
    return result
end
