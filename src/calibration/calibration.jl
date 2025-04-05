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
    I <: Real,
    Q <: Real
}
    pricing_problem::BasketPricingProblem{P,M}
    pricing_method::A
    accessors::Vector{Accessor}
    quotes::Vector{Q}
    initial_guess::Vector{I}
end

abstract type CalibrationAlgo end
struct OptimizerAlgo{D, O} <: CalibrationAlgo
    diff::D
    optim_algo::O
end

function OptimizerAlgo()
    return OptimizerAlgo(AutoForwardDiff(), Optimization.LBFGS())
end

function solve(calib::CalibrationProblem, calib_algo::OptimizerAlgo; kwargs...)
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

struct RootFinderAlgo{R} <: CalibrationAlgo
    root_method::R
end

RootFinderAlgo() = RootFinderAlgo(nothing)

function solve(calib::CalibrationProblem, calib_algo::RootFinderAlgo; kwargs...)
    @assert length(calib.accessors) == 1 "Root-finding only supports calibration of a single parameter"
    @assert length(calib.quotes) == 1 "Root-finding expects a single target quote"
    lens = calib.accessors[1]
    quote_val = calib.quotes[1]
    pricing_problem = PricingProblem(calib.pricing_problem.payoffs[1], calib.pricing_problem.market_inputs)

    function f(x, _)
        updated_prob = set(pricing_problem, lens, x)
        sol = Hedgehog2.solve(updated_prob, calib.pricing_method)
        return sol.price - quote_val
    end

    problem = IntervalNonlinearProblem(f, (1E-6, 5.0))
    return NonlinearSolve.solve(problem, calib_algo.root_method; kwargs...)
end
