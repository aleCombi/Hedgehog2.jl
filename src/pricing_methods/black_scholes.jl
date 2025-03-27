# Black-Scholes pricing function for European vanilla options
import DifferentialEquations
using NonlinearSolve, Distributions, Roots

export BlackScholesAnalytic, implied_vol

"""
    BlackScholesAnalytic <: AbstractPricingMethod

The Black-Scholes pricing method.

Represents the analytical Black-Scholes model for pricing European-style vanilla options. 
Assumes the underlying follows lognormal dynamics with constant volatility and interest rate.
"""
struct BlackScholesAnalytic <: AbstractPricingMethod end

"""
    log_dynamics(::BlackScholesAnalytic) -> LognormalDynamics

Returns the assumed lognormal price dynamics for the Black-Scholes model.
"""
function log_dynamics(::BlackScholesAnalytic)
  return LognormalDynamics()
end

"""
    solve(prob::PricingProblem, ::BlackScholesAnalytic) -> AnalyticSolution

Computes the price of a European vanilla option under Black-Scholes.
Returns an `AnalyticSolution` with the price.
"""
function solve(
    prob::PricingProblem{VanillaOption{European, A, B}, BlackScholesInputs},
    ::BlackScholesAnalytic
) where {A, B}

    K = prob.payoff.strike
    r = prob.market.rate
    σ = prob.market.sigma
    cp = prob.payoff.call_put()
    T = Dates.value(prob.payoff.expiry - prob.market.referenceDate) / 365
    F = prob.market.spot * exp(r * T)

    price = if σ == 0
        exp(-r * T) * prob.payoff(prob.market.spot * exp(r * T))
    else
        d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
        d2 = d1 - σ * sqrt(T)
        exp(-r * T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
    end

    return AnalyticSolution(price)
end

# Define the calibration problem
struct BlackScholesCalibrationProblem{P<:PricingProblem, M}
    prob::P
    method::M
    price_target::Float64
end

function solve(calib::BlackScholesCalibrationProblem; initial_guess=0.2, lower=1e-6, upper=5.0, kwargs...)
    # Define the root-finding function for implied volatility calibration
    function calibration_function(σ, p)
        # Update the market inputs with candidate volatility σ using the @set macro
        market = calib.prob.market
        new_market = @set market.sigma = σ
        new_prob = PricingProblem(calib.prob.payoff, new_market)
        sol = solve(new_prob, calib.method)
        return sol.price - calib.price_target  # Objective function
    end

    # Set up the nonlinear solver problem
    problem = NonlinearProblem(calibration_function, initial_guess)

    # Solve the problem using the NonlinearSolver
    solution = NonlinearSolve.solve(problem; kwargs...)

    return solution
end

# function ImpliedVolSurface(
#     reference_date::Date,
#     tenors::AbstractVector,
#     strikes::AbstractVector,
#     rate,
#     spot,
#     call_prices::AbstractMatrix;
#     interp_type = Gridded(Linear()),
#     extrap_type = Flat();
#     kwargs...
# )
#     function vol_func(strike, tenor, price)
#         market_inputs = BlackScholesInputs(reference_date, rate, spot, 0.0)
#         payoff = VanillaOption(strike, reference_date + Day(365 * tenor), European(), Call(), Spot())
#         pricer = Pricer(payoff, market_inputs, BlackScholesAnalytic())
#         return implied_vol(price, pricer; kwargs...)
#     end

#     vols = [vol_func(strikes[j], tenors[i], call_prices[i,j]) for i in eachindex(tenors), j in eachindex(strikes)]
#     return RectVolSurface(reference_date, tenors, strikes, vols, interp_type, extrap_type)
# end
