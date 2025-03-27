# Black-Scholes pricing function for European vanilla options
import DifferentialEquations

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

"""
    implied_vol(price, pricer::Pricer{VanillaOption{European, A, B}, BlackScholesInputs, BlackScholesAnalytic};
                initial_guess=0.2, lower=1e-6, upper=5.0, kwargs...)

Finds implied volatility using Newton's method with AD. Falls back to Brent if needed.
TODO: in the input, pricer.marketInputs.sigma has a value which is not used at all.
"""
function implied_vol(
    price,
    pricer::Pricer{VanillaOption{European, A, B}, BlackScholesInputs, BlackScholesAnalytic};
    initial_guess = 0.2,
    lower = 1e-6,
    upper = 5.0,
    kwargs...
) where {A, B}

    function f(σ)
        pricer_sigma = @set pricer.marketInputs.sigma = σ
        return (pricer_sigma() - price)^2
    end

    try
        return find_zero(f, initial_guess, Roots.Order1(); kwargs...)
    catch e
        @info "Newton failed with error: $(e). Falling back to Brent."
        return find_zero(f, (lower, upper), Roots.Bisection(); kwargs...)
    end
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
