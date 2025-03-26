# Black-Scholes pricing function for European vanilla options

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
    compute_price(payoff::VanillaOption{European, A, B}, marketInputs::BlackScholesInputs, ::BlackScholesAnalytic) -> Float64

Computes the price of a European vanilla option (call or put) using the Black-Scholes formula.

# Arguments
- `payoff::VanillaOption{European, A, B}`: 
  A European-style vanilla option, where `A` and `B` denote the underlying and payout type.
- `marketInputs::BlackScholesInputs`: 
  Market inputs including spot price, risk-free rate, volatility, and reference date.
- `::BlackScholesAnalytic`: 
  Specifies the use of the Black-Scholes pricing method.

# Returns
- The present value of the option computed under the Black-Scholes model.

# Formula
```
d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
price = exp(-r * T) * cp * (F * Φ(cp * d1) - K * Φ(cp * d2))
```
where:
- `F = S * exp(r * T)` is the forward price,
- `K` is the strike price,
- `σ` is volatility,
- `r` is the risk-free rate,
- `T` is time to expiry in years,
- `cp` is +1 for call, -1 for put,
- `Φ` is the standard normal CDF.

# Notes
- If volatility `σ` is zero, the price equals the discounted payoff at expiry under deterministic growth.
- Time to maturity `T` is calculated in years assuming a 365-day year.
"""
function compute_price(payoff::VanillaOption{European, A, B}, marketInputs::BlackScholesInputs, ::BlackScholesAnalytic) where {A,B}
    K = payoff.strike
    r = marketInputs.rate
    σ = marketInputs.sigma
    cp = payoff.call_put()
    T = Dates.value.(payoff.expiry .- marketInputs.referenceDate) ./ 365  # Assuming 365-day convention
    F = marketInputs.spot * exp(r * T)

    if σ == 0
        print("ciao")
        return exp(-r * T) * payoff(marketInputs.spot * exp(r * T))
    end

    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return exp(-r * T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
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
