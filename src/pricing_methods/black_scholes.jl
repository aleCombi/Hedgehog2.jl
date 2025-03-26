# Black-Scholes pricing function for European vanilla options

export BlackScholesAnalytic

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
        return exp(-r * T) * payoff(marketInputs.spot * exp(r * T))
    end

    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return exp(-r * T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
end
