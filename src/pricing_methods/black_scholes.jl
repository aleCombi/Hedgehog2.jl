# Black-Scholes pricing function for European vanilla options

export BlackScholesAnalytic

"""
The Black-Scholes pricing method.

This struct represents the Black-Scholes pricing model for option pricing, which assumes a lognormal distribution for the underlying asset and continuous hedging.
"""
struct BlackScholesAnalytic <: AbstractPricingMethod end

log_dynamics(::BlackScholesAnalytic) = LognormalDynamics()

"""
Computes the price of a vanilla European call or put option using the Black-Scholes model.

# Arguments
- `payoff::VanillaOption{European}`: 
  A European-style vanilla option, either a call or put.
- `marketInputs::BlackScholesInputs`: 
  The market inputs required for Black-Scholes pricing, including forward price, risk-free rate, and volatility.
- `::BlackScholesMethod`: 
  A placeholder argument to specify the Black-Scholes pricing method.

# Returns
- The discounted expected price of the option under the Black-Scholes model.

# Formula
The Black-Scholes price is computed using:

```
d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
price = exp(-r * T) * cp * (F * Φ(cp * d1) - K * Φ(cp * d2))
```

where:
- `F` is the forward price of the underlying,
- `K` is the strike price,
- `σ` is the volatility,
- `r` is the risk-free rate,
- `T` is the time to expiry in years,
- `cp` is +1 for calls and -1 for puts,
- `Φ(x)` is the cumulative distribution function (CDF) of the standard normal distribution.

# Notes
- The time to expiry `T` is computed as the difference between the option's expiry date and the reference date, assuming a 365-day year.
- The function supports both call and put options through the `cp` factor, ensuring a unified formula.
"""
function compute_price(payoff::VanillaOption{European, A, B}, marketInputs::BlackScholesInputs, ::BlackScholesAnalytic) where {A,B}
    K = payoff.strike
    r = marketInputs.rate
    σ = marketInputs.sigma
    cp = payoff.call_put()
    T = Dates.value.(payoff.expiry .- marketInputs.referenceDate) ./ 365  # Assuming 365-day convention
    F = marketInputs.spot*exp(r*T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return exp(-r*T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
end
