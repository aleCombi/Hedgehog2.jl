
"""Computes the price of a vanilla European call option using the Black-Scholes model.

# Arguments
- `pricer::Pricer{VanillaCall, BlackScholesInputs, BlackScholesMethod}`: 
  A `Pricer` configured for Black-Scholes pricing of a vanilla European call.

# Returns
- The computed Black-Scholes price of the option.

The Black-Scholes formula used is:
```
d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
price = S * Φ(d1) - K * exp(-r * T) * Φ(d2)
```
where `Φ` is the CDF of the standard normal distribution.
"""
function compute_price(payoff::VanillaOption{European}, marketInputs::BlackScholesInputs, ::BlackScholesMethod)
  F = marketInputs.forward
  K = payoff.strike
  r = marketInputs.rate
  σ = marketInputs.sigma
  cp = payoff.call_put()
  T = Dates.value.(payoff.expiry .- marketInputs.referenceDate) ./ 365 # we might want to specify daycount conventions to ensure consistency
  d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
  d2 = d1 - σ * sqrt(T)
  return exp(-r*T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
end