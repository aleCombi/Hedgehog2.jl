"""An abstract type representing a pricing method."""
abstract type AbstractPricingMethod end

"""The Black-Scholes pricing method.

This struct represents the Black-Scholes pricing model for option pricing.
"""
struct BlackScholesMethod <: AbstractPricingMethod end

"""A pricer that calculates the price of a derivative using a given payoff, market data, and a pricing model.

# Type Parameters
- `P <: AbstractPayoff`: The type of payoff being priced.
- `M <: AbstractMarketInputs`: The type of market data inputs required for pricing.
- `S <: AbstractPricingMethod`: The pricing method used.

# Fields
- `marketInputs::M`: The market data inputs used for pricing.
- `payoff::P`: The derivative payoff.
- `pricingMethod::S`: The pricing model used for valuation.

A `Pricer` is a callable struct that computes the price of the derivative using the specified pricing method.
"""
struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S <: AbstractPricingMethod}
    marketInputs::M
    payoff::P
    pricingMethod::S
end

"""Computes the price of a vanilla European call option using the Black-Scholes model.

# Arguments
- `pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod}`: 
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
function (pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod})()
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end
