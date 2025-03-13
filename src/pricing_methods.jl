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
  payoff::P  
  marketInputs::M
  pricingMethod::S
end

"""
  Computes the price based on a Pricer input object.

  # Arguments
- `pricer::Pricer{A, B, C}`: 
  A `Pricer`, specifying a payoff, a market inputs and a method.

# Returns
- The computed price of the derivative.

"""
function (pricer::Pricer{A,B,C})() where {A<:AbstractPayoff,B<:AbstractMarketInputs,C<:AbstractPricingMethod}
  return price(pricer.payoff, pricer.marketInputs, pricer.pricingMethod)
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
function price(payoff::VanillaEuropeanCall, marketInputs::BlackScholesInputs, method::BlackScholesMethod)
    S = marketInputs.spot
    K = payoff.strike
    r = marketInputs.rate
    σ = marketInputs.sigma
    T = payoff.expiry
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end

"""The Cox-Ross-Rubinstein binomial tree pricing method.

This struct represents the Cox-Ross-Rubinstein binomial pricing model for option pricing.
"""
struct CoxRossRubinsteinMethod <: AbstractPricingMethod 
  steps
end

function price(payoff::P, market_inputs::BlackScholesInputs, method::CoxRossRubinsteinMethod) where P <: AbstractPayoff
  ΔT = (payoff.expiry - market_inputs.today) / method.steps
  step_size = exp(market_inputs.sigma * sqrt(ΔT))
  up_probability = 1 / (1 + step_size) # this should be specified by the tree choices, configurations of the
  spots_at_i = i -> step_size .^ (-i:2:i)

  p = up_probability
  value = payoff(spots_at_i(method.steps))
  print(spots_at_i(method.steps))
  for step in (method.steps-1):-1:0
    continuation = p * value[2:end] + (1 - p) * value[1:end-1]
    df = exp(-market_inputs.rate * ΔT)
    value = df * continuation
  end

  return value

end