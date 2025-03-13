export AbstractPricingMethod, BlackScholesMethod, Pricer, compute_price, CoxRossRubinsteinMethod

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
  return compute_price(pricer.payoff, pricer.marketInputs, pricer.pricingMethod)
end

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

"""The Cox-Ross-Rubinstein binomial tree pricing method.

This struct represents the Cox-Ross-Rubinstein binomial pricing model for option pricing.
"""
struct CoxRossRubinsteinMethod <: AbstractPricingMethod 
  steps
end

# this should be specified by the tree choices, configurations of the pricing method
function binomial_tree_up_probability(up_step, down_step, ::CoxRossRubinsteinMethod)
  return (1 - down_step) / (up_step - down_step) 
end

function binomial_tree_step_sizes(sigma, time_step, ::CoxRossRubinsteinMethod)
  up_step = exp(sigma * sqrt(time_step))
  return up_step, 1 / up_step
end

function binomial_tree_forward(time_step, forward, up_step, down_step, ::CoxRossRubinsteinMethod)
  forward * up_step .^ (-time_step:2:time_step)
end

function binomial_tree_value(step, discounted_continuation, underlying_at_i, payoff, ::European)
  return discounted_continuation
end

function binomial_tree_value(step, discounted_continuation, underlying_at_i, payoff, ::American)
  return max.(discounted_continuation, payoff(underlying_at_i(step)))
end

function binomial_tree_underlying(time_step, forward, rate, delta_time, ::Spot)
  return exp(-rate*time_step*delta_time) * forward
end

function binomial_tree_underlying(_, forward, _, _, ::Forward)
  return forward
end

function compute_price(payoff::P, market_inputs::BlackScholesInputs, method::CoxRossRubinsteinMethod) where P <: AbstractPayoff
  ΔT = Dates.value.(payoff.expiry .- market_inputs.referenceDate) ./ 365 / method.steps # we might want to specify daycount conventions to ensure consistency
  up_step, down_step = binomial_tree_step_sizes(market_inputs.sigma, ΔT, method)
  forward_at_i(i) = binomial_tree_forward(i, market_inputs.forward, up_step, down_step, method)
  underlying_at_i(i) = binomial_tree_underlying(i, forward_at_i(i), market_inputs.rate, ΔT, payoff.underlying)
  p = binomial_tree_up_probability(up_step, down_step, method)
  value = payoff(forward_at_i(method.steps))

  for step in (method.steps-1):-1:0
    continuation = p * value[2:end] + (1 - p) * value[1:end-1]
    df = exp(-market_inputs.rate * ΔT)
    value = binomial_tree_value(step, df * continuation, underlying_at_i, payoff, payoff.exercise_style)
  end

  return value[1]

end