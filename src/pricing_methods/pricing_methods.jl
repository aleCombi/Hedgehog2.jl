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