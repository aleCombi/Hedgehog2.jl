# Exported Types and Functions
export AbstractPricingMethod, Pricer, compute_price, PricingProblem
using SciMLBase

"""
An abstract type representing a pricing method.

All pricing methods, such as Black-Scholes or binomial trees, should inherit from this type.
"""
abstract type AbstractPricingMethod end

"""
A pricer that calculates the price of a derivative using a given payoff, market data, and a pricing model.

# Type Parameters
- `P`: The type of payoff being priced (must be a subtype of `AbstractPayoff`).
- `M`: The type of market data inputs required for pricing (must be a subtype of `AbstractMarketInputs`).
- `S`: The pricing method used (must be a subtype of `AbstractPricingMethod`).

# Fields
- `payoff`: The derivative payoff.
- `marketInputs`: The market data inputs used for pricing.
- `pricingMethod`: The pricing model used for valuation.

A `Pricer` is a callable struct that computes the price of the derivative using the specified pricing method when invoked.
"""
struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S <: AbstractPricingMethod}
  payoff::P  
  marketInputs::M
  pricingMethod::S
end

"""
Computes the price of a derivative using a `Pricer` object.

# Arguments
- `pricer`: A `Pricer` containing the payoff, market inputs, and pricing method.

# Returns
- The computed price of the derivative based on the specified pricing model.

This function allows a `Pricer` instance to be called directly as a function, forwarding the computation to `compute_price`.
"""
function (pricer::Pricer{A,B,C})() where {A<:AbstractPayoff,B<:AbstractMarketInputs,C<:AbstractPricingMethod}
  return compute_price(pricer.payoff, pricer.marketInputs, pricer.pricingMethod)
end

struct PricingProblem{P<:AbstractPayoff, M<:AbstractMarketInputs}
  payoff::P
  market::M
end

function solve(prob::PricingProblem{P,I}, method::M) where {P<:AbstractPayoff,I<:AbstractMarketInputs,M<:AbstractPricingMethod}
  return compute_price(prob.payoff, prob.market, method)
end

function solve(payoff::P, market::I, method::M) where {P<:AbstractPayoff, I<:AbstractMarketInputs, M<:AbstractPricingMethod}
  return solve(PricingProblem(payoff, market), method)
end
