# Exported Types and Functions
"""
An abstract type representing a pricing method.

All pricing methods, such as Black-Scholes or binomial trees, should inherit from this type.
"""
abstract type AbstractPricingMethod end

struct PricingProblem{P<:AbstractPayoff,M<:AbstractMarketInputs}
    payoff::P
    market::M
end

function solve(
    payoff::P,
    market::I,
    method::M,
) where {P<:AbstractPayoff,I<:AbstractMarketInputs,M<:AbstractPricingMethod}
    return solve(PricingProblem(payoff, market), method)
end
