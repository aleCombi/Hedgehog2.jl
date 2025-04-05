"""
An abstract type representing a pricing method.

All pricing methods should inherit from this type.
"""
abstract type AbstractPricingMethod end

"""
A `PricingProblem` bundles the payoff and the market inputs required to price a derivative.

# Type Parameters
- `P<:AbstractPayoff`: The payoff type (e.g., VanillaOption).
- `M<:AbstractMarketInputs`: The market data type (e.g., volatility surface, interest rate curve).

# Fields
- `payoff::P`: The payoff object describing the contract to be priced.
- `market::M`: The market data needed for pricing.
"""
struct PricingProblem{P<:AbstractPayoff,M<:AbstractMarketInputs}
    payoff::P
    market_inputs::M
end
