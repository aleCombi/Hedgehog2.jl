export BlackScholesInputs, AbstractMarketInputs

"""An abstract type representing market data inputs required for pricers."""
abstract type AbstractMarketInputs end

"""Market data inputs for the Black-Scholes model.

# Fields
- `rate`: The risk-free interest rate.
- `spot`: The current spot price of the underlying asset.
- `sigma`: The volatility of the underlying asset.

This struct encapsulates the necessary inputs for pricing derivatives under the Black-Scholes model.
It is assumed that the volatility is annual.
"""
struct BlackScholesInputs <: AbstractMarketInputs
    referenceDate
    rate
    forward
    sigma
end
