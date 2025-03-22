export BlackScholesInputs, AbstractMarketInputs, HestonInputs

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
    spot
    sigma
end

# in distribution.jl t is Real, hence we need to redefine it.
cf(d::Normal, t) = exp(im * t * d.μ - d.σ^2 / 2 * t^2)

struct HestonInputs <: AbstractMarketInputs
    referenceDate
    rate
    S0
    V0
    κ
    θ
    σ
    ρ
end