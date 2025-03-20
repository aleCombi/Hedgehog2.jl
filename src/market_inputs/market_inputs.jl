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

function log_distribution(m::BlackScholesInputs)
    r, σ = m.rate, m.sigma
    d(t) = Normal((r - σ^2 / 2)t, σ√t)  
    return d
end

struct HestonInputs <: AbstractMarketInputs
    referenceDate
    rate
    S0
    V0
    κ
    θ
    σ
    ρ
    r
end

function log_distribution(m::HestonInputs)
    d(t) = HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)
    return d
end