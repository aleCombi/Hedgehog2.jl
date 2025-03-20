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
    spot
    sigma
end

function log_distribution(m::BlackScholesInputs)
    r, σ, S0 = m.rate, m.sigma, m.spot
    d(t) = Normal(log(S0) + (r - σ^2 / 2)t, σ√t)  
    return d
end

price_process(m::BlackScholesInputs) = GeometricBrownianMotionProcess(m.rate, m.sigma, 0.0, 1.0)

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

log_distribution(m::HestonInputs) = t -> HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)