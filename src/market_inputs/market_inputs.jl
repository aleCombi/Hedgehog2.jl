export BlackScholesInputs, AbstractMarketInputs, HestonInputs

"""
    AbstractMarketInputs

An abstract type representing market data inputs required for pricers.
"""
abstract type AbstractMarketInputs end

"""
    BlackScholesInputs <: AbstractMarketInputs

Market data inputs for the Black-Scholes model.

# Fields
- `referenceDate`: The date from which maturity is measured.
- `rate`: The risk-free interest rate (annualized).
- `spot`: The current spot price of the underlying asset.
- `sigma`: The volatility of the underlying asset (annualized).

This struct encapsulates the necessary inputs for pricing derivatives under the Black-Scholes model.
"""
struct BlackScholesInputs <: AbstractMarketInputs
    referenceDate::Real
    rate::RateCurve
    spot
    sigma
end

BlackScholesInputs(
    reference_date::TimeType,
    rate::RateCurve,
    spot,
    sigma
) = BlackScholesInputs(to_ticks(reference_date), rate, spot, sigma)


BlackScholesInputs(
    reference_date::TimeType,
    rate::Real,
    spot,
    sigma
) = BlackScholesInputs(reference_date, FlatRateCurve(rate; reference_date=reference_date), spot, sigma)

# in distribution.jl t is Real, hence we need to redefine it.
"""
    cf(d::Normal, t) -> Complex

Computes the characteristic function of a normal distribution `d` evaluated at `t`.

# Arguments
- `d`: A `Normal` distribution.
- `t`: The evaluation point (can be real or complex).

# Returns
- The characteristic function value `E[e^{itX}]`.
"""
function cf(d::Normal, t) 
    return exp(im * t * d.μ - d.σ^2 / 2 * t^2)
end

"""
    HestonInputs <: AbstractMarketInputs

Market data inputs for the Heston stochastic volatility model.

# Fields
- `referenceDate`: The base date for maturity calculation.
- `rate`: The risk-free interest rate (annualized).
- `spot`: The current spot price of the underlying asset.
- `V0`: The initial variance of the underlying.
- `κ`: The rate at which variance reverts to its long-term mean.
- `θ`: The long-term mean of the variance.
- `σ`: The volatility of variance (vol-of-vol).
- `ρ`: The correlation between the asset and variance processes.

Used for pricing under the Heston model and simulation of stochastic volatility paths.
"""
struct HestonInputs <: AbstractMarketInputs
    referenceDate
    rate::RateCurve
    spot
    V0
    κ
    θ
    σ
    ρ
end

HestonInputs(
    reference_date,
    rate::Real,
    spot,
    V0,
    κ,
    θ,
    σ,
    ρ
) = HestonInputs(reference_date, FlatRateCurve(rate), spot, V0, κ, θ, σ, ρ)
