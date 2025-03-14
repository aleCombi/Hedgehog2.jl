export AbstractDeltaMethod, BlackScholesAnalyticalDelta, DeltaCalculator, ADDelta

"""A method for delta calculation."""
abstract type AbstractDeltaMethod end

"""A method for delta calculation analytically using the Black-Scholes model."""
struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end

"""A delta calculator that computes the sensitivity of an option's price to changes in the underlying asset's price.

# Type Parameters
- `D <: AbstractDeltaMethod`: The method used for delta calculation.
- `P <: AbstractPayoff`: The type of payoff being priced.
- `I <: AbstractMarketInputs`: The type of market data inputs required for pricing.
- `S <: AbstractPricingMethod`: The pricing method used.

# Fields
- `pricer::Pricer{P, I, S}`: The pricer used to compute the option price.
- `deltaMethod::D`: The delta calculation method.

A `DeltaCalculator` is a callable struct that computes delta using the specified method.
"""
struct DeltaCalculator{P<:AbstractPayoff, I<:AbstractMarketInputs, S<:AbstractPricingMethod, D<:AbstractDeltaMethod}
    pricer::Pricer{P, I, S}
    deltaMethod::D
end

function (delta_calc::DeltaCalculator{A, B, C, D})() where {A<:AbstractPayoff, B<:AbstractMarketInputs, C<:AbstractPricingMethod, D<:AbstractDeltaMethod}
    return compute_delta(delta_calc.pricer, delta_calc.deltaMethod)
end

"""Computes delta analytically for a vanilla European call option using the Black-Scholes model.

# Arguments
- `delta_calc::DeltaCalculator{BlackScholesAnalyticalDelta, VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod}`: 
  A `DeltaCalculator` configured for Black-Scholes analytical delta calculation.

# Returns
- The computed delta value.

The Black-Scholes delta formula for a call option is:
```
d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
delta = Φ(d1)
```
where `Φ` is the CDF of the standard normal distribution.
"""
function compute_delta(pricer::Pricer{VanillaOption{European, CallPut, Style}, BlackScholesInputs, BlackScholesMethod}, ::BlackScholesAnalyticalDelta) where {CallPut,Style}
    F = pricer.marketInputs.forward
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = Dates.value.(pricer.payoff.expiry .- pricer.marketInputs.referenceDate) ./ 365 # we might want to specify daycount conventions to ensure consistency
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)
end

"""A method for computing delta using automatic differentiation (AD)."""
struct ADDelta <: AbstractDeltaMethod end

"""Computes delta using automatic differentiation (AD).

# Arguments
- `delta_calc::DeltaCalculator{ADDelta, P, BlackScholesInputs, S}`: 
  A `DeltaCalculator` configured for AD-based delta calculation.

# Returns
- The computed delta value using AD.

This method uses `ForwardDiff.derivative` to compute the delta by differentiating the option price with respect to the spot price.
"""
function compute_delta(pricer::Pricer{Payoff, BlackScholesInputs, Method}, ::ADDelta) where {Payoff <: AbstractPayoff, Method <: AbstractPricingMethod}
    return ForwardDiff.derivative(
        forward -> begin
            new_pricer = @set pricer.marketInputs.forward = forward
            new_pricer()
        end,
        pricer.marketInputs.forward
    )
end
