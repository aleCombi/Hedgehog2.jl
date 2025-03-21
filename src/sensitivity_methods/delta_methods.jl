# Exported Types and Functions
export AbstractDeltaMethod, BlackScholesAnalyticalDelta, DeltaCalculator, ADDelta

"""
An abstract type representing a method for delta calculation.

All delta calculation methods, such as analytical Black-Scholes delta or automatic differentiation, should inherit from this type.
"""
abstract type AbstractDeltaMethod end

"""
A method for delta calculation analytically using the Black-Scholes model.

This struct represents the analytical delta computation under the Black-Scholes pricing model, assuming continuous hedging and a lognormal underlying asset distribution.
"""
struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end

"""
A delta calculator that computes the sensitivity of an option's price to changes in the underlying asset's price.

# Type Parameters
- `D`: The method used for delta calculation (must be a subtype of `AbstractDeltaMethod`).
- `P`: The type of payoff being priced (must be a subtype of `AbstractPayoff`).
- `I`: The type of market data inputs required for pricing (must be a subtype of `AbstractMarketInputs`).
- `S`: The pricing method used (must be a subtype of `AbstractPricingMethod`).

# Fields
- `pricer`: The pricer used to compute the option price.
- `deltaMethod`: The delta calculation method.

A `DeltaCalculator` is a callable struct that computes delta using the specified method when invoked.
"""
struct DeltaCalculator{P<:AbstractPayoff, I<:AbstractMarketInputs, S<:AbstractPricingMethod, D<:AbstractDeltaMethod}
    pricer::Pricer{P, I, S}
    deltaMethod::D
end

"""
Computes the delta of a derivative using a `DeltaCalculator` object.

# Arguments
- `delta_calc`: A `DeltaCalculator` containing the pricer and delta calculation method.

# Returns
- The computed delta value of the derivative.

This function allows a `DeltaCalculator` instance to be called directly as a function, forwarding the computation to `compute_delta`.
"""
function (delta_calc::DeltaCalculator{A, B, C, D})() where {A<:AbstractPayoff, B<:AbstractMarketInputs, C<:AbstractPricingMethod, D<:AbstractDeltaMethod}
    return compute_delta(delta_calc.pricer, delta_calc.deltaMethod)
end

"""
Computes delta analytically for a vanilla European call or put option using the Black-Scholes model.

# Arguments
- `pricer`: A `Pricer` configured for Black-Scholes analytical delta calculation.

# Returns
- The computed delta value.

The Black-Scholes delta formula for a call option is:
```
d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
delta = Φ(d1)
```
and for a put option:
```
delta = Φ(d1) - 1
```
where `Φ` is the cumulative distribution function (CDF) of the standard normal distribution.
"""
function compute_delta(pricer::Pricer{VanillaOption{European, CallPut, Style}, BlackScholesInputs, BlackScholesAnalytic}, ::BlackScholesAnalyticalDelta) where {CallPut,Style}
    F = pricer.marketInputs.forward
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = Dates.value.(pricer.payoff.expiry .- pricer.marketInputs.referenceDate) ./ 365 # Assuming ACT/365 day count
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)
end

"""
A method for computing delta using automatic differentiation (AD).

This approach numerically computes the sensitivity of the option price to changes in the underlying price by leveraging `ForwardDiff` for differentiation.
"""
struct ADDelta <: AbstractDeltaMethod end

"""
Computes delta using automatic differentiation (AD).

# Arguments
- `pricer`: A `Pricer` configured for AD-based delta calculation.

# Returns
- The computed delta value using automatic differentiation.

This method uses `ForwardDiff.derivative` to compute the delta by differentiating the option price with respect to the spot price, avoiding explicit formulae and supporting complex payoffs.
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
