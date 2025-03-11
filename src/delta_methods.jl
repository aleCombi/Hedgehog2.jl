"""A method for delta calculation"""
abstract type AbstractDeltaMethod end

"""A method for delta calculation analytically using black scholes"""
struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end

"""Delta calculator"""
struct DeltaCalculator{D<:AbstractDeltaMethod, P<:AbstractPayoff, I<:AbstractMarketInputs, S<:AbstractPricingMethod}
    pricer::Pricer{P, I, S}
    deltaMethod::D
end

"""Callable struct: Computes delta when called, using black scholes on a call."""
function (delta_calc::DeltaCalculator{BlackScholesAnalyticalDelta, VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod})()
    S = delta_calc.pricer.marketInputs.spot
    K = delta_calc.pricer.payoff.strike
    r = delta_calc.pricer.marketInputs.rate
    σ = delta_calc.pricer.marketInputs.sigma
    T = delta_calc.pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)  # Black-Scholes delta for calls
end

"""Delta with AD"""
struct ADDelta <: AbstractDeltaMethod end

"""Delta with AD callable"""
function (delta_calc::DeltaCalculator{ADDelta, P, BlackScholesInputs, S})() where {P,S}
    pricer = delta_calc.pricer
    return ForwardDiff.derivative(
        S -> begin
            new_pricer = @set pricer.marketInputs.spot = S
            new_pricer()
        end,
        pricer.marketInputs.spot
    )
end
