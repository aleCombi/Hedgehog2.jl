"""A pricing method"""
abstract type AbstractPricingMethod end

"""Black scholes method"""
struct BlackScholesMethod <: AbstractPricingMethod end

"""# the whole algorithm to price a specific derivative, using specific market inputs and a pricing method.
# it should be a callable made up of all the ingredients: a payoff, market data, a pricing model"""
struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S<:AbstractPricingMethod}
    marketInputs::M
    payoff::P
    pricingMethod::S
end

"""Dispatch of pricer for call black scholes pricing"""
function (pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod})() 
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end
