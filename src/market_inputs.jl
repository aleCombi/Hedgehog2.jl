"""market data inputs for pricers"""
abstract type AbstractMarketInputs end

"""Inputs for black scholes model"""
struct BlackScholesInputs <: AbstractMarketInputs
    rate
    spot
    sigma
end