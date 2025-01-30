module Hedgehog2

# # MARKET DATA
# market conventions, e.g. day-count conventions, rate conventions.
abstract type AbstractStaticData end

# past and present security prices, such as past fixed interest rates, stock prices.
abstract type AbstractSecurityPriceData end

# past and present derivatives prices, such as options quotes.
abstract type AbstractDerivativePriceData end

# derivatives trades specifics, such as vanilla call options, barrier options, interest rate swaps.
abstract type AbstractTradeData end

# # MODELS
# models, such as Black-Scholes, rate curve pricer, binomial model.
abstract type AbstractPricingModel end

# dynamics of the underlying of a derivatives, such as lognormal, Heston.
abstract type AbstractUnderlyingDynamics end 

# market-implied structures created by models, such as rate curves, volatility surfaces.
abstract type AbstractMarketImpliedStructure end

# model outputs relative to a trade, such as price, greeks.
abstract type AbstractModelOutputs end

# the whole algorithm to price a specific derivative.
# it should be a callable made up of all the ingredients: trade data, market data, market-implied structures, dynamics (if needed), a pricing model (coherent with the dynamics)
abstract type AbstractDerivativePricer end

end
