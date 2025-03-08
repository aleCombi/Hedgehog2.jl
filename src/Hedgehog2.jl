module Hedgehog2

# A payoff, such as a vanilla european option or an asian option or a forward.
abstract type AbstractPayoff end

struct VanillaEuropeanCall <: AbstractPayoff
    strike
    time
end

# vanilla european option callable to get the payoff given a spot price.
(payoff::VanillaEuropeanCall)(spot) = max(spot - payoff.strike, 0.0)

# market data inputs for pricers
abstract type AbstractMarketInputs end

struct BlackScholesInputs <: AbstractMarketInputs
    rate
    spot
    sigma
end


# # MODELS
# the whole algorithm to price a specific derivative.
# it should be a callable made up of all the ingredients: a payoff, market data, a pricing model
abstract type AbstractDerivativePricer end

using Distributions

struct BlackScholesPricer
    marketInputs::BlackScholesInputs
    payoff::VanillaEuropeanCall
end

(p::BlackScholesPricer)() = 
    let S = p.marketInputs.spot,
        K = p.payoff.strike,
        r = p.marketInputs.rate,
        σ = p.marketInputs.sigma,
        T = p.payoff.time,
        d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T)),
        d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
    end

market_inputs = BlackScholesInputs(0.01, 1, 0.4)
payoff = VanillaEuropeanCall(1, 1)
pricer = BlackScholesPricer(market_inputs, payoff)
print(pricer())

end
