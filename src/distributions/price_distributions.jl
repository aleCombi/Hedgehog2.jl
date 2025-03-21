abstract type PriceDistribution end
struct BlackScholesPriceDistribution <: PriceDistribution end

function (black_scholes_distribution::BlackScholesPriceDistribution)(m::BlackScholesInputs)
    r, σ, S0 = m.rate, m.sigma, m.spot
    d(t) = Normal(log(S0) + (r - σ^2 / 2)t, σ√t)  
    return d
end

function price_process(m::BlackScholesInputs, ::BlackScholesPriceDistribution, ::M) where M<:AbstractPricingMethod
    return GeometricBrownianMotionProcess(m.rate, m.sigma, 0.0, 1.0)
end

struct HestonPriceDistribution <: PriceDistribution end

(h::HestonPriceDistribution)(m::HestonInputs) = t -> HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)
function price_process(m::HestonInputs, ::HestonPriceDistribution, method::MontecarloExact)
    return HestonNoise(t0, m, Z0=nothing; method.kwargs)
end