abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end

function (black_scholes_distribution::LognormalDynamics)(m::BlackScholesInputs)
    r, σ, S0 = m.rate, m.sigma, m.spot
    d(t) = Normal(log(S0) + (r - σ^2 / 2)t, σ√t)  
    return d
end

function price_process(m::BlackScholesInputs, ::LognormalDynamics, ::M) where M<:AbstractPricingMethod
    return GeometricBrownianMotionProcess(m.rate, m.sigma, 0.0, 1.0)
end

struct HestonDynamics <: PriceDynamics end

(h::HestonDynamics)(m::HestonInputs) = t -> HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)

function price_process(m::HestonInputs, dynamics::HestonDynamics, method::MontecarloExact)
    return HestonNoise(0, dynamics(m), Z0=nothing; method.kwargs...)
end