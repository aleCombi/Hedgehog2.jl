abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end
using DiffEqFinancial
function (black_scholes_distribution::LognormalDynamics)(m::BlackScholesInputs)
    r, σ, S0 = m.rate, m.sigma, m.spot
    d(t) = Normal(log(S0) + (r - σ^2 / 2)t, σ√t)  
    return d
end

function sde_problem(m::BlackScholesInputs, ::LognormalDynamics, ::M, tspan) where M<:AbstractPricingMethod
    noise = GeometricBrownianMotionProcess(m.rate, m.sigma, 0.0, m.spot)
    return NoiseProblem(noise, tspan)
end

struct HestonDynamics <: PriceDynamics end

(h::HestonDynamics)(m::HestonInputs) = t -> HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)

function sde_problem(m::HestonInputs, dynamics::HestonDynamics, method::Montecarlo, tspan)
    noise = HestonNoise(0, dynamics(m), Z0=nothing; method.kwargs...)
    return NoiseProblem(noise, tspan)
end

struct HestonDynamicsEM <: PriceDynamics end

(h::HestonDynamicsEM)(m::HestonInputs) = t -> HestonDistribution(m.S0, m.V0, m.κ, m.θ, m.σ, m.ρ, m.rate, t)

function sde_problem(m::HestonInputs, ::HestonDynamicsEM, method::Montecarlo, tspan)
    return HestonProblem(m.rate, m.κ, m.θ, m.σ, m.ρ, [m.S0, m.V0], tspan)
end
