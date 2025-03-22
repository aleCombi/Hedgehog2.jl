abstract type PriceDynamics end
struct LognormalDynamics <: PriceDynamics end

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


function montecarlo_solution(problem, ::HestonDynamicsEM, method)
    solution = solve(EnsembleProblem(problem), EM(); dt=method.dt, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    prices = getindex.(last.(solution.u), 1)
    return prices
end

function montecarlo_solution(problem, ::PriceDynamics, method)
    solution = solve(EnsembleProblem(problem); dt=T, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    return last.(solution.u)
end

function HestonProblem(μ, κ, Θ, σ, ρ, u0::AbstractVector{<:AbstractFloat}, tspan::Tuple{<:Real, <:Real}; seed = UInt64(0), kwargs...)
    f = function (u, p, t)
        return @. [μ * u[1], κ * (Θ - u[2])]
    end
    g = function (u, p, t)
        adj_var = sqrt(max(u[2], 0))
        return @. [adj_var * u[1], σ * adj_var]
    end
    Γ = [1 ρ; ρ 1]  # ensure this is Float64

    noise = CorrelatedWienerProcess(Γ, tspan[1], zeros(Float64, 2))

    sde_f = SDEFunction(f, g)
    return SDEProblem(sde_f, u0, (tspan[1], tspan[2]), noise=noise, seed=seed, kwargs...)
end
