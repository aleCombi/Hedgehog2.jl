using Distributions, Random, SpecialFunctions, Roots

using Distributions, DifferentialEquations, Random, StaticArrays

function HestonProblem(μ, κ, Θ, σ, ρ, u0, tspan; seed = UInt64(0), kwargs...)
    f = function (u, p, t)
        adj_var = max(u[2], 0)
        return @. [μ * u[1], κ * (Θ - adj_var)]
    end
    g = function (u, p, t)
        adj_var = sqrt(max(u[2], 0))
        return @. [adj_var * u[1], σ * adj_var]
    end
    Γ = [1 ρ; ρ 1]  # ensure this is Float64

    noise = CorrelatedWienerProcess(Γ, tspan[1], zeros(2))

    sde_f = SDEFunction(f, g)
    return SDEProblem(sde_f, u0, (tspan[1], tspan[2]), noise=noise, seed=seed, kwargs...)
end

# the first component of W is the log of price in heston, the second its vol
function HestonNoise(μ, κ, θ, σ, ρ, t0, W0, Z0 = nothing; kwargs...)
    @inline function Heston_dist(DW, W, dt, u, p, t, rng) #dist
        S, V = exp(W[end][1]), W[end][2]
        heston_dist_at_t = HestonDistribution(S, V, κ, θ, σ, ρ, μ, t + dt)
        return @fastmath rand(rng, heston_dist_at_t; kwargs...)  # Calls exact Heston sampler
    end

    return NoiseProcess{false}(t0, W0, Z0, Heston_dist, nothing)
end

struct HestonDistribution <: ContinuousMultivariateDistribution
    S0
    V0
    κ
    θ
    σ
    ρ
    r
    T
end

# sample Variance at T
function sample_V_T(rng::AbstractRNG, d::HestonDistribution)
    κ, θ, σ, V0, T = d.κ, d.θ, d.σ, d.V0, d.T

    d = 4*κ*θ / σ^2  # Degrees of freedom
    λ = 4*κ * exp(-κ * T) * V0 / (σ^2 * (- expm1(-κ * T)))  # Noncentrality parameter
    c = σ^2 * (- expm1(-κ * T)) / (4*κ)  # Scaling factor

    V_T = c * Distributions.rand(rng, NoncentralChisq(d, λ))
    return V_T
end

using SpecialFunctions

# Precompute and return characteristic function ϕ(a) for sampling
function sample_integral_V(VT, rng, dist::HestonDistribution; kwargs...)
    ϕ = HestonCFIterator(VT, dist)
    return sample_from_cf(rng, ϕ; kwargs...)
end

struct HestonCFIterator
    VT::Float64
    dist::HestonDistribution
    logIκ::ComplexF64
    ζκ::ComplexF64
    ηκ::ComplexF64
    ν::Float64
end

function HestonCFIterator(VT, dist::HestonDistribution)
    κ, θ, σ, V0, T = dist.κ, dist.θ, dist.σ, dist.V0, dist.T
    d = 4 * κ * θ / σ^2
    ν = 0.5 * d - 1

    ζκ = (-expm1(-κ * T)) / κ
    ηκ = κ * (1 + exp(-κ * T)) / (-expm1(-κ * T))
    νκ = √(V0 * VT) * 4 * κ * exp(-0.5 * κ * T) / σ^2 / (-expm1(-κ * T))
    logIκ = log(besseli(ν, νκ))

    return HestonCFIterator(VT, dist, logIκ, ζκ, ηκ, ν)
end


"""
    evaluate_chf(iter::HestonCFIterator, a::Real, θ_prev::Union{Nothing,Float64})

Evaluates the characteristic function at `a`, using previous angle `θ_prev` for unwrapping.
Returns a tuple `(ϕ, θ_unwrapped)` to be passed to the next step.
"""
function evaluate_chf(iter::HestonCFIterator, a::Real, θ_prev::Union{Nothing, Float64})
    κ, θ, σ, V0, T = iter.dist.κ, iter.dist.θ, iter.dist.σ, iter.dist.V0, iter.dist.T
    ν = iter.ν
    ζκ, ηκ, logIκ = iter.ζκ, iter.ηκ, iter.logIκ
    VT = iter.VT

    γ = sqrt(κ^2 - 2 * σ^2 * a * im)
    ζγ = (1 - exp(-γ * T)) / γ
    ηγ = γ * (1 + exp(-γ * T)) / (1 - exp(-γ * T))
    νγ = √(V0 * VT) * 4 * γ * exp(-0.5 * γ * T) / σ^2 / (1 - exp(-γ * T))

    first_term = exp(-0.5 * (γ - κ) * T) * (ζκ / ζγ)
    second_term = exp((V0 + VT) / σ^2 * (ηκ - ηγ))

    θ = angle(νγ)
    θ_unwrapped = if θ_prev === nothing
        θ
    else
        δ = θ - θ_prev
        δ -= 2π * round(δ / (2π))
        θ_prev + δ
    end

    νγ_unwrapped = abs(νγ) * cis(θ_unwrapped)
    logIγ = log(besseli(ν, νγ_unwrapped)) + im * ν * (θ_unwrapped - θ)
    bessel_ratio = exp(logIγ - logIκ)

    ϕ = first_term * second_term * bessel_ratio
    return ϕ, θ_unwrapped
end


"""
    log_besseli_corrected(ν, z, θ_ref)

Corrects besseli using argument unwrapping across branch cuts.
"""
function log_besseli_corrected(ν, z::Complex, θ_ref::Ref{Float64})
    θ = angle(z)
    if isnan(θ_ref[])
        θ_ref[] = θ  # initialize
    end

    # unwrap angle relative to previous
    δ = θ - θ_ref[]
    if δ > π
        δ -= 2π
    elseif δ < -π
        δ += 2π
    end
    θ_unwrapped = θ_ref[] + δ
    θ_ref[] = θ_unwrapped

    z_unwrapped = abs(z) * cis(θ_unwrapped)
    return log(besseli(ν, z_unwrapped)) + im * ν * (θ_unwrapped - θ)
end

""" Sample log(S_T) given V_T and integral_V. """
function sample_log_S_T(V_T, integral_V, rng::AbstractRNG, d::HestonDistribution)
    κ, θ, σ, ρ, V0, T, S0, r = d.κ, d.θ, d.σ, d.ρ, d.V0, d.T, d.S0, d.r

    # Compute conditional mean
    mu = log(S0) + r*T - 0.5*integral_V + (ρ/σ)*(V_T - V0 - κ*θ*T + κ*integral_V)

    # Compute conditional variance
    sigma2 = (1 - ρ^2) * integral_V

    # Sample log(S_T)
    log_S_T = mu + sqrt(sigma2) * randn(rng)

    return log_S_T
end

""" Full sampling process for S_T """
function rand(rng::AbstractRNG, d::HestonDistribution; kwargs...)
    # Step 1: Sample V_T
    V_T = sample_V_T(rng, d)

    # Step 2: Sample ∫ V_t dt, conditional V0 and VT
    integral_V = sample_integral_V(V_T, rng, d; kwargs...)

    # Step 3: Sample log(S_T)
    log_S_T = sample_log_S_T(V_T, integral_V, rng, d)

    return exp(log_S_T)
end

function rand(rng::AbstractRNG, d::HestonDistribution; kwargs...)
    d1 = HestonDistribution(d.S0, d.V0, d.κ, d.θ, d.σ, d.ρ, d.r, d.T)
    # Step 1: Sample V_T
    V_T = sample_V_T(rng, d1)

    # Step 2: Sample ∫ V_t dt, conditional V0 and VT
    integral_V = sample_integral_V(V_T, rng, d1; kwargs...)

    # Step 3: Sample log(S_T)
    log_S_T = sample_log_S_T(V_T, integral_V, rng, d1)

    return [log_S_T, V_T]
end

"""
    characteristic_function(d::HestonDistribution, u)

Computes the characteristic function of `log(S_T)`.
"""
function cf(d::HestonDistribution, u)
    T, S0, V0, κ, θ, σ, ρ, r = d.T, d.S0, d.V0, d.κ, d.θ, d.σ, d.ρ, d.r

    d1 = sqrt((κ - ρ * σ * im * u)^2 + σ^2 * (im * u + u^2))
    g = (κ - ρ * σ * im * u - d1) / (κ - ρ * σ * im * u + d1)

    C = (κ * θ / σ^2) * ((κ - ρ * σ * im * u - d1) * T - 2 * log((1 - g * exp(-d1 * T)) / (1 - g)))
    D = ((κ - ρ * σ * im * u - d1) / σ^2) * ((1 - exp(-d1 * T)) / (1 - g * exp(-d1 * T)))

    return exp(C + D * V0 + im * u * log(S0) + im * u * r * T)
end