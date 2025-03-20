using Distributions, Random, SpecialFunctions, Roots

"""
    HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)

Defines a univariate distribution for `log(S_T)` under the exact Heston model.

- `S0`: Initial stock price
- `V0`: Initial variance
- `κ`: Mean-reversion speed
- `θ`: Long-run variance level
- `σ`: Volatility of variance (vol of vol)
- `ρ`: Correlation between price and variance
- `r`: Risk-free rate
- `T`: Time to maturity
"""
struct HestonDistribution <: ContinuousUnivariateDistribution
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

function sample_integral_V(VT, rng, dist::HestonDistribution)
    ϕ(u) = integral_var_chf(u, VT, dist)
    return sample_from_cf(rng, ϕ)
end

using SpecialFunctions

# characteristic function of ∫ V_t between 0 and T, conditional V_T. Check Broadie-Kaya or Lech-Osterlee 
function integral_var_chf(a, VT, dist::HestonDistribution)
    κ, θ, σ, V0, T = dist.κ, dist.θ, dist.σ, dist.V0, dist.T
    γ(a) = √(κ^2 - 2 * σ^2 * a * im)
    d = 4*κ*θ / σ^2

    ζ(x) = (- expm1(- x * T)) / x
    first_term = exp(-0.5 * (γ(a) - κ) * T) * ζ(κ) / ζ(γ(a))

    η(x) = x * (1 + exp(- x * T)) / (- expm1(- x * T))
    second_term = exp((V0 + VT) / σ^2 * ( η(κ) - η(γ(a)) ))

    ν(x) = √(V0 * VT) * 4 * x * exp(-0.5 * x * T) / σ^2 / (- expm1(- x * T))

    numerator = besseli(0.5*d - 1, ν(γ(a)))
    denominator = besseli(0.5*d - 1, ν(κ))
    third_term = numerator / denominator

    return first_term * second_term * third_term
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
function rand(rng::AbstractRNG, d::HestonDistribution)
    # Step 1: Sample V_T
    V_T = sample_V_T(rng, d)

    # Step 2: Sample ∫ V_t dt, conditional V0 and VT
    integral_V = sample_integral_V(V_T, rng, d)

    # Step 3: Sample log(S_T)
    log_S_T = sample_log_S_T(V_T, integral_V, rng, d)

    return exp(log_S_T)
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