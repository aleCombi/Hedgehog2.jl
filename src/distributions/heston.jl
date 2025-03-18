using Distributions, Random

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

"""
    rand(rng::AbstractRNG, d::HestonDistribution)

Samples `log(S_T)` using the exact Broadie-Kaya method.
"""
function rand(rng, d::HestonDistribution)
    κ, θ, σ, ρ, V0, T, S0, r = d.κ, d.θ, d.σ, d.ρ, d.V0, d.T, d.S0, d.r
    # 1. Sample V_T using the noncentral chi-squared distribution
    ν = (4κ * θ) / σ^2
    ψ = (4κ * exp(-κ * T) * V0) / (σ^2 * (1 - exp(-κ * T)))
    V_T = (σ^2 * (1 - exp(-κ * T)) / (4κ)) * Distributions.rand(rng, NoncentralChisq(ν, ψ))

    # 2. Sample log(S_T) given V_T
    μ_XT = log(S0) + (r - 0.5 * V0) * T + (1 - exp(-κ * T)) * (θ - V0) / (2κ)
    var_XT = (ρ^2 * V0 * (1 - exp(-κ * T))^2) / (2κ) + (1 - ρ^2) * (V0 * T + θ * (T - (1 - exp(-κ * T)) / κ))
    
    log_S_T = μ_XT + sqrt(var_XT) * randn(rng)  # Sample from Normal(μ_XT, sqrt(var_XT))

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
