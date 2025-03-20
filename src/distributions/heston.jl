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

"""
    rand(rng::AbstractRNG, d::HestonDistribution)

Samples `log(S_T)` using the exact Broadie-Kaya method.
"""
function sample_V_T(rng::AbstractRNG, d::HestonDistribution)
    κ, θ, σ, V0, T = d.κ, d.θ, d.σ, d.V0, d.T

    d = 4*κ*θ / σ^2  # Degrees of freedom
    λ = 4*κ * exp(-κ * T) * V0 / (σ^2 * (- expm1(-κ * T)))  # Noncentrality parameter
    c = σ^2 * (- expm1(-κ * T)) / (4*κ)  # Scaling factor

    V_T = c * Distributions.rand(rng, NoncentralChisq(d, λ))
    return V_T
end

function integral_V_cdf(VT, rng, dist::HestonDistribution)
    Φ(u) = integral_var_char(u, VT, dist)
    integrand(x) = u -> sin(u * x) / u * real(Φ(u))
    integral_value(x) = quadgk(integrand(x), 0, 200; maxevals=100)[1]
    F(x) = 2 / π * integral_value(x) # specify like in paper (trapz)
    return F
end

function sample_integral_V(VT, rng, dist::HestonDistribution)
    F = integral_V_cdf(VT, rng, dist)

    # Generate samples
    unif = Uniform(0,1)  # Define uniform distribution
    u = Distributions.rand(rng, unif)
    res = inverse_cdf_rootfinding(F, u, 0, 1)
    return res
end

function inverse_cdf_rootfinding(cdf_func, u, y_min, y_max)
    func = y -> cdf_func(y) - u
    if (func(y_min)*func(y_max) < 0)
        return find_zero(y -> cdf_func(y) - u, (y_min, y_max); atol=1E-5, maxiters=100)  # Solve F(y) = u like in paper (newton 2nd order)
    else
        return find_zero(y -> cdf_func(y) - u, y_min; atol=1E-5, maxiters=100)
    end
end

using SpecialFunctions

""" Adjusts the argument of z to lie in (-π, π] by shifting it appropriately. """
function adjust_argument(z)
    θ = angle(z)  # Compute current argument
    m = round(Int, θ / π)  # Find the nearest integer multiple of π
    z_adjusted = z * exp(-im * m * π)  # Shift argument back into (-π, π]
    return z_adjusted, m
end

""" Compute the modified Bessel function I_{-ν}(z) with argument correction. """
function besseli_corrected(nu, z)
    return besseli(nu, z) #TODO: check if adjustment is included
    z_adj, m = adjust_argument(z)  # Ensure argument is in (-π, π]
    # println("adjusted besseli: ",m, " value ", nu, " val: ", z_adj)
    return exp(im * m * π * nu) * besseli(nu, z_adj)  # Compute Bessel function with corrected input
end

function integral_var_char(a, VT, dist::HestonDistribution)
    κ, θ, σ, V0, T = dist.κ, dist.θ, dist.σ, dist.V0, dist.T
    γ(a) = √(κ^2 - 2 * σ^2 * a * im)
    d = 4*κ*θ / σ^2  # Degrees of freedom

    ζ(x) = (- expm1(- x * T)) / x
    first_term = exp(-0.5 * (γ(a) - κ) * T) * ζ(κ) / ζ(γ(a))

    η(x) = x * (1 + exp(- x * T)) / (- expm1(- x * T))
    second_term = exp((V0 + VT) / σ^2 * ( η(κ) - η(γ(a)) ))

    ν(x) = √(V0 * VT) * 4 * x * exp(-0.5 * x * T) / σ^2 / (- expm1(- x * T))

    numerator = besseli_corrected(0.5*d - 1, ν(γ(a)))
    denominator = besseli_corrected(0.5*d - 1, ν(κ))
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

function randHest(rng, d)
    return rand(rng, d)
end

""" Full sampling process for S_T """
function rand(rng::AbstractRNG, d::HestonDistribution)
    # Step 1: Sample V_T
    V_T = sample_V_T(rng, d)

    # Step 2: Sample ∫ V_t dt
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