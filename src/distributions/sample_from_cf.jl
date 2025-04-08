# sample a distribution knowing its characteristic function ϕ. ϵ is the error tolerance. 
# n is the choice of how many standard deviations to use in h derivation (check Broadie Kaya)
"""
    sample_from_cf(rng, ϕ; n=5, kwargs...)

Samples a random variable using its characteristic function `ϕ`, based on Fourier inversion as in Broadie-Kaya.
Uses moment-based heuristics for initial guesses and applies root-finding to invert the CDF.
- `n`: number of standard deviations used to derive the discretization step `h`
"""
function sample_from_cf(rng, ϕ; n = 5, kwargs...)
    # sample a uniform on [0,1]
    u = Distributions.rand(rng, Uniform(0, 1))
    # calculate initial guess using the moments of the distribution (following Broadie-Kaya)
    mean, variance = moments_from_cf(ϕ)
    σ² = max(variance, 1e-12)  # tiny epsilon floor
    normal_sample = mean + sqrt(σ²) * quantile(Normal(), u)
    initial_guess = normal_sample > 0 ? normal_sample : mean * 0.01
    max_guess = mean + 11 * sqrt(σ²)
    # Fourier inversion to find the cdf, following Broadie Kaya 
    h = π / (mean + n * √σ²)
    cdf = x -> cdf_from_cf(ϕ, x, h; kwargs...)
    sample = inverse_cdf(cdf, u, initial_guess, max_guess; kwargs...)
    return sample
end

# calculate moments using the characteristic function
# Estimate mean and variance from characteristic function ϕ
"""
    moments_from_cf(ϕ_iter::HestonCFIterator; h=1e-4)

Estimates the mean and variance from the characteristic function `ϕ_iter` using central differences.
"""
function moments_from_cf(ϕ_iter::HestonCFIterator; h = 1e-2)
    θ_prev = nothing

    ϕp, θ_prev = evaluate_chf(ϕ_iter, h, θ_prev)
    ϕ0, θ_prev = evaluate_chf(ϕ_iter, 0.0, θ_prev)
    ϕm, _ = evaluate_chf(ϕ_iter, -h, θ_prev)

    first_derivative = (ϕp - ϕm) / (2h)
    second_derivative = (ϕp - 2ϕ0 + ϕm) / h^2

    mean = real(-im * first_derivative)
    variance = real(-second_derivative - mean^2)

    return mean, variance
end

# Calculate cdf, integrating the cf, like in Broadie Kaya.
# h is the discretization step, ϕ the chararacteristic function, ϵ the error tolerance, x the cdf argument.
# TODO: this could be done with SampleIntegration using Integration.jl, might improve the elegance and performance.
"""
    cdf_from_cf(ϕ_iter::HestonCFIterator, x, h; cf_tol=1e-4, kwargs...)

Approximates the cumulative distribution function at `x` by numerically integrating the characteristic function.
Implements the method of Broadie-Kaya with truncation tolerance `cf_tol`.
"""
function cdf_from_cf(ϕ_iter::HestonCFIterator, x, h; cf_tol = 1e-3, kwargs...)
    if x < 0
        return 0.0
    end

    result = h * x / π
    prefactor = 2 / π
    θ_prev = nothing

    for j = 1:10^9
        a = h * j
        ϕ_val, θ_prev = evaluate_chf(ϕ_iter, a, θ_prev)
        term = prefactor * sin(a * x) / j * real(ϕ_val)
        result += term

        if abs(ϕ_val) / j < π * cf_tol / 2
            break
        end
    end

    return result
end

# invert a cdf, trying with second order newton (secant), otherwise using bisection-style method
"""
    inverse_cdf(cdf_func, u, initial_guess, max_guess; atol=1E-4, maxiter_newton=10, maxiter_bisection=100, kwargs...)

Inverts a monotonic CDF to find `x` such that `cdf_func(x) ≈ u`. 
Uses second-order Newton’s method (secant-style), falling back to bisection if needed.
"""
function inverse_cdf(
    cdf_func,
    u,
    initial_guess,
    max_guess;
    atol = 1E-4,
    maxiter_newton = 10,
    maxiter_bisection = 100,
    kwargs...,
)
    func = y -> cdf_func(y) - u
    try
        sol =
            find_zero(func, initial_guess, Order2(); atol = atol, maxeval = maxiter_newton)
        if (sol < 0 || abs(func(sol)) > atol)
            error("error in newton at $u.")
        end
        return sol
    catch
        if (func(0) * func(max_guess) > 0)
            @warn "Using a fall-back value $max_guess to invert the cdf at $u"
            return max_guess #fall-back value, when u∼1
        end
        sol = find_zero(func, (0, max_guess); xtol = atol, maxeval = maxiter_bisection)
        sol_val = abs(func(sol))
        if sol_val > atol
            @warn "Inverse CDF has precision $sol_val at $u."
        end
        return sol
    end
end
