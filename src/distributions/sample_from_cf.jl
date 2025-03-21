# sample a distribution knowing its characteristic function ϕ. ϵ is the error tolerance. 
# n is the choice of how many standard deviations to use in h derivation (check Broadie Kaya)
function sample_from_cf(rng, ϕ; ϵ=1E-5, n=7) 
    # sample a uniform on [0,1]
    u = Distributions.rand(rng, Uniform(0,1))

    # calculate initial guess using the moments of the distribution (following Broadie-Kaya)
    mean, variance = moments_from_cf(ϕ)
    normal_sample = mean + quantile(Normal(0,1), u) * variance
    initial_guess = normal_sample > 0 ? normal_sample : mean * 0.01

    # Fourier inversion to find the cdf, following Broadie Kaya 
    h = π / (mean + n * √variance)
    cdf = x -> cdf_from_cf(ϕ, x, h, ϵ)
    sample = inverse_cdf(cdf, u, initial_guess)
    return sample
end

# calculate moments using the characteristic function
function moments_from_cf(ϕ; h=1E-4)
    ϕhigh, ϕ0, ϕlow = ϕ(h), ϕ(0), ϕ(h)
    derivative = (ϕhigh - ϕ0) / h
    second_derivative = (ϕhigh + ϕlow - 2*ϕ0) / h^2
    mean = real(-im*derivative)
    variance = real(- second_derivative - mean^2)
    return mean, variance
end

# Calculate cdf, integrating the cf, like in Broadie Kaya.
# h is the discretization step, ϕ the chararacteristic function, ϵ the error tolerance, x the cdf argument.
# TODO: this could be done with SampleIntegration using Integration.jl, might improve the elegance and performance.
function cdf_from_cf(ϕ, x, h, ϵ)
    y = h* x / π
    result = y
    j = 0
    while true
        j += 1
        phi = ϕ(h*j)
        y = 2 / π * sin(h*j * x) / j * real(phi)
        result += y
        # Check stopping condition
        if abs(phi) / j < π * ϵ / 2
            break
        end
    end

    return result
end

# invert a cdf, trying with second order newton (secant), otherwise using bisection-style method
function inverse_cdf(cdf_func, u, initial_guess)
    func = y -> cdf_func(y) - u
    try
        sol = find_zero(func, initial_guess, Order2(); atol=1E-6, maxiters=10)
        return sol
    catch
        return find_zero(func, (0, 1); atol=1E-5, maxiters=10)
    end
end