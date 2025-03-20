# fourier inversion to find the cdf from the characteristic function
function cdf_function(ϕ)
    integrand(x) = u -> sin(u * x) / u * real(ϕ(u))
    F(x) = 2 / π * quadgk(integrand(x), 0, 200; maxevals=100)[1]
    return F
end

function inverse_cdf(cdf_func, u, initial_guess)
    func = y -> cdf_func(y) - u
    try
        return find_zero(func, initial_guess, Order2(); atol=1E-5, maxiters=100)
    catch
        return find_zero(func, (0, 1); atol=1E-5, maxiters=100)
    end
end

function sample_from_cdf(rng, cdf)
    u = Distributions.rand(rng, Uniform(0,1))
    res = inverse_cdf(cdf, u, 0)
    return res
end

function sample_from_cf(rng, ϕ) 
    u = Distributions.rand(rng, Uniform(0,1))
    h = 1E-4
    derivative = (ϕ(h) - ϕ(0)) / h
    second_derivative = (ϕ(h) + ϕ(-h) - 2* ϕ(0)) / h^2
    mean = real(-im*derivative)
    variance = real(- second_derivative - mean^2)
    normal_sample = mean + quantile(Normal(0,1), u) * variance
    initial_guess = normal_sample > 0 ? normal_sample : mean * 0.01

    cdf = cdf_function(ϕ)
    sample = inverse_cdf(cdf, u, initial_guess)
    return sample
end