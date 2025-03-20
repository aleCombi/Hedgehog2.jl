# fourier inversion to find the cdf from the characteristic function
function cdf_function(ϕ)
    integrand(x) = u -> sin(u * x) / u * real(ϕ(u))
    F(x) = 2 / π * quadgk(integrand(x), 0, 200; maxevals=100)[1]
    return F
end

function inverse_cdf(cdf_func, u, y_min, y_max)
    func = y -> cdf_func(y) - u
    if (func(y_min)*func(y_max) < 0)
        return find_zero(y -> cdf_func(y) - u, (y_min, y_max); atol=1E-5, maxiters=100)  # Solve F(y) = u like in paper (newton 2nd order)
    else
        return find_zero(y -> cdf_func(y) - u, y_min; atol=1E-5, maxiters=100)
    end
end

function sample_from_cdf(rng, cdf)
    u = Distributions.rand(rng, Uniform(0,1))
    res = inverse_cdf(cdf, u, 0, 1)
    return res
end

sample_from_cf(rng, ϕ) = sample_from_cdf(rng, cdf_function(ϕ))