using QuadGK, Dates

# at the moment, only for black scholes. The dynamics should be use to get more generality.
struct CarrMadanBS <: AbstractPricingMethod 
    α 
    bound
end

function characteristic_function(t, market_inputs::BlackScholesInputs, method::CarrMadanBS)
    r = market_inputs.rate
    σ = market_inputs.sigma
    return u -> exp(im * (r - σ^2 / 2) * u * t - u^2 * t * σ^2 / 2)
end

function compute_price(payoff::VanillaOption{European, Call, Spot}, market_inputs::BlackScholesInputs, method::CarrMadanBS)
    damp = exp(- method.α * log(payoff.strike)) / 2π
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    ϕ = characteristic_function(T, market_inputs, method)
    integrand(v) = call_transform(market_inputs.rate, T, ϕ, v, method) * exp(- im * v * log(payoff.strike))
    integral, error = quadgk(integrand, -method.bound, method.bound)
    return real(damp * integral)
end

function call_transform(rate, time, ϕ, v, method::CarrMadanBS)
    numerator = exp(- rate * time) * ϕ(v - (method.α + 1)im)
    denominator = method.α^2 + method.α - v^2 + v * (2 * method.α + 1)im
    return numerator / denominator
end