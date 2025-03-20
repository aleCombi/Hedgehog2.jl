using QuadGK, Dates, Distributions

struct CarrMadan <: AbstractPricingMethod
    α 
    bound
    kwargs #quadgk keyword arguments
end

# Constructor with default empty kwargs
CarrMadan(α, bound; kwargs...) = CarrMadan(α, bound, Dict(kwargs...))

function compute_price(payoff::VanillaOption{European, Call, Spot}, market_inputs::I, method::CarrMadan) where I <: AbstractMarketInputs
    damp = exp(- method.α * log(payoff.strike)) / 2π
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    ϕ(u) = cf(log_distribution(market_inputs)(T), u)
    integrand(v) = call_transform(market_inputs.rate, T, ϕ, v, method) * exp(- im * v * log(payoff.strike))
    integral, _ = quadgk(integrand, -method.bound, method.bound; method.kwargs...)
    return real(damp * integral)
end

function call_transform(rate, time, ϕ, v, method::CarrMadan)
    numerator = exp(- rate * time) * ϕ(v - (method.α + 1)im)
    denominator = method.α^2 + method.α - v^2 + v * (2 * method.α + 1)im
    return numerator / denominator
end