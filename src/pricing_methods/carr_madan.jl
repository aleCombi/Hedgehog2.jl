using QuadGK, Dates, Distributions

export CarrMadan 

struct CarrMadan <: AbstractPricingMethod
    α 
    bound
    dynamics
    kwargs #quadgk keyword arguments
end

log_dynamics(m::CarrMadan) = m.distribution

# Constructor with default empty kwargs
CarrMadan(α, bound, distribution; kwargs...) = CarrMadan(α, bound, distribution, Dict(kwargs...))

function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::CarrMadan) where {C,I <: AbstractMarketInputs}
    damp = exp(- method.α * log(payoff.strike)) / 2π
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    terminal_law = marginal_law(method.dynamics, market_inputs, T)
    ϕ(u) = cf(terminal_law, u)
    integrand(v) = call_transform(market_inputs.rate, T, ϕ, v, method) * exp(- im * v * log(payoff.strike))
    integral, _ = quadgk(integrand, -method.bound, method.bound; method.kwargs...)
    call_price = real(damp * integral)
    
    if C <: Call
        return call_price
    elseif C <: Put
        # Put-call parity
        return call_price - market_inputs.S0 + payoff.strike * exp(-market_inputs.rate * T)
    else
        error("Unsupported option type: $T")
    end
end


function call_transform(rate, time, ϕ, v, method::CarrMadan)
    numerator = exp(- rate * time) * ϕ(v - (method.α + 1)im)
    denominator = method.α^2 + method.α - v^2 + v * (2 * method.α + 1)im
    return numerator / denominator
end