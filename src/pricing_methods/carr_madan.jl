using Dates, Distributions
import Integrals
export CarrMadan 

"""
    CarrMadan <: AbstractPricingMethod

Fourier transform-based pricing method for European options.

Implements the Carr-Madan method, which prices European options using the inverse Fourier transform
of the characteristic function of the log-price under the risk-neutral measure.

# Fields
- `α`: Damping factor to ensure integrability of the Fourier transform.
- `bound`: Integration bound for numerical quadrature.
- `dynamics`: The model dynamics providing the terminal characteristic function.
- `kwargs`: Additional keyword arguments passed to the integral solver.
"""
struct CarrMadan <: AbstractPricingMethod
    α 
    bound
    dynamics
    kwargs # integral keyword arguments
end

"""
    log_dynamics(m::CarrMadan)

Returns the log-price dynamics (distribution) used in the Carr-Madan method.
"""
function log_dynamics(m::CarrMadan) 
    return m.distribution
end

"""
    CarrMadan(α, bound, dynamics; kwargs...)

Constructs a `CarrMadan` method with optional integration settings for `quadgk`.

# Arguments
- `α`: Damping factor.
- `bound`: Integration bound (positive real number).
- `dynamics`: The price dynamics (must support `marginal_law`).
- `kwargs...`: Additional keyword arguments for `quadgk`.
"""
function CarrMadan(α, bound, dynamics; kwargs...) 
    return CarrMadan(α, bound, dynamics, Dict(kwargs...))
end

"""
    compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::CarrMadan) -> Float64

Computes the price of a European call or put using the Carr-Madan method.

# Arguments
- `payoff`: European vanilla option.
- `market_inputs`: Market data including spot, rate, and reference date.
- `method`: A `CarrMadan` method instance.

# Returns
- The present value of the option computed via Fourier inversion.

# Notes
- Uses the characteristic function of the terminal log-price.
- Integrates a damped and transformed version of the call payoff.
- Applies call-put parity to return the correct price for puts.
"""
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::CarrMadan) where {C,I <: AbstractMarketInputs}
    damp = exp(- method.α * log(payoff.strike)) / 2π
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    terminal_law = marginal_law(method.dynamics, market_inputs, T)
    ϕ(u) = cf(terminal_law, u)
    integrand(v, p) = damp * call_transform(market_inputs.rate, T, ϕ, v, method) * exp(-im * v * log(payoff.strike))

    iprob = IntegralProblem(integrand, -method.bound, method.bound)
    result = Integrals.solve(iprob, Integrals.HCubatureJL(); method.kwargs...)

    call_price = real(result.u)
    price = parity_transform(call_price, payoff, market_inputs.spot, T)
    return price
end

function solve(
    prob::PricingProblem{VanillaOption{European, C, Spot}, I},
    method::CarrMadan
) where {C, I <: AbstractMarketInputs}

    K = prob.payoff.strike
    r = prob.market.rate
    S = prob.market.spot
    T = Dates.value(prob.payoff.expiry - prob.market.referenceDate) / 365

    terminal_law = marginal_law(method.dynamics, prob.market, T)
    ϕ(u) = cf(terminal_law, u)

    logK = log(K)
    damp = exp(-method.α * logK) / (2π)
    integrand(v, p) = damp * call_transform(r, T, ϕ, v, method) * exp(-im * v * logK)

    iprob = IntegralProblem(integrand, -method.bound, method.bound, nothing)
    integral_result = Integrals.solve(iprob, Integrals.HCubatureJL(); method.kwargs...)

    call_price = real(integral_result.u)
    price = parity_transform(call_price, prob.payoff, S, T)

    return CarrMadanSolution(price, integral_result)
end

"""
    call_transform(rate, time, ϕ, v, method::CarrMadan)

Returns the Fourier-space representation of the damped call payoff.

# Arguments
- `rate`: Risk-free rate.
- `time`: Time to maturity.
- `ϕ`: Characteristic function of the log-price.
- `v`: Fourier variable.
- `method`: The `CarrMadan` pricing method instance.

# Returns
- The value of the integrand for the Carr-Madan integral.
"""
function call_transform(rate, time, ϕ, v, method::CarrMadan)
    numerator = exp(- rate * time) * ϕ(v - (method.α + 1)im)
    denominator = method.α^2 + method.α - v^2 + v * (2 * method.α + 1)im
    return numerator / denominator
end