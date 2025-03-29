# Exports
export ForwardAD, FiniteDifference, GreekProblem, SecondOrderGreekProblem, AnalyticGreek

# Method types
abstract type GreekMethod end

abstract type FDScheme end

struct FDForward <: FDScheme end
struct FDBackward <: FDScheme end
struct FDCentral <: FDScheme end

struct AnalyticGreek <: GreekMethod end

struct ForwardAD <: GreekMethod end

struct FiniteDifference{S<:FDScheme} <: GreekMethod
    bump
    scheme::S
end

FiniteDifference(bump) = FiniteDifference(bump, FDCentral())

# First-order GreekProblem
struct GreekProblem{P, L}
    pricing_problem::P
    wrt::L  # accessor (from Accessors.jl)
end

function solve(gprob::GreekProblem, ::ForwardAD, pricing_method::P) where P<:AbstractPricingMethod
    prob = gprob.pricing_problem
    lens = gprob.wrt

    x₀ = lens(prob)
    f = x -> solve(set(prob, lens, x), pricing_method).price

    deriv = ForwardDiff.derivative(f, x₀)
    return (greek = deriv,)
end

function compute_fd_derivative(::FDForward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ + ε)

    v_up = solve(prob_up, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v_up - v₀) / ε
end

function compute_fd_derivative(::FDBackward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_down = set(prob, lens, x₀ - ε)
    v_down = solve(prob_down, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v₀ - v_down) / ε
end

function compute_fd_derivative(::FDCentral, prob, lens, ε, pricing_method)
    x₀ = lens(prob)

    prob_up = set(prob, lens, x₀ + ε)
    prob_down = set(prob, lens, x₀ - ε)
    v_up = solve(prob_up, pricing_method).price
    v_down = solve(prob_down, pricing_method).price
    return (v_up - v_down) / (2ε)
end

function solve(gprob::GreekProblem, method::FiniteDifference{S}, pricing_method::P) where {S<:FDScheme, P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens = gprob.wrt
    ε = method.bump
    scheme = method.scheme
    deriv = compute_fd_derivative(scheme, prob, lens, ε, pricing_method)
    return (greek = deriv,)
end

# Second-order GreekProblem
struct SecondOrderGreekProblem{P, L1, L2}
    pricing_problem::P
    wrt1::L1
    wrt2::L2
end

function solve(gprob::SecondOrderGreekProblem, ::ForwardAD, pricing_method::P) where P<:AbstractPricingMethod
    prob = gprob.pricing_problem
    lens1, lens2 = gprob.wrt1, gprob.wrt2
    x₀, y₀ = lens1(prob), lens2(prob)

    f = (x, y) -> solve(set(set(prob, lens1, x), lens2, y), pricing_method).price

    if lens1 === lens2
        ∂f = x -> ForwardDiff.derivative(z -> f(x, z), x)
        deriv = ForwardDiff.derivative(∂f, x₀)
    else
        ∂f = x -> ForwardDiff.derivative(y -> f(x, y), y₀)
        deriv = ForwardDiff.derivative(∂f, x₀)
    end

    return (greek = deriv,)
end

function solve(gprob::SecondOrderGreekProblem, method::FiniteDifference, pricing_method::P) where P<:AbstractPricingMethod
    prob = gprob.pricing_problem
    lens1, lens2 = gprob.wrt1, gprob.wrt2
    ε = method.bump

    x₀, y₀ = lens1(prob), lens2(prob)

    f = (x, y) -> solve(set(set(prob, lens1, x), lens2, y), pricing_method).price

    if lens1 === lens2
        f_plus  = f(x₀ + ε, y₀ + ε)
        f_0     = f(x₀, y₀)
        f_minus = f(x₀ - ε, y₀ - ε)
        deriv = (f_plus - 2f_0 + f_minus) / (ε^2)
    else
        f_pp = f(x₀ + ε, y₀ + ε)
        f_pm = f(x₀ + ε, y₀ - ε)
        f_mp = f(x₀ - ε, y₀ + ε)
        f_mm = f(x₀ - ε, y₀ - ε)
        deriv = (f_pp - f_pm - f_mp + f_mm) / (4ε^2)
    end

    return (greek = deriv,)
end

function solve(gprob::GreekProblem, ::AnalyticGreek, ::BlackScholesAnalytic)
    prob = gprob.pricing_problem
    lens = gprob.wrt
    inputs = prob.market

    S = inputs.spot
    σ = inputs.sigma
    T = yearfrac(prob.market.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike

    D = df(prob.market.rate, T)
    F = prob.market.spot / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    Φ = x -> cdf(Normal(), x)
    ϕ = x -> pdf(Normal(), x)

    greek = if lens === @optic _.market.spot
        # Delta = ∂V/∂S = ∂V/∂F * ∂F/∂S = (cp * N(cp·d1)) * (1/D)
        cp * Φ(cp * d1)

    elseif lens === @optic _.market.sigma
        # Vega = ∂V/∂σ = D · F · φ(d1) · √T
        D * F * ϕ(d1) * √T

    else
        error("Unsupported lens for analytic Greek")
    end

    return (greek = greek,)
end

function solve(
    gprob::SecondOrderGreekProblem,
    ::AnalyticGreek,
    ::BlackScholesAnalytic
)
    prob = gprob.pricing_problem
    lens1 = gprob.wrt1
    lens2 = gprob.wrt2
    inputs = prob.market

    S = inputs.spot
    σ = inputs.sigma
    T = yearfrac(inputs.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike

    D = df(inputs.rate, prob.payoff.expiry)
    F = S / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    ϕ = x -> pdf(Normal(), x)

    greek = if (lens1 === @optic _.market.spot) && (lens2 === @optic _.market.spot)
        # Gamma = ∂²V/∂S² = φ(d1) / (Sσ√T)
        ϕ(d1) / (S * σ * √T)

    elseif (lens1 === @optic _.market.sigma) && (lens2 === @optic _.market.sigma)
        # Volga = Vega * d1 * d2 / σ
        vega = D * F * ϕ(d1) * √T
        vega * d1 * d2 / σ

    else
        error("Unsupported second-order analytic Greek")
    end

    return (greek = greek,)
end
