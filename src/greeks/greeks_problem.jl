# Method types
abstract type GreekMethod end

struct GreekResult{T}
    greek::T
end


abstract type FDScheme end

struct FDForward <: FDScheme end
struct FDBackward <: FDScheme end
struct FDCentral <: FDScheme end

struct AnalyticGreek <: GreekMethod end

struct ForwardAD <: GreekMethod end

struct FiniteDifference{S<:FDScheme, A <: Number} <: GreekMethod
    bump::A
    scheme::S
end

FiniteDifference(bump) = FiniteDifference(bump, FDCentral())

# First-order GreekProblem
struct GreekProblem{P,L}
    pricing_problem::P
    wrt::L  # accessor (from Accessors.jl)
end

function solve(
    gprob::GreekProblem,
    ::ForwardAD,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens = gprob.wrt

    x₀ = lens(prob)
    f = x -> solve(set(prob, lens, x), pricing_method).price

    deriv = ForwardDiff.derivative(f, x₀)
    return (greek = deriv,)
end

function compute_fd_derivative(::FDForward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ * (1 + ε))

    v_up = solve(prob_up, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v_up - v₀) / (x₀ * ε)
end

function compute_fd_derivative(::FDBackward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_down = set(prob, lens, x₀ * (1 - ε))
    v_down = solve(prob_down, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v₀ - v_down) / (x₀ * ε)
end

function compute_fd_derivative(::FDCentral, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ * (1 + ε))
    prob_down = set(prob, lens, x₀ * (1 - ε))
    v_up = solve(prob_up, pricing_method).price
    v_down = solve(prob_down, pricing_method).price
    return (v_up - v_down) / (2ε * x₀)
end

function solve(
    gprob::GreekProblem,
    method::FiniteDifference{S,A},
    pricing_method::P,
) where {S<:FDScheme,P<:AbstractPricingMethod,A}
    prob = gprob.pricing_problem
    lens = gprob.wrt
    ε = method.bump
    scheme = method.scheme
    deriv = compute_fd_derivative(scheme, prob, lens, ε, pricing_method)
    return GreekResult(deriv)
end

# Second-order GreekProblem
struct SecondOrderGreekProblem{P,L1,L2}
    pricing_problem::P
    wrt1::L1
    wrt2::L2
end

function solve(
    gprob::SecondOrderGreekProblem,
    ::ForwardAD,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
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

    return GreekResult(deriv)
end

function solve(
    gprob::SecondOrderGreekProblem,
    method::FiniteDifference,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens1, lens2 = gprob.wrt1, gprob.wrt2
    ε = method.bump

    x₀, y₀ = lens1(prob), lens2(prob)

    f = (x, y) -> solve(set(set(prob, lens1, x), lens2, y), pricing_method).price

    if lens1 === lens2
        f_plus = f(x₀ + ε, y₀ + ε)
        f_0 = f(x₀, y₀)
        f_minus = f(x₀ - ε, y₀ - ε)
        deriv = (f_plus - 2f_0 + f_minus) / (ε^2)
    else
        f_pp = f(x₀ + ε, y₀ + ε)
        f_pm = f(x₀ + ε, y₀ - ε)
        f_mp = f(x₀ - ε, y₀ + ε)
        f_mm = f(x₀ - ε, y₀ - ε)
        deriv = (f_pp - f_pm - f_mp + f_mm) / (4ε^2)
    end

    return GreekResult(deriv)
end

function solve(
    gprob::GreekProblem{
        PricingProblem{VanillaOption{TS,TE,European,B,C},I},
        L,
    },
    ::AnalyticGreek,
    ::BlackScholesAnalytic,
) where {TS,TE,B,C,L, I<:BlackScholesInputs}
    prob = gprob.pricing_problem
    lens = gprob.wrt
    inputs = prob.market_inputs
    cp = prob.payoff.call_put()

    S = inputs.spot
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike
    σ = get_vol_yf(inputs.sigma, T, K)

    D = df(prob.market_inputs.rate, prob.payoff.expiry)
    F = prob.market_inputs.spot / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    Φ = x -> cdf(Normal(), x)
    ϕ = x -> pdf(Normal(), x)

    greek = if lens === @optic _.market_inputs.spot
        # Delta = ∂V/∂S = ∂V/∂F * ∂F/∂S = (cp * N(cp·d1)) * (1/D)
        cp * Φ(cp * d1)

    elseif lens === VolLens(1,1) #TODO: add logic
        # Vega = ∂V/∂σ = D · F · φ(d1) · √T
        D * F * ϕ(d1) * √T

    elseif lens === @optic _.payoff.expiry
        # Assume flat rate: z(T) = r ⇒ D(T) = exp(-rT), F(T) = S / D(T)
        r = zero_rate_yf(prob.market_inputs.rate, T)
        (r * prob.payoff.strike * D * Φ(d2) + F * D * σ * ϕ(d1) / (2√T)) / (MILLISECONDS_IN_YEAR_365) #against ticks, to match AD and FD. Observe that the sign is counterintuitive as it is a derivative against expiry in tticks, not against time-to-maturity in yearfrac

    else
        error("Unsupported lens for analytic Greek")
    end

    return GreekResult(greek)
end

function solve(gprob::SecondOrderGreekProblem, ::AnalyticGreek, ::BlackScholesAnalytic)
    prob = gprob.pricing_problem
    lens1 = gprob.wrt1
    lens2 = gprob.wrt2
    inputs = prob.market_inputs

    S = inputs.spot
    T = yearfrac(inputs.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike
    σ = get_vol_yf(inputs.sigma, T, K)

    D = df(inputs.rate, prob.payoff.expiry)
    F = S / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    ϕ = x -> pdf(Normal(), x)

    greek = if (lens1 === @optic _.market_inputs.spot) && (lens2 === @optic _.market_inputs.spot)
        # Gamma = ∂²V/∂S² = φ(d1) / (Sσ√T)
        ϕ(d1) / (S * σ * √T)

    elseif (lens1 === VolLens(1,1)) && (lens2 === VolLens(1,1)) #TODO: introduce logic for sigma
        # Volga = Vega * d1 * d2 / σ
        vega = D * F * ϕ(d1) * √T
        vega * d1 * d2 / σ

    else
        error("Unsupported second-order analytic Greek")
    end

    return GreekResult(greek)
end

struct BatchGreekProblem{P,L}
    pricing_problem::P
    lenses::L
end

function solve(
    gprob::BatchGreekProblem{P,L},
    method::GreekMethod,
    pricing_method::AbstractPricingMethod
) where {P,L}
    lenses = gprob.lenses
    prob = gprob.pricing_problem
   
    Dict(lens => solve(GreekProblem(prob, lens), method, pricing_method).greek for lens in lenses)
end
