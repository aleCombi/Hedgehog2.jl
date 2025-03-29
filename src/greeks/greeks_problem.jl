# Exports
export ForwardAD, FiniteDifference, GreekProblem, SecondOrderGreekProblem

# Method types
abstract type GreekMethod end

abstract type FDScheme end

struct FDForward <: FDScheme end
struct FDBackward <: FDScheme end
struct FDCentral <: FDScheme end

struct ForwardAD <: GreekMethod end
struct FiniteDifference{S<:FDScheme} <: GreekMethod
    bump
    scheme::S
end

FiniteDifference(bump) = FiniteDifference(bump, CentralFiniteDifference())

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

function compute_fd_derivative(::ForwardFiniteDifference, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ + ε)
    v_up = solve(prob_up, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v_up - v₀) / ε
end

function compute_fd_derivative(::BackwardFiniteDifference, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_down = set(prob, lens, x₀ - ε)
    v_down = solve(prob_down, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v₀ - v_down) / ε
end

function compute_fd_derivative(::CentralFiniteDifference, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ + ε)
    prob_down = set(prob, lens, x₀ - ε)
    v_up = solve(prob_up, pricing_method).price
    v_down = solve(prob_down, pricing_method).price
    return (v_up - v_down) / (2ε)
end

function solve(gprob::GreekProblem, method::FiniteDifference{S}, pricing_method::P) where {S<:FiniteDifferenceScheme, P<:AbstractPricingMethod}
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
