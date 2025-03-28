export ForwardAD, FiniteDifference, GreekProblem

abstract type GreekMethod end

struct ForwardAD <: GreekMethod end
struct FiniteDifference <: GreekMethod
    bump::Float64
end

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

function solve(gprob::GreekProblem, method::FiniteDifference, pricing_method::P) where P<:AbstractPricingMethod
    prob = gprob.pricing_problem
    lens = gprob.wrt
    ε = method.bump

    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ + ε)
    prob_down = set(prob, lens, x₀ - ε)

    v_up = solve(prob_up, pricing_method).price
    v_down = solve(prob_down, pricing_method).price

    deriv = (v_up - v_down) / (2ε)
    return (greek = deriv,)
end
