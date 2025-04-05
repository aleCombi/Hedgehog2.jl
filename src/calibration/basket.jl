struct BasketPricingProblem{P<:AbstractPayoff,M<:AbstractMarketInputs}
    payoffs::Vector{P}
    market_inputs::M
end

struct BasketPricingSolution{P<:AbstractPayoff,M<:AbstractMarketInputs,S}
    problem::BasketPricingProblem{P,M}
    solutions::Vector{S}
end

function solve(prob::BasketPricingProblem, method::M) where {M<:AbstractPricingMethod}
    sols = [solve(PricingProblem(payoff, prob.market_inputs), method) for payoff in prob.payoffs]
    return BasketPricingSolution(prob, sols)
end


