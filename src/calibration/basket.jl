""
"""
    BasketPricingProblem(payoffs, market_inputs)

Container for pricing several payoffs under a single market scenario.

# Fields
- `payoffs::Vector{P}`       — collection of payoffs to be priced.
- `market_inputs::M`         — market data (yield curves, vols, etc.) shared by all payoffs.
"""
struct BasketPricingProblem{P<:AbstractPayoff,M<:AbstractMarketInputs}
    payoffs::Vector{P}
    market_inputs::M
end

"""
    BasketPricingSolution(problem, solutions)

Result of solving a `BasketPricingProblem`.

# Fields
- `problem::BasketPricingProblem` — the original problem definition.
- `solutions::Vector{S}`          — pricing results, one per payoff.
"""
struct BasketPricingSolution{P<:AbstractPayoff,M<:AbstractMarketInputs,S}
    problem::BasketPricingProblem{P,M}
    solutions::Vector{S}
end

"""
    solve(prob::BasketPricingProblem, method)

Price every payoff in `prob` with the given pricing `method` and return
a `BasketPricingSolution` collecting the individual results.
"""
function solve(prob::BasketPricingProblem, method::M) where {M<:AbstractPricingMethod}
    sols = [solve(PricingProblem(payoff, prob.market_inputs), method) for payoff in prob.payoffs]
    return BasketPricingSolution(prob, sols)
end
