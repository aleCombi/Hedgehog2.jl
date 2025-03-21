# a montecarlo method is defined by an SDEProblem or a NoiseProblem.
# the SDE problem needs market data and a payoff to be defined
abstract type MontecarloMethod <: AbstractPricingMethod end

struct MontecarloExact <: MontecarloMethod
    trajectories
    distribution
    kwargs
end

MontecarloExact(trajectories, distribution; kwargs...) = MontecarloExact(trajectories, distribution, Dict(kwargs...))

# log price distribution must be specified
log_distribution(m::MontecarloExact) = m.distribution

# we could make an ExactMontecarlo, dispath to get the noise problem and always make just one step. Uses antithetic variates.
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::M) where {C, I <: AbstractMarketInputs, M<:MontecarloMethod}
    T = Dates.value.(payoff.expiry .- market_inputs.referenceDate) ./ 365  # Assuming 365-day convention
    distribution = log_distribution(method)
    noise = distribution(market_inputs)
    problem = NoiseProblem(noise, (0, T))
    solution = solve(EnsembleProblem(problem); dt=T, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    final_payoffs = payoff.(last.(solution.u))
    mean_payoff = mean(final_payoffs)
    return mean_payoff
end