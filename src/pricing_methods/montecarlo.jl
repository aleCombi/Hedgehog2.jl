# a montecarlo method is defined by an SDEProblem or a NoiseProblem.
# the SDE problem needs market data and a payoff to be defined
abstract type MontecarloMethod <: AbstractPricingMethod end

struct BSMontecarlo <: MontecarloMethod
    trajectories
end

# we could make an ExactMontecarlo, dispath to get the noise problem and always make just one step.
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::BlackScholesInputs, method::BSMontecarlo) where {C}
    T = Dates.value.(payoff.expiry .- marketInputs.referenceDate) ./ 365  # Assuming 365-day convention
    noise = GeometricBrownianMotionProcess(market_inputs.r, market_inputs.sigma, market_inputs.referenceDate, 0)
    problem = NoiseProblem(noise, (0, T))
    ensemble_problem = EnsembleProblem(problem)
    solution = solve(ensemble_problem; dt=T, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    mean_payoff = mean(payoff.(last.(solution.u)))
    return mean_payoff
end