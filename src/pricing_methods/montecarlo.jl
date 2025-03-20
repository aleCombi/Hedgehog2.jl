# a montecarlo method is defined by an SDEProblem or a NoiseProblem.
# the SDE problem needs market data and a payoff to be defined
abstract type MontecarloMethod <: AbstractPricingMethod end

struct MontecarloExact <: MontecarloMethod
    trajectories
end

# we could make an ExactMontecarlo, dispath to get the noise problem and always make just one step. Uses antithetic variates.
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::MontecarloExact) where {C, I <: AbstractMarketInputs}
    T = Dates.value.(payoff.expiry .- market_inputs.referenceDate) ./ 365  # Assuming 365-day convention
    spot = market_inputs.forward * exp(- market_inputs.rate * T)

    noise = price_process(market_inputs)
    problem = NoiseProblem(noise, (0, T))
    solution = solve(EnsembleProblem(problem); dt=T, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    
    antithetic_noise = GeometricBrownianMotionProcess(market_inputs.rate, -market_inputs.sigma, 0.0, 1.0)
    antithetic_problem = NoiseProblem(antithetic_noise, (0, T), seed=problem.seed)
    antithetic_solution = solve(EnsembleProblem(antithetic_problem); dt=T, trajectories = method.trajectories, seed=problem.seed) # its an exact simulation, hence we use just one step
    
    final_payoffs = (payoff.(last.(solution.u)) + payoff.(last.(antithetic_solution.u))) * 0.5
    mean_payoff = mean(final_payoffs)
    var_payoff = var(final_payoffs)
    return mean_payoff
end