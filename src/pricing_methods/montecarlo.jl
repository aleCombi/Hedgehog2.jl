struct Montecarlo <: AbstractPricingMethod
    trajectories
    dynamics
    kwargs
end

Montecarlo(trajectories, distribution; kwargs...) = Montecarlo(trajectories, distribution, Dict(kwargs...))

# log price distribution must be specified
log_dynamics(m::Montecarlo) = m.dynamics

# we could make an ExactMontecarlo, dispath to get the noise problem and always make just one step. Uses antithetic variates.
function compute_price(payoff::VanillaOption{European, C, Spot}, market_inputs::I, method::Montecarlo) where {C, I <: AbstractMarketInputs}
    T = Dates.value.(payoff.expiry .- market_inputs.referenceDate) ./ 365  # Assuming 365-day convention
    distribution = log_dynamics(method)
    problem = sde_problem(market_inputs, distribution, method, (0,T))
    solution = solve(EnsembleProblem(problem); dt=T, trajectories = method.trajectories) # its an exact simulation, hence we use just one step
    final_payoffs = payoff.(last.(solution.u))
    mean_payoff = mean(final_payoffs)
    println(sqrt(var(final_payoffs) / length(final_payoffs)))
    return exp(-market_inputs.rate*T) * mean_payoff
end