using StochasticDiffEq
using Hedgehog

μ = 0.01
σ = 0.2
W0 = 0.0
problem = Hedgehog.LogGBMProblem(μ, σ, W0, (0.0, 1.0))
# function LogGBMProblem(μ, σ, u0, tspan; seed = UInt64(0), kwargs...)

ensemble_prob = EnsembleProblem(problem)

steps = 100
trajectories = 2
StochasticDiffEq.solve(ensemble_prob, EM(); dt=1.0/steps, trajectories=trajectories)