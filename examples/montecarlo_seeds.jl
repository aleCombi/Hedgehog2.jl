using DifferentialEquations

μ = 0.02
σ = 0.04
t0 = 0.0
W0 = 0.0
tspan = (0.0,1.0)
trajectories = 100
proc = GeometricBrownianMotionProcess(μ, σ, t0, W0, nothing)
noise = NoiseProblem(proc, tspan)
seeds = rand(UInt64,trajectories)
prob_func = (prob, i, repeat) -> remake(prob; seed=seeds[i])
problem = EnsembleProblem(noise; prob_func=prob_func)
sol = solve(problem, EM(); trajectories=trajectories, dt=1.0)
