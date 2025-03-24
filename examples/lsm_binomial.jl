using Revise, Hedgehog2, BenchmarkTools, Dates, Random, SciMLBase

# define payoff
strike = 1.0
expiry = Date(2021, 1, 1)
american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Call(), Spot())

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1.0
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# create Cox Ross Rubinstein pricer
steps = 80
crr = CoxRossRubinsteinMethod(steps)
crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

# LSM pricer
dynamics = LognormalDynamics()
trajectories = 1
steps = 100
using StableRNGs
rng = StableRNG(42)  # drop-in replacement for MersenneTwister

strategy = BlackScholesExact(trajectories, steps; rng=rng, ensemblealg=EnsembleSerial())
degree = 3
lsm = LSM(dynamics, strategy, degree)
lsm_american_pricer = Pricer(american_payoff, market_inputs, lsm)

println("Cox Ross Rubinstein American Price:")
println(crr_american_pricer())

println("LSM American Price:")
println(lsm_american_pricer())
