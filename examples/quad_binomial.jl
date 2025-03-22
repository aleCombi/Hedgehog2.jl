using Revise, Hedgehog2, BenchmarkTools, Dates

# define payoff
strike = 1.0
expiry = Date(2021, 1, 1)
american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Put(), Spot())

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.02
spot = 1.0
sigma = 0.04
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# create Cox Ross Rubinstein pricer
steps = 1000
crr = CoxRossRubinsteinMethod(steps)
crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

# LSM pricer
dynamics = LognormalDynamics()
trajectories = 10000
n_grid = 1000
n_exercise = 100
quad = QUAD(5, n_grid, n_exercise, dynamics)
quad_pricer = Pricer(american_payoff, market_inputs, quad)

println("Cox Ross Rubinstein American Price:")
println(crr_american_pricer())

println("Quad Price:")
println(quad_pricer())