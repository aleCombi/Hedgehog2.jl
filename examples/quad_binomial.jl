using Revise, Hedgehog2, BenchmarkTools, Dates

# define payoff
strike = 1.2
expiry = Date(2021, 1, 1)
american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Put(), Spot())

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1.0
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# create Cox Ross Rubinstein pricer
steps = 800
crr = CoxRossRubinsteinMethod(steps)
crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

# LSM pricer
dynamics = LognormalDynamics()
trajectories = 10000
quad = QUAD(5.0, 1000, 1000, dynamics)
quad_pricer = Pricer(american_payoff, market_inputs, quad)

println("Cox Ross Rubinstein American Price:")
println(crr_american_pricer())

println("Quad Price:")
println(quad_pricer())