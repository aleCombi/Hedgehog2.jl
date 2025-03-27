using Revise, Hedgehog2, BenchmarkTools, Dates, Random, SciMLBase

# Define payoff
strike = 1.0
expiry = Date(2021, 1, 1)
american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Call(), Spot())

# Define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1.0
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# --- Cox–Ross–Rubinstein (unchanged) ---
steps = 80
crr = CoxRossRubinsteinMethod(steps)
crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

println("Cox Ross Rubinstein American Price:")
println(crr_american_pricer())

# --- LSM using new `solve(...)` style ---
dynamics = LognormalDynamics()
trajectories = 1000
steps = 100

strategy = BlackScholesExact(trajectories, steps)
degree = 3
lsm = LSM(dynamics, strategy, degree)

prob = PricingProblem(american_payoff, market_inputs)
lsm_solution = solve(prob, lsm)

println("LSM American Price:")
println(lsm_solution.price)
