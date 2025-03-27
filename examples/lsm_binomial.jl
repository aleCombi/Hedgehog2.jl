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

# -- Wrap everything into a pricing problem
prob = PricingProblem(american_payoff, market_inputs)

# --- Cox–Ross–Rubinstein using `solve(...)` style
steps_crr = 80
crr_method = CoxRossRubinsteinMethod(steps_crr)
crr_solution = solve(prob, crr_method)

println("Cox Ross Rubinstein American Price:")
println(crr_solution.price)

# --- LSM using `solve(...)` style
dynamics = LognormalDynamics()
trajectories = 1000
steps_lsm = 100

strategy = BlackScholesExact(trajectories, steps_lsm)
degree = 3
lsm_method = LSM(dynamics, strategy, degree)
lsm_solution = solve(prob, lsm_method)

println("LSM American Price:")
println(lsm_solution.price)
