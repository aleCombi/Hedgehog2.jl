using Revise, Hedgehog, BenchmarkTools, Dates, Random, Accessors

# Define market inputs
strike = 10.0
reference_date = Date(2020, 1, 1)
expiry = reference_date + Year(1)
rate = 0.05
spot = 10.0
sigma = 0.2
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Define payoff
american_payoff = VanillaOption(strike, expiry, American(), Put(), Spot())

# -- Wrap everything into a pricing problem
prob = PricingProblem(american_payoff, market_inputs)

# --- Cox–Ross–Rubinstein using `solve(...)` style
steps_crr = 800
crr_method = CoxRossRubinsteinMethod(steps_crr)
crr_solution = Hedgehog.solve(prob, crr_method)

println("Cox Ross Rubinstein American Price:")
println(crr_solution.price)

# --- LSM using `solve(...)` style
dynamics = LognormalDynamics()
trajectories = 10000
steps_lsm = 100

strategy = BlackScholesExact()
config = Hedgehog.SimulationConfig(trajectories; steps=steps_lsm, variance_reduction=Hedgehog.Antithetic())
degree = 5
lsm_method = LSM(dynamics, strategy, config, degree)
lsm_solution = Hedgehog.solve(prob, lsm_method)

println("LSM American Price:")
println(lsm_solution.price)

euro_prob = @set prob.payoff.exercise_style = European()
black_scholes_method = BlackScholesAnalytic()
bs_solution = Hedgehog.solve(euro_prob, black_scholes_method)
println("Black Scholes European Price:")    
println(bs_solution.price)

@btime Hedgehog.solve(prob, lsm_method)


# Accessors
spot_lens = @optic _.market_inputs.spot
sigma_lens = Hedgehog.VolLens(1,1)

# Methods
fd_method = FiniteDifference(1e-3)
ad = ForwardAD()

println("\n--- Greeks (Analytic Method) ---")
delta_fd = Hedgehog.solve(GreekProblem(prob, spot_lens), fd_method, lsm_method).greek
vega_fd = Hedgehog.solve(GreekProblem(prob, sigma_lens), fd_method, lsm_method).greek
delta_ad = Hedgehog.solve(GreekProblem(prob, spot_lens), ad, lsm_method).greek
vega_ad = Hedgehog.solve(GreekProblem(prob, sigma_lens), ad, lsm_method).greek 
