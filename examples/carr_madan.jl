using Revise, Hedgehog2, BenchmarkTools, Dates

# -- Market Inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 100.0
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# -- Payoff
expiry = reference_date + Day(365)
strike = 100.0
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# -- Construct pricing problem
prob = PricingProblem(payoff, market_inputs)

# -- Carr–Madan method
α = 1.0
boundary = 16.0
method_carr_madan = CarrMadan(α, boundary, LognormalDynamics())

# -- Analytic method
method_analytic = BlackScholesAnalytic()

# -- Solve both
sol_carr = solve(prob, method_carr_madan)
sol_analytic = solve(prob, method_analytic)

# -- Print results
println("Analytic price:   ", sol_analytic.price)
println("Carr-Madan price: ", sol_carr.price)

# -- Benchmark
println("\n--- Benchmarking ---")
@btime solve($prob, $method_analytic).price
@btime solve($prob, $method_carr_madan).price
