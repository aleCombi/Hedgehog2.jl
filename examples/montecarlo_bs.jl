using Revise
using Hedgehog2
using BenchmarkTools
using Dates

println("=== European Call Option under Black-Scholes ===")

# -- Market Inputs
reference_date = Date(2020, 1, 1)
expiry         = Date(2021, 1, 1)
spot           = 12.0
sigma          = 0.4
rate           = 0.0

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# -- Payoff
strike = 1.2
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# -- Construct Pricing Problem
prob = PricingProblem(payoff, market_inputs)

# -- Monte Carlo Method
trajectories = 10_000
strategy = BlackScholesExact(trajectories)
dynamics = LognormalDynamics()
method_mc = MonteCarlo(dynamics, strategy)

# -- Analytic Method
method_analytic = BlackScholesAnalytic()

# -- Solve Both
solution_analytic = solve(prob, method_analytic)
solution_mc = solve(prob, method_mc)

println("Analytic price:   ", solution_analytic.price)
println("Monte Carlo price:", solution_mc.price)

# -- Benchmark
println("\n--- Benchmarking ---")
@btime solve($prob, $method_analytic).price
@btime solve($prob, $method_mc).price
