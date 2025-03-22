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

# -- Monte Carlo Pricer
trajectories = 10_000
strategy = Hedgehog2.BlackScholesExact(trajectories)
dynamics = Hedgehog2.LognormalDynamics()
method_mc = Hedgehog2.MonteCarlo(dynamics, strategy)
mc_pricer = Pricer(payoff, market_inputs, method_mc)

# -- Analytic Pricer
method_analytic = BlackScholesAnalytic()
analytic_pricer = Pricer(payoff, market_inputs, method_analytic)

# -- Run both and show results
println("Analytic price:  ", analytic_pricer())
println("Monte Carlo price:", mc_pricer())

# -- Benchmark
println("\n--- Benchmarking ---")
@btime $analytic_pricer()
@btime $mc_pricer()
