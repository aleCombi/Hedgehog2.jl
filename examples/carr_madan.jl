using Revise, Hedgehog2, BenchmarkTools, Dates

"""Example code with benchmarks"""

# Define market inputs
reference_date = Date(2020, 1, 1)
rate=0.2
spot=100
sigma=0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Define payoff
expiry = reference_date + Day(365)
strike = 100
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Define carr madan method
boundary = 16
α = 1
method = Hedgehog2.CarrMadan(α, boundary, LognormalDynamics())

# Define pricer
carr_madan_pricer = Pricer(payoff, market_inputs, method)

# Define analytical pricer
analytical_pricer = Pricer(payoff, market_inputs, BlackScholesAnalytic())

println(analytical_pricer())
println(carr_madan_pricer())