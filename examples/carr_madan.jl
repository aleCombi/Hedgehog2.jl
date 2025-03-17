using Revise, Hedgehog2, BenchmarkTools, Dates

"""Example code with benchmarks"""

# Define market inputs
reference_date = Date(2020, 1, 1)
rate=0
spot=1
sigma=0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Define payoff
expiry = Date(2021, 1, 1)
strike = 1.5
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Define carr madan method
boundary = 32
α = 1
method = Hedgehog2.CarrMadanBS(α, boundary)

# Define pricer
carr_madan_pricer = Pricer(payoff, market_inputs, method)

# Define analytical pricer
analytical_pricer = Pricer(payoff, market_inputs, BlackScholesMethod())

println(analytical_pricer())
println(carr_madan_pricer())