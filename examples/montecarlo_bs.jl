using Revise, Hedgehog2, BenchmarkTools, Dates

"""Example code with benchmarks"""

# Define market inputs
reference_date = Date(2020, 1, 1)
rate=0
spot=12.0
sigma=0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Define payoff
expiry = Date(2021, 1, 1)
strike = 1.2
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Define montecarlo methods
trajectories = 10000
method = Hedgehog2.MontecarloExact(trajectories, Hedgehog2.LognormalDynamics())

# Define pricer
mc_pricer = Pricer(payoff, market_inputs, method)

# Define analytical pricer
analytical_pricer = Pricer(payoff, market_inputs, BlackScholesAnalytic())

println(analytical_pricer())
println(mc_pricer())