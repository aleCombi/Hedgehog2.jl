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
strike = 1.2
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Create pricer
analytical_pricer = Pricer(payoff, market_inputs, BlackScholesMethod())
println("Analytical price: ", analytical_pricer())

# Analytical Delta
analytical_delta_calc = DeltaCalculator(analytical_pricer, BlackScholesAnalyticalDelta())
println("Analytical delta: ", analytical_delta_calc())

# AD Delta
ad_delta_calc = DeltaCalculator(analytical_pricer, ADDelta())
println("AD Delta: ", ad_delta_calc())

println("Benchmarking pricer:")
@btime analytical_pricer()

# Run benchmarks
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()
