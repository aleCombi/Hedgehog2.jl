using Revise, Hedgehog2, BenchmarkTools

"""Example code with benchmarks"""

# Define market data and payoff
today=0
rate=0
spot=1
sigma=0.4
market_inputs = BlackScholesInputs(today, rate, spot, sigma)
payoff = VanillaCall(1, 1)
analytical_pricer = Pricer(payoff, market_inputs, BlackScholesMethod())
bs_price = compute_price(payoff, market_inputs, BlackScholesMethod())

crr = CoxRossRubinsteinMethod(800)
crr_pricer = Pricer(payoff, market_inputs, crr)
println("Cox Ross Rubinstein:")
println(crr_pricer())

println("Analytical price:")
println(analytical_pricer())
# Analytical Delta
analytical_delta_calc = DeltaCalculator(analytical_pricer, BlackScholesAnalyticalDelta())
println(analytical_delta_calc())

# AD Delta
ad_delta_calc = DeltaCalculator(analytical_pricer, ADDelta())
println(ad_delta_calc())

println("Benchmarking pricer:")
@btime analytical_pricer()

# Run benchmarks
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()
