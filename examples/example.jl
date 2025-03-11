import Hedgehog2 as h
using BenchmarkTools

"""Example code with benchmarks"""
# Define market data and payoff
market_inputs = h.BlackScholesInputs(0.01, 1, 0.4)
payoff = h.VanillaEuropeanCall(1, 1)
pricer = h.Pricer(market_inputs, payoff, h.BlackScholesMethod())
println(pricer())
# Analytical Delta
analytical_delta_calc = h.DeltaCalculator(pricer, h.BlackScholesAnalyticalDelta())
println(analytical_delta_calc())

# AD Delta
ad_delta_calc = h.DeltaCalculator(pricer, h.ADDelta())
println(ad_delta_calc())

println("Benchmarking pricer:")
@btime pricer()

# Run benchmarks
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()
