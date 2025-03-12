import Hedgehog2 as h
using BenchmarkTools

"""Example code with benchmarks"""
# Define market data and payoff
market_inputs = h.BlackScholesInputs(0, 0.4, 1, 0.4)
payoff = h.VanillaEuropeanCall(1, 1)
analytical_pricer = h.Pricer(payoff, market_inputs, h.BlackScholesMethod())
price = h.price(payoff, market_inputs, h.BlackScholesMethod())

crr = h.CoxRossRubinsteinMethod(2000)
crr_pricer = h.Pricer(payoff, market_inputs, crr)
println("Cox Ross Rubinstein:")
println(crr_pricer())

println("Analytical price:")
println(analytical_pricer())
# Analytical Delta
analytical_delta_calc = h.DeltaCalculator(analytical_pricer, h.BlackScholesAnalyticalDelta())
println(analytical_delta_calc())

# AD Delta
ad_delta_calc = h.DeltaCalculator(analytical_pricer, h.ADDelta())
println(ad_delta_calc())

println("Benchmarking pricer:")
@btime analytical_pricer()

# Run benchmarks
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()
