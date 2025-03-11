module Hedgehog2

using BenchmarkTools, ForwardDiff, Distributions, Accessors

include("payoffs.jl")
include("market_inputs.jl")
include("pricing_methods.jl")
include("delta_methods.jl")

"""Example code with benchmarks"""
function example()
    # Define market data and payoff
    market_inputs = BlackScholesInputs(0.01, 1, 0.4)
    payoff = VanillaEuropeanCall(1, 1)
    pricer = Pricer(market_inputs, payoff, BlackScholesMethod())
    println(pricer())
    # Analytical Delta
    analytical_delta_calc = DeltaCalculator(pricer, BlackScholesAnalyticalDelta())
    println(analytical_delta_calc())

    # AD Delta
    ad_delta_calc = DeltaCalculator(pricer, ADDelta())
    println(ad_delta_calc())

    println("Benchmarking pricer:")
    @btime pricer()

    # Run benchmarks
    println("Benchmarking Analytical Delta:")
    @btime analytical_delta_calc()

    println("Benchmarking AD Delta:")
    @btime ad_delta_calc()
end

example() 

end