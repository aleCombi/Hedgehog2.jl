module Hedgehog2

using BenchmarkTools, ForwardDiff, Distributions, Accessors, Dates

if false 
    include("../examples/includer.jl")
end

if false
    include("../test/runtests.jl")
end

# payoffs
include("payoffs/payoffs.jl")

# market inputs
include("market_inputs/market_inputs.jl")

# pricing methods
include("pricing_methods/pricing_methods.jl")
include("pricing_methods/black_scholes.jl")
include("pricing_methods/cox_ross_rubinstein.jl")
include("pricing_methods/montecarlo.jl")
include("pricing_methods/carr_madan.jl")
include("pricing_methods/least_squares_montecarlo.jl")

# sensitivities
include("sensitivity_methods/delta_methods.jl")

# distributions
include("distributions/heston.jl")
include("distributions/sample_from_cf.jl")

end