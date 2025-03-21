module Hedgehog2

using BenchmarkTools, ForwardDiff, Distributions, Accessors, Dates

if false 
    include("../examples/includer.jl")
end

include("payoffs/payoffs.jl")

include("market_inputs/market_inputs.jl")

include("pricing_methods/pricing_methods.jl")
include("pricing_methods/black_scholes.jl")
include("pricing_methods/cox_ross_rubinstein.jl")
include("pricing_methods/montecarlo.jl")
include("pricing_methods/carr_madan.jl")
include("pricing_methods/least_squares_montecarlo.jl")
include("pricing_methods/quad_american.jl")

include("sensitivity_methods/delta_methods.jl")

include("distributions/heston.jl")
include("distributions/sample_from_cf.jl")

end