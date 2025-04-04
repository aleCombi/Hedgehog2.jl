module Hedgehog2

using DifferentialEquations, ForwardDiff, Distributions, Accessors, Dates

if false
    include("../examples/includer.jl")
end

if false
    include("../test/runtests.jl")
end

# utilities
include("date_functions.jl")
include("solutions/pricing_solutions.jl")

# payoffs
include("payoffs/payoffs.jl")

# market inputs
include("market_inputs/rate_curve.jl")
include("market_inputs/market_inputs.jl")
include("market_inputs/vol_surface.jl")

# pricing methods
include("pricing_methods/pricing_methods.jl")
include("pricing_methods/black_scholes.jl")
include("pricing_methods/cox_ross_rubinstein.jl")
include("pricing_methods/montecarlo.jl")
include("pricing_methods/carr_madan.jl")
include("pricing_methods/least_squares_montecarlo.jl")

# sensitivities
include("greeks/greeks_problem.jl")

# distributions
include("distributions/heston.jl")
include("distributions/sample_from_cf.jl")

# calibration
include("calibration/basket.jl")
include("calibration/calibration.jl")

end
