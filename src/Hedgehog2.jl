module Hedgehog2

using DifferentialEquations, ForwardDiff, Distributions, Dates
using NonlinearSolve, Roots
using Integrals
using Polynomials
using DiffEqNoiseProcess
using Statistics
using DataInterpolations
using Roots
using Random, SpecialFunctions, StaticArrays
using Optimization
using Accessors
import Accessors: set

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
include("market_inputs/vol_surface.jl")
include("market_inputs/rate_curve.jl")
include("market_inputs/market_inputs.jl")

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

# Utilities
export yearfrac, add_yearfrac

# Payoffs
export VanillaOption, European, Spot, Forward, Call, Put, parity_transform

# Market data
export BlackScholesInputs, AbstractMarketInputs, HestonInputs
export RateCurve,
    df,
    zero_rate,
    forward_rate,
    spine_tenors,
    spine_zeros,
    FlatRateCurve,
    is_flat,
    ZeroRateSpineLens
export RectVolSurface, spine_strikes, spine_tenors, spine_vols, get_vol, get_vol_yf, Interpolator2D, to_ticks, VolLens

# Pricers
export PricingProblem, solve
export BlackScholesAnalytic, implied_vol
export CarrMadan
export CoxRossRubinsteinMethod
export MonteCarlo,
    HestonBroadieKaya,
    EulerMaruyama,
    BlackScholesExact,
    LognormalDynamics,
    HestonDynamics
export LSM
export solve_custom_ensemble, CustomEnsembleProblem

# Sensitivities
export ForwardAD, FiniteDifference, GreekProblem, SecondOrderGreekProblem, AnalyticGreek

# Calibration
export RootFinderAlgo, OptimizerAlgo, CalibrationProblem, BasketPricingProblem

end
