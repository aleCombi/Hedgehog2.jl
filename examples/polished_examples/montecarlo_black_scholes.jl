using Revise, Hedgehog, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using DifferentialEquations

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2021, 1, 1)

euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 0.04

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

dynamics = LognormalDynamics()
trajectories = 10000
config = Hedgehog.SimulationConfig(trajectories; steps=100, variance_reduction=Hedgehog.NoVarianceReduction())
strategy = EulerMaruyama()
montecarlo_method = MonteCarlo(dynamics, strategy, config)

solution_analytic = Hedgehog.solve(euro_pricing_prob, BlackScholesAnalytic()).price
solution = Hedgehog.solve(euro_pricing_prob, montecarlo_method).price

@btime Hedgehog.solve($euro_pricing_prob, $montecarlo_method).price
@btime Hedgehog.solve($euro_pricing_prob, BlackScholesAnalytic()).price

fd_method = FiniteDifference(1E-4, Hedgehog.FDForward())
ad_method = ForwardAD()

spot_lens = @optic _.market_inputs.spot
delta_prob = Hedgehog.GreekProblem(euro_pricing_prob, spot_lens)
solve(delta_prob, ad_method, BlackScholesAnalytic())
@btime solve(delta_prob, fd_method, montecarlo_method)
@btime solve(delta_prob, ad_method, montecarlo_method)

rate_greek_prob = GreekProblem(euro_pricing_prob, ZeroRateSpineLens(1))
solve(rate_greek_prob, ad_method, montecarlo_method)
@btime solve($rate_greek_prob, $ad_method, $montecarlo_method)
@btime solve($rate_greek_prob, $fd_method, $montecarlo_method)
