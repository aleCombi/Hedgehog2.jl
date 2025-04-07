using Revise, Hedgehog2, BenchmarkTools, Dates
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
strategy = EulerMaruyama(trajectories; steps=100, variance_reduction=Hedgehog2.Antithetic())
montecarlo_method = MonteCarlo(dynamics, strategy)

solution_analytic = Hedgehog2.solve(euro_pricing_prob, BlackScholesAnalytic()).price
solution = Hedgehog2.solve(euro_pricing_prob, montecarlo_method)

@btime Hedgehog2.solve($euro_pricing_prob, $montecarlo_method).price
@btime Hedgehog2.solve($euro_pricing_prob, BlackScholesAnalytic()).price

spot_lens = @optic _.market_inputs.spot
delta_prob = Hedgehog2.GreekProblem(euro_pricing_prob, spot_lens)