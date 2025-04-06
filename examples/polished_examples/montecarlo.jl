using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2020, 1, 2)

euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

dynamics = LognormalDynamics()
trajectories = 1
strategy = BlackScholesExact(trajectories)
montecarlo_method = MonteCarlo(dynamics, strategy)

solution_analytic = solve(euro_pricing_prob, BlackScholesAnalytic()).price
solution = solve(euro_pricing_prob, montecarlo_method).price

@code_warntype solve(euro_pricing_prob, montecarlo_method)