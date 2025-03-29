using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2021, 1, 1)
underlying = Hedgehog2.Forward()

payoff = VanillaOption(strike, expiry, European(), Call(), underlying)

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
pricing_prob = PricingProblem(payoff, market_inputs)
bs_method = BlackScholesAnalytic()
# Theta (1st order w.r.t. expiry)
# ------------------------------
# note that derivatives are in ticks, hence very small
thetaproblem = Hedgehog2.GreekProblem(pricing_prob, @optic _.payoff.expiry)
theta_ad = solve(thetaproblem, ForwardAD(), bs_method).greek
theta_fd = solve(thetaproblem, FiniteDifference(1), bs_method).greek
theta_an = solve(thetaproblem, AnalyticGreek(), bs_method).greek

println("Theta (Analytic):     $theta_an")
println("Theta (Forward AD):  $theta_ad")
println("Theta (Finite Diff): $theta_fd")
