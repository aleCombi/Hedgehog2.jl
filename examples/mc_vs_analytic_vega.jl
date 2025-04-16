using Hedgehog
using Accessors
using Dates
using Printf

# --------------------------
# Setup
# --------------------------
strike = 1.0
expiry = Date(2021, 1, 1)
reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

underlying = Spot()
payoff = VanillaOption(strike, expiry, European(), Call(), underlying)
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
prob = PricingProblem(payoff, market_inputs)

# --------------------------
# Lenses
# --------------------------
vol_lens = @optic _.market.sigma

# --------------------------
# Monte Carlo Method (with fixed seeds)
# --------------------------
trajectories = 10_000
mc_method_seeded = MonteCarlo(LognormalDynamics(), BlackScholesExact(trajectories))

# --------------------------
# Monte Carlo Method (naive: completely independent)
# --------------------------
ε = 1e-4
prob_up_naive = @set prob.market.sigma = sigma + ε
prob_dn_naive = @set prob.market.sigma = sigma - ε
method_naive_up = MonteCarlo(LognormalDynamics(), BlackScholesExact(trajectories))
method_naive_dn = MonteCarlo(LognormalDynamics(), BlackScholesExact(trajectories))

# --------------------------
# Analytic Method
# --------------------------
analytic_method = BlackScholesAnalytic()

# --------------------------
# Compute Greeks
# --------------------------
gprob = GreekProblem(prob, vol_lens)
vega_mc_fd_seeded = solve(gprob, FiniteDifference(ε), mc_method_seeded).greek
vega_analytic = solve(gprob, AnalyticGreek(), analytic_method).greek

price_up = solve(prob_up_naive, method_naive_up).price
price_dn = solve(prob_dn_naive, method_naive_dn).price
vega_naive = (price_up - price_dn) / (2ε)

# --------------------------
# Compute Prices
# --------------------------
price_analytic = solve(prob, analytic_method).price
price_mc_seeded = solve(prob, mc_method_seeded).price
price_mc_naive =
    solve(prob, MonteCarlo(LognormalDynamics(), BlackScholesExact(trajectories))).price

# --------------------------
# Print Comparison
# --------------------------
println("Greek Comparison: Vega (Black-Scholes)")
@printf("Analytic Vega        = %.6f\n", vega_analytic)
@printf("Monte Carlo (Seeded) = %.6f\n", vega_mc_fd_seeded)
@printf("Monte Carlo (Naive)  = %.6f\n", vega_naive)
@printf(
    "RelError Seeded      = %.4e\n",
    abs(vega_mc_fd_seeded - vega_analytic) / vega_analytic
)
@printf("RelError Naive       = %.4e\n", abs(vega_naive - vega_analytic) / vega_analytic)

println("\nPrice Comparison:")
@printf("Analytic Price       = %.6f\n", price_analytic)
@printf("Monte Carlo (Seeded) = %.6f\n", price_mc_seeded)
@printf("Monte Carlo (Naive)  = %.6f\n", price_mc_naive)
@printf(
    "RelError Seeded      = %.4e\n",
    abs(price_mc_seeded - price_analytic) / price_analytic
)
@printf(
    "RelError Naive       = %.4e\n",
    abs(price_mc_naive - price_analytic) / price_analytic
)
