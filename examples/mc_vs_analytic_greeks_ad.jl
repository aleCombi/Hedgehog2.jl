using Hedgehog2
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
spot_lens = @optic _.market.spot
rate_lens = ZeroRateSpineLens(1)

# --------------------------
# Methods
# --------------------------
trajectories = 100_000
mc_method = MonteCarlo(LognormalDynamics(), BlackScholesExact(trajectories))
analytic_method = BlackScholesAnalytic()
ε = 1e-4

# --------------------------
# Price Comparison
# --------------------------
price_mc = solve(prob, mc_method).price
price_an = solve(prob, analytic_method).price

# --------------------------
# Delta
# --------------------------
gprob = GreekProblem(prob, spot_lens)
delta_mc = solve(gprob, ForwardAD(), mc_method).greek
delta_an = solve(gprob, AnalyticGreek(), analytic_method).greek

# --------------------------
# Gamma
# --------------------------
gprob2 = SecondOrderGreekProblem(prob, spot_lens, spot_lens)
gamma_mc = solve(gprob2, FiniteDifference(1E-1), mc_method).greek
gamma_an = solve(gprob2, AnalyticGreek(), analytic_method).greek

# --------------------------
# Vega
# --------------------------
gprob = GreekProblem(prob, vol_lens)
vega_mc = solve(gprob, ForwardAD(), mc_method).greek
vega_an = solve(gprob, AnalyticGreek(), analytic_method).greek

# --------------------------
# Rho (flat curve, first zero rate)
# --------------------------
gprob = GreekProblem(prob, rate_lens)
rho_mc = solve(gprob, ForwardAD(), mc_method).greek
rho_an = solve(gprob, ForwardAD(), analytic_method).greek

# --------------------------
# Print Report
# --------------------------
println("---- Black-Scholes Pricing & Greeks ----\n")
@printf("Price (Analytic)      = %.6f\n", price_an)
@printf("Price (Monte Carlo)   = %.6f\n", price_mc)
@printf("RelError              = %.4e\n\n", abs(price_mc - price_an) / price_an)

@printf("Delta  (Analytic)     = %.6f\n", delta_an)
@printf("Delta  (Monte Carlo)  = %.6f\n", delta_mc)
@printf("RelError              = %.4e\n\n", abs(delta_mc - delta_an) / delta_an)

@printf("Gamma  (Analytic)     = %.6f\n", gamma_an)
@printf("Gamma  (Monte Carlo)  = %.6f\n", gamma_mc)
@printf("RelError              = %.4e\n\n", abs(gamma_mc - gamma_an) / gamma_an)

@printf("Vega   (Analytic)     = %.6f\n", vega_an)
@printf("Vega   (Monte Carlo)  = %.6f\n", vega_mc)
@printf("RelError              = %.4e\n\n", abs(vega_mc - vega_an) / vega_an)

@printf("Rho    (Analytic)     = %.6f\n", rho_an)
@printf("Rho    (Monte Carlo)  = %.6f\n", rho_mc)
@printf("RelError              = %.4e\n", abs(rho_mc - rho_an) / rho_an)
