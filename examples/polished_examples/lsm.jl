using Revise, Hedgehog, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using Test
using DataFrames
using Random

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 100.0
reference_date = Date(2020, 1, 1)
expiry = reference_date + Year(1)
rate = 0.05
spot = 100.0
sigma = 0.2

# --- American Put Option ---
american_put = VanillaOption(strike, expiry, American(), Put(), Spot())
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
put_prob = PricingProblem(american_put, market_inputs)

# Create deterministic seeds for reproducibility
seeds = Base.rand(UInt64, trajectories)
lsm_config = SimulationConfig(trajectories, steps=steps_lsm, variance_reduction=Hedgehog.Antithetic(), seeds=seeds)
degree = 5  # Polynomial degree for regression
lsm_method = LSM(dynamics, BlackScholesExact(), lsm_config, degree)

# ------------------------------
# Define lenses for Greeks
# ------------------------------
vol_lens = VolLens(1,1)
spot_lens = Hedgehog.SpotLens()
rate_lens = ZeroRateSpineLens(1)
lenses = (spot_lens, vol_lens, rate_lens)

delta_prob = GreekProblem(put_prob, spot_lens)

ad_sol = solve(delta_prob, ForwardAD(), lsm_method)
fd_sol = solve(delta_prob, FiniteDifference(1), lsm_method)

price_sol = solve(put_prob, lsm_method).price

market_inputs_bumped = BlackScholesInputs(reference_date, rate, spot + 1, sigma)
put_prob_bumped = PricingProblem(american_put, market_inputs_bumped)

price_sol_bumped = solve(put_prob_bumped, lsm_method).price

