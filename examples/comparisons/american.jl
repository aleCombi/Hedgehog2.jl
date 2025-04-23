using Revise, Hedgehog, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using Test
using DataFrames
using Random

include("run_model_comparison.jl")

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

# ------------------------------
# Define pricing methods
# ------------------------------
# Binomial Tree Method
steps_crr = 800
crr_method = CoxRossRubinsteinMethod(steps_crr)

# LSM Monte Carlo Method
dynamics = LognormalDynamics()
trajectories = 10000
steps_lsm = 100
seed = 42

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

# ------------------------------
# Run comparison table for Put
# ------------------------------
println("American Put Option (1-year maturity):")
df_put = run_model_comparison_table(
    put_prob,
    [crr_method, lsm_method],
    lenses;
    ad_method = ForwardAD(),
    fd_method = FiniteDifference(1e-2),
    analytic_method = nothing,  # No analytic for American
    use_belapsed = true
)
println(df_put)

# ------------------------------
# American Call Option
# ------------------------------
american_call = VanillaOption(strike, expiry, American(), Call(), Spot())
call_prob = PricingProblem(american_call, market_inputs)

# ------------------------------
# Define ITM American Call for dividend case
# ------------------------------

# Create a new market input with effective rate (r-q)
dividend_market = BlackScholesInputs(reference_date, rate, spot, sigma)
american_call_div = VanillaOption(strike * 0.8, expiry, American(), Call(), Spot()) # ITM call
div_call_prob = PricingProblem(american_call_div, dividend_market)

println("\nAmerican Call Option (1-year maturity, ITM):")
df_call_div = run_model_comparison_table(
    div_call_prob,
    [crr_method, lsm_method],
    lenses;
    ad_method = ForwardAD(),
    fd_method = FiniteDifference(1e-2),
    analytic_method = nothing, # No analytic for American
    use_belapsed = true

)
println(df_call_div)