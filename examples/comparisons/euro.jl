using Revise, Hedgehog, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using Test
using DataFrames

include("run_model_comparison.jl")

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2020, 1, 2)
underlying = Hedgehog.Spot()
euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# ------------------------------
# Define pricing methods
# ------------------------------
bs_method = BlackScholesAnalytic()
crr_method = CoxRossRubinsteinMethod(800)

# Monte Carlo method with fixed seed for reproducibility
trajectories = 10000
seed = 42
mc_config = SimulationConfig(trajectories, steps=100, seeds=rand(Xoshiro(seed), UInt64, trajectories))
mc_exact_method = MonteCarlo(LognormalDynamics(), BlackScholesExact(), mc_config)

# ------------------------------
# Define lenses for Greeks
# ------------------------------
vol_lens = VolLens(1,1)
spot_lens = @optic _.market_inputs.spot
rate_lens = ZeroRateSpineLens(1)
lenses = (spot_lens, vol_lens, rate_lens)

# ------------------------------
# Run comparison table
# ------------------------------
println("European Put Option (1-day maturity):")
df_put = run_model_comparison_table(
    euro_pricing_prob,
    [bs_method, crr_method, mc_exact_method],
    lenses;
    ad_method = ForwardAD(),
    fd_method = FiniteDifference(1e-4),
    analytic_method = AnalyticGreek(),
)
println(df_put)

# ------------------------------
# Long-dated call option comparison
# ------------------------------
long_expiry = reference_date + Year(5)
call_payoff = VanillaOption(strike, long_expiry, European(), Call(), underlying)
call_pricing_prob = PricingProblem(call_payoff, market_inputs)

println("\nEuropean Call Option (5-year maturity):")
df_call = run_model_comparison_table(
    call_pricing_prob,
    [bs_method, crr_method, mc_exact_method],
    lenses;
    ad_method = ForwardAD(),
    fd_method = FiniteDifference(1e-4),
    analytic_method = AnalyticGreek(),
)
println(df_call)