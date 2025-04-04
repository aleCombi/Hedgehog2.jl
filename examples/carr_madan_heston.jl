using Revise, Hedgehog2, BenchmarkTools, Dates

"""Example code with benchmarks"""

# Define market inputs
reference_date = Date(2020, 1, 1)

# Define Heston model parameters
S0 = 100    # Initial stock price
V0 = 0.010201    # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0.0319      # Risk-free rate
T = 1.0       # Time to maturity
market_inputs = Hedgehog2.HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)
bs_market_inputs = BlackScholesInputs(reference_date, r, S0, sqrt(V0))
# Define payoff
expiry = reference_date + Day(365)
strike = 100
payoff =
    VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Define carr madan method
boundary = 32
α = 1
method_heston = Hedgehog2.CarrMadan(α, boundary, HestonDynamics())

# Define pricer

println(solve(payoff, market_inputs, method_heston).price)

