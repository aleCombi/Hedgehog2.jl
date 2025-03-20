using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots

# Define Heston model parameters
S0 = 1.0    # Initial stock price
V0 = 0.010201    # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0.0319      # Risk-free rate
T = 1.0     # Time to maturity

# Create the exact sampling Heston distribution
heston_dist = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)

rng = Xoshiro()
VT = Hedgehog2.sample_V_T(rng, heston_dist)
phi(u) = Hedgehog2.integral_var_char(u, VT, heston_dist)
F = Hedgehog2.integral_V_cdf(VT, rng, heston_dist)

