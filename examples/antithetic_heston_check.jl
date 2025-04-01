using Revise
using Hedgehog2
using Dates
using Statistics
using Random
using Printf
using Plots
println("=== Heston Model: Monte Carlo with Antithetic Variates ===\n")

# --- Market Inputs ---
reference_date = Date(2020, 1, 1)

# Heston model parameters
S0 = 100.0        # Initial spot price
V0 = 0.04         # Initial variance
κ = 2.0           # Mean reversion speed
θ = 0.04          # Long-term variance
σ = 0.3           # Volatility of variance (vol-of-vol)
ρ = -0.7          # Correlation between asset and variance processes
r = 0.05          # Risk-free rate

market_inputs = HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# --- Payoff ---
expiry = reference_date + Year(1)
strike = S0  # ATM European call
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# --- Pricing problem ---
prob = PricingProblem(payoff, market_inputs)

# --- Reference price (using Carr-Madan Fourier method) ---
carr_madan_method = CarrMadan(1.0, 32.0, HestonDynamics())
carr_madan_solution = solve(prob, carr_madan_method)
reference_price = carr_madan_solution.price

println("Reference price (Carr-Madan): $reference_price\n")

# --- Function to run Monte Carlo trials ---
# Create Euler-Maruyama strategy
trajectories = 5000
steps = 100
strategy = EulerMaruyama(trajectories, steps; antithetic=true)
method = MonteCarlo(HestonDynamics(), strategy)
sol = solve(prob, method)

first_path = sol.ensemble.solutions[1]
first_path_u = [first_path.u[i][1] for i in range(1,length(first_path))]
plot(first_path.t, first_path_u)

anti_path = sol.ensemble.solutions[5001]
anti_path_u = [anti_path.u[i][1] for i in range(1,length(first_path))]
plot!(first_path.t, anti_path_u)