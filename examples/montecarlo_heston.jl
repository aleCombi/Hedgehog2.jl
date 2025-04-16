using Revise
using Hedgehog
using Distributions
using Random
using Plots
using Dates

println("=== Heston: Carr-Madan vs Euler–Maruyama vs Broadie–Kaya ===")

# --- Market Inputs ---
reference_date = Date(2020, 1, 1)

S0 = 1.0
V0 = 0.010201
κ = 6.21
θ = 0.019
σ = 0.61
ρ = -0.7
r = 0.0319

market_inputs = HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# --- Payoff ---
expiry = reference_date + Year(5)
strike = S0  # ATM put
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# --- Dynamics ---
dynamics = HestonDynamics()

# --- Carr-Madan ---
α = 1.0
boundary = 32.0
carr_madan_method = CarrMadan(α, boundary, dynamics)
carr_madan_problem = PricingProblem(payoff, market_inputs)
carr_madan_solution = solve(carr_madan_problem, carr_madan_method)

# --- Monte Carlo Parameters ---
trajectories = 1_000
steps = 500

# --- Euler–Maruyama ---
euler_strategy = EulerMaruyama(trajectories, steps = steps)
euler_method = MonteCarlo(dynamics, euler_strategy)
euler_problem = PricingProblem(payoff, market_inputs)
euler_solution = solve(euler_problem, euler_method)

# --- Broadie–Kaya ---
bk_strategy = HestonBroadieKaya(trajectories, steps = 1)
bk_method = MonteCarlo(dynamics, bk_strategy)
bk_problem = PricingProblem(payoff, market_inputs)
bk_solution = solve(bk_problem, bk_method)

# --- Results ---
println("\nCarr-Madan price:")
@time println(carr_madan_solution.price)

println("\nEuler–Maruyama price:")
@time println(euler_solution.price)

println("\nBroadie–Kaya price:")
@time println(bk_solution.price)
