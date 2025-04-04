using Revise
using Hedgehog2
using Dates
using Statistics
using Random
using Printf

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
function run_mc_trials(n_trials, trajectories, steps; use_antithetic = false)
    prices = Float64[]
    times = Float64[]
    for trial = 1:n_trials
        # Create Euler-Maruyama strategy
        strategy = EulerMaruyama(trajectories, steps; antithetic = use_antithetic)
        method = MonteCarlo(HestonDynamics(), strategy)

        # Measure time and solve
        time = @elapsed begin
            sol = solve(prob, method)
            push!(prices, sol.price)
        end

        push!(times, time)
    end

    return prices, times
end

# --- Run experiments ---
n_trials = 20
trajectories = 5_000
steps = 100

println("Running experiments with $n_trials trials, $trajectories paths, $steps steps...")

# Standard Monte Carlo
std_prices, std_times = run_mc_trials(n_trials, trajectories, steps, use_antithetic = false)

# Antithetic Monte Carlo (same number of total paths)
anti_prices, anti_times =
    run_mc_trials(n_trials, trajectories ÷ 2, steps, use_antithetic = true)

# --- Compute metrics ---
std_mean = mean(std_prices)
std_error = std_mean - reference_price
std_stdev = std(std_prices)
std_rmse = sqrt(mean((std_prices .- reference_price) .^ 2))
std_time = mean(std_times)

anti_mean = mean(anti_prices)
anti_error = anti_mean - reference_price
anti_stdev = std(anti_prices)
anti_rmse = sqrt(mean((anti_prices .- reference_price) .^ 2))
anti_time = mean(anti_times)

# --- Print results ---
println("\n=== Results ===")
println("Metric               | Standard MC       | Antithetic MC      | Improvement")
println("--------------------+-------------------+--------------------+-------------")
@printf("Mean price           | %.6f          | %.6f          | -\n", std_mean, anti_mean)
@printf(
    "Bias                 | %.6f          | %.6f          | %.2fx\n",
    std_error,
    anti_error,
    abs(std_error) / max(abs(anti_error), 1e-10)
)
@printf(
    "Standard deviation   | %.6f          | %.6f          | %.2fx\n",
    std_stdev,
    anti_stdev,
    std_stdev / anti_stdev
)
@printf(
    "RMSE                 | %.6f          | %.6f          | %.2fx\n",
    std_rmse,
    anti_rmse,
    std_rmse / anti_rmse
)
@printf(
    "Avg. execution time  | %.3f s           | %.3f s           | %.2fx\n",
    std_time,
    anti_time,
    std_time / anti_time
)

# --- Path distribution visualization (code) ---
println(
    "\nNote: To visualize the distribution of prices across trials, you can plot a histogram",
)
println(
    "of the price vectors 'std_prices' and 'anti_prices' to see the variance reduction effect.",
)
