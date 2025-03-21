using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots, Dates

reference_date = Date(2020,1,1)
# Define Heston model parameters like in Broadie-Kaya
S0 = 1.0   # Initial stock price
V0 = 0.010201  # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0.0319     # Risk-free rate
T = 1.0       # Time to maturity
market_inputs = Hedgehog2.HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# Define option payoff
expiry = reference_date + Day(365)
strike = S0 # ATM call
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Define Carr-Madan pricer as benchmark
boundary = 32
α = 1
method = Hedgehog2.CarrMadan(α, boundary)
carr_madan_pricer = Pricer(payoff, market_inputs, method)
carr_madan_price = carr_madan_pricer()

# Construct the Heston Noise Process
# TODO: this should be embedded in the Montecarlo pricer
heston_dist = Hedgehog2.log_distribution(market_inputs)
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist(T))

# Define `NoiseProblem`
# TODO: this should be embedded in the Montecarlo pricer
trajectories = 10000
problem = NoiseProblem(heston_noise, (0.0, T))
ensemble_problem = EnsembleProblem(problem)
rng = Xoshiro()

heston_dist = Hedgehog2.log_distribution(market_inputs)

function simulated_prices(N)
    @time values = [Hedgehog2.rand(rng, heston_dist(T)) for i in 1:N]
    final_payoffs = exp(-r) .* payoff.(values)
    price = mean(final_payoffs)
    variance = var(final_payoffs)
    return price, variance
end
# @time solution = solve(ensemble_problem; dt=T, trajectories=trajectories)
# final_payoffs = exp(-r) .* payoff.(last.(solution.u))

# TODO: this should be embedded in the Montecarlo pricer

# println("Trajectories: ", trajectories)
# println("Montecarlo price: ", price)
# println("Carr-Madan price: ", carr_madan_price)
# println("Montecarlo variance: ", variance)

# # calculating bias and rms_error to compare with Broadie-Kaya paper results
# bias = price - carr_madan_price
# rms_error = √(bias^2 + variance)
# println("Bias: ", bias)
# println("RMS error: ", rms_error)

using Random, Statistics, Plots

function run_rms_test(N; true_price)
    price, payoff_var = simulated_prices(N)
    bias = price - true_price
    variance_of_estimator = payoff_var / N
    rms_error = sqrt(bias^2 + variance_of_estimator)
    return rms_error, bias, variance_of_estimator
end

using Statistics, Plots

# One RMS estimate per N
function run_rms_test(N; true_price)
    global rng = MersenneTwister(123 + N)  # Different seed per N
    price, payoff_var = simulated_prices(N)
    bias = price - true_price
    estimator_var = payoff_var / N
    rms_error = sqrt(bias^2 + estimator_var)
    return rms_error, bias, estimator_var
end

# Path counts to test
Ns = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

# Replace with actual Heston price from analytic formula
true_price = carr_madan_price

# Storage
rms_errors = Float64[]
biases = Float64[]
variances = Float64[]

for N in Ns
    println("Running for N = $N")
    rms, bias, var = run_rms_test(N; true_price=true_price)
    push!(rms_errors, rms)
    push!(biases, bias)
    push!(variances, var)
end

# Plot
plot(Ns, rms_errors;
    xscale = :log10, yscale = :log10,
    label = "RMS Error",
    lw = 2, marker = :circle,
    xlabel = "Number of Paths (log)",
    ylabel = "Error (log)",
    title = "RMS Convergence (Broadie-Kaya Style)")

plot!(Ns, sqrt.(variances); label = "√Variance", lw = 2, marker = :diamond)
plot!(Ns, abs.(biases); label = "|Bias|", lw = 2, marker = :x)

