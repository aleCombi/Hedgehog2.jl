using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots, Dates

reference_date = Date(2020, 1, 1)
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
payoff =
    VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

# Define Carr-Madan pricer as benchmark
boundary = 32
α = 1
distribution = Hedgehog2.HestonDynamics()
method = Hedgehog2.CarrMadan(α, boundary, distribution)
carr_madan_pricer = Pricer(payoff, market_inputs, method)
carr_madan_price = carr_madan_pricer()

# Construct the Heston Noise Process
# TODO: this should be embedded in the Montecarlo pricer
heston_noise = Hedgehog2.HestonNoise(0, dynamics(market_inputs), Z0 = nothing)

# Define `NoiseProblem`
# TODO: this should be embedded in the Montecarlo pricer
trajectories = 10000
problem = NoiseProblem(heston_noise, (0.0, T))
ensemble_problem = EnsembleProblem(problem)
rng = Xoshiro()

using Statistics, Plots, PrettyTables

# Your simulation function
function simulated_prices(N)
    values = [Hedgehog2.rand(rng, heston_dist(T)) for i = 1:N]
    final_payoffs = exp(-r) .* payoff.(values)
    price = mean(final_payoffs)
    payoff_variance = var(final_payoffs)
    return price, payoff_variance
end

function run_rms_test(N; true_price)
    global rng = MersenneTwister(123 + N)
    price, payoff_var = simulated_prices(N)
    bias = price - true_price
    estimator_var = payoff_var / N
    rms_error = sqrt(bias^2 + estimator_var)
    return price, bias, estimator_var, rms_error
end

# Parameters
Ns = [10_000, 40_000, 160_000, 640_000, 2_560_000, 10_240_000]# , 20_000, 50_000, 100_000]
true_price = carr_madan_price # Replace with your analytic price

# Collect results
results = zeros(Float64, length(Ns), 5)  # [N, Price, Bias, Var/N, RMS]

# Header
println(
    rpad("N", 10),
    rpad("Price", 15),
    rpad("Bias", 15),
    rpad("Variance/N", 20),
    rpad("RMS Error", 15),
    "Time (s)",
)
println("="^85)

for (i, N) in enumerate(Ns)
    t = @elapsed begin
        price, bias, varN, rms = run_rms_test(N; true_price = true_price)
        results[i, :] = [N, price, bias, varN, rms]
    end

    line =
        rpad(string(N), 10) *
        rpad(string(round(price, digits = 6)), 15) *
        rpad(string(round(bias, digits = 6)), 15) *
        rpad(string(round(varN, sigdigits = 3)), 20) *
        rpad(string(round(rms, sigdigits = 3)), 15) *
        string(round(t, digits = 2))
    println(line)
end


# Plot
plot(
    Ns,
    results[:, 5];
    xscale = :log10,
    yscale = :log10,
    label = "RMS Error",
    lw = 2,
    marker = :circle,
    xlabel = "Number of Paths (log)",
    ylabel = "Error (log)",
    title = "RMS Convergence (Broadie-Kaya Style)",
)
plot!(Ns, sqrt.(results[:, 4]); label = "√Variance", lw = 2, marker = :diamond)
plot!(Ns, abs.(results[:, 3]); label = "|Bias|", lw = 2, marker = :x)
