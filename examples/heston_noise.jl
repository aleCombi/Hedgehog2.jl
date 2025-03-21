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
trajectories = 100000
problem = NoiseProblem(heston_noise, (0.0, T))
ensemble_problem = EnsembleProblem(problem)
@time solution = solve(ensemble_problem; dt=T, trajectories=trajectories)
final_payoffs = payoff.(last.(solution.u))

# TODO: this should be embedded in the Montecarlo pricer
price = mean(final_payoffs)
variance = var(final_payoffs)
println("Montecarlo price: ", price)
println("Carr-Madan price: ", carr_madan_price)
println("Montecarlo variance: ",variance)

# calculating bias and rms_error to compare with Broadie-Kaya paper results
bias = price - carr_madan_price
rms_error = √(bias^2 + variance)
println("Bias: ", bias)
println("RMS error: ", rms_error)