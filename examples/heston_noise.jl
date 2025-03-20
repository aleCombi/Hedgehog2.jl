using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots, Dates

reference_date = Date(2020,1,1)
# Define Heston model parameters
S0 = 1.0   # Initial stock price
V0 = 0.010201  # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0.0319     # Risk-free rate
T = 1.0       # Time to maturity
market_inputs = Hedgehog2.HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# Define carr madan method
boundary = 32
α = 1
method = Hedgehog2.CarrMadan(α, boundary)

# Define pricer
expiry = reference_date + Day(365)
strike = 1
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())

carr_madan_pricer = Pricer(payoff, market_inputs, method)
println(carr_madan_pricer())

# Construct the Heston Noise Process
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist)

# Define `NoiseProblem`
trajectories = 1000
problem = NoiseProblem(heston_noise, (0.0, T))
ensemble_problem = EnsembleProblem(problem)
@time solution = solve(ensemble_problem; dt=T, trajectories=trajectories)
final_payoffs = payoff.(last.(solution.u))

price = mean(final_payoffs)
variance = var(final_payoffs)
println("Exact: ", price, " variance: ",variance, " error ")