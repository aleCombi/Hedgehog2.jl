using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots, Dates

reference_date = Date(2020,1,1)
# Define Heston model parameters
S0 = 1.0   # Initial stock price
V0 = 0.010201  # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0 #0.0319     # Risk-free rate
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

boundary = 32
α = 1
method = Hedgehog2.CarrMadan(α, boundary)

carr_madan_pricer = Pricer(payoff, market_inputs, method)
println(carr_madan_pricer())

# Create the exact sampling Heston distribution
heston_dist = Hedgehog2.log_distribution(market_inputs)(1)

# Construct the Heston Noise Process
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist)

# Define `NoiseProblem`
problem = NoiseProblem(heston_noise, (0.0, T))
trajectories = 100000

rng = Xoshiro()
@time x = [Hedgehog2.rand(rng, heston_dist) for x in 1:trajectories]
histogram(x, bins=30, normalize=true, title="Histogram of data") |> display

# Solve with multiple trajectories
# @time solution_exact = solve(EnsembleProblem(problem), dt=T, trajectories=trajectories)

final_prices_exact = [a[end] for a in x]  # Transform log-prices to prices

# println("Euler: ", mean(max.(final_prices_2 .- 1, 0)), " ", var(final_prices_2))
price = mean(max.(final_prices_exact.- strike, 0))
variance = var(final_prices_exact)
println("Exact: ", price, " variance: ",variance, " error ")