using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots, Dates

reference_date = Date(2020,1,1)
# Define Heston model parameters like in Broadie-Kaya
# S0 = 1.0   # Initial stock price
# V0 = 0.010201  # Initial variance
# κ = 6.21      # Mean reversion speed
# θ = 0.019      # Long-run variance
# σ = 0.61   # Volatility of variance
# ρ = -0.7     # Correlation
# r = 0.0319     # Risk-free rate
# T = 1.0       # Time to maturity

S0 = 1.0   # Initial stock price
V0 = 0.09  # Initial variance
κ = 2.0      # Mean reversion speed
θ = 0.09      # Long-run variance
σ = 1.00   # Volatility of variance
ρ = -0.3     # Correlation
r = 0.05     # Risk-free rate
T = 5.0       # Time to maturity

market_inputs = Hedgehog2.HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# Define option payoff
expiry = reference_date + Day(5*365)
strike = S0 # ATM call
payoff = VanillaOption(strike, expiry, Hedgehog2.European(), Hedgehog2.Put(), Hedgehog2.Spot())
 
# Define Carr-Madan pricer as benchmark
boundary = 32
α = 1
dynamics = Hedgehog2.HestonDynamics()
method = Hedgehog2.CarrMadan(α, boundary, dynamics)
carr_madan_pricer = Pricer(payoff, market_inputs, method)
carr_madan_price = carr_madan_pricer()

# Construct the Heston Noise Processs
trajectories = 10000
#higher tolerance in the CDF inversion
montecarlo_method = Hedgehog2.Montecarlo(trajectories, dynamics; atol=1E-4, cf_tol=1E-4) 
mc_pricer = Pricer(payoff, market_inputs, montecarlo_method)

println(carr_madan_price)
@time println(mc_pricer())