using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using DifferentialEquations

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2021, 1, 1)

euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 0.04

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

dynamics = LognormalDynamics()
trajectories = 10000
strategy = EulerMaruyama(trajectories; steps=100, variance_reduction=Hedgehog2.Antithetic())
montecarlo_method = MonteCarlo(dynamics, strategy)

solution_analytic = Hedgehog2.solve(euro_pricing_prob, BlackScholesAnalytic()).price
solution = Hedgehog2.solve(euro_pricing_prob, montecarlo_method)

solution.ensemble.solution
solution.ensemble.antithetic_sol
@show solution
@show solution_analytic

using Plots

# Assuming `solution` is your MonteCarloSolution with `.ensemble` of type MonteCarloSol
primary_path     = solution.ensemble.solution.u[1].u
antithetic_path  = solution.ensemble.antithetic_sol.u[1].u

# Extract time axis from one of them (they should match)
t = solution.ensemble.solution.u[1].t
# Plot
plot(t, (primary_path), label = "Primary Path", lw = 2)
plot!(t, (antithetic_path), label = "Antithetic Path", lw = 2, ls = :dash)


@btime Hedgehog2.solve($euro_pricing_prob, $montecarlo_method).price
@btime Hedgehog2.solve($euro_pricing_prob, BlackScholesAnalytic()).price

# antithetic = get(strategy.kwargs, :antithetic, false)::Bool

# Step 1: simulate original paths
normal_prob = @code_warntype Hedgehog2.sde_problem(dynamics, strategy, market_inputs, tspan)

@code_warntype Hedgehog2.get_ensemble_problem(normal_prob, strategy)