
# Simulate GBM process
"""
    simulate_gbm()

Simulates a Geometric Brownian Motion (GBM) process using the Euler-Maruyama method.
Returns the numerical solution.
"""
function gbm_problem()
    u0 = 100.0  # Initial stock price
    tspan = (0.0, 1.0)  # Simulation for 1 year
    process = GBMProcess(0.05, 0.2)  # μ = 5%, σ = 20%
    sde_function = get_sde_function(process)
    prob = SDEProblem(sde_function, u0, tspan)
    return prob
end

# Simulate Heston process
"""
    simulate_heston()

Simulates the Heston stochastic volatility model using the Euler-Maruyama method.
Returns the numerical solution.
"""
function heston_problem()
    u0 = [100.0, 0.04]  # Initial (Stock Price, Variance)
    tspan = (0.0, 1.0)  # Simulation for 1 year
    process = HestonProcess(0.05, 2.0, 0.04, 0.3, -0.5)  # Heston parameters
    prob = SDEProblem(get_sde_function(process), u0, tspan, process)

    return prob
end

# Running the simulations
using Revise, Plots, Hedgehog, DifferentialEquations

gbm = gbm_problem()
heston = heston_problem()

# Plot GBM
sol_gbm = solve(gbm, EM(), dt = 0.01)  # Euler-Maruyama solver
gbm_plot = plot(
    sol_gbm,
    plot_analytic = true,
    label = "GBM Path",
    xlabel = "Time",
    ylabel = "Stock Price",
    lw = 2,
)

# Plot Heston
sol_heston = solve(heston, EM(), dt = 0.01)  # Euler-Maruyama solver
plot(
    sol_heston.t,
    [u[1] for u in sol_heston.u],
    label = "Heston Stock Price",
    xlabel = "Time",
    ylabel = "Stock Price",
    lw = 2,
)
plot(sol_heston.t, [u[2] for u in sol_heston.u], label = "Variance", lw = 2)

# ensemble GBM for 1000 trajectories
ensembleprob = EnsembleProblem(gbm)
sol = solve(ensembleprob, EnsembleThreads(), trajectories = 1000)

using DifferentialEquations.EnsembleAnalysis
summ = EnsembleSummary(sol, 0:0.01:1)
plot(summ, labels = "Middle 95%")
summ = EnsembleSummary(sol, 0:0.01:1; quantiles = [0.25, 0.75])
plot!(summ, labels = "Middle 50%", legend = true)
