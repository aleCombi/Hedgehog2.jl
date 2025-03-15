
# Simulate GBM process
"""
    simulate_gbm()

Simulates a Geometric Brownian Motion (GBM) process using the Euler-Maruyama method.
Returns the numerical solution.
"""
function simulate_gbm()
    u0 = 100.0  # Initial stock price
    tspan = (0.0, 1.0)  # Simulation for 1 year
    process = GBMProcess(0.05, 0.2)  # μ = 5%, σ = 20%
    sde_function = get_sde_function(process)
    prob = SDEProblem(sde_function, u0, tspan)
    sol = solve(prob, EM(), dt=0.01)  # Euler-Maruyama solver

    return sol
end

# Simulate Heston process
"""
    simulate_heston()

Simulates the Heston stochastic volatility model using the Euler-Maruyama method.
Returns the numerical solution.
"""
function simulate_heston()
    u0 = [100.0, 0.04]  # Initial (Stock Price, Variance)
    tspan = (0.0, 1.0)  # Simulation for 1 year
    process = HestonProcess(0.05, 2.0, 0.04, 0.3, -0.5)  # Heston parameters
    prob = SDEProblem(get_sde_function(process), u0, tspan, process)
    sol = solve(prob, EM(), dt=0.01)  # Euler-Maruyama solver

    return sol
end

# Running the simulations
using Revise, Plots, Hedgehog2, DifferentialEquations

sol_gbm = simulate_gbm()
sol_heston = simulate_heston()

# Plot GBM
gbm_plot = plot(sol_gbm, plot_analytic = true, label="GBM Path", xlabel="Time", ylabel="Stock Price", lw=2)

# Plot Heston
heston_plot = plot!(sol_heston.t, [u[1] for u in sol_heston.u], label="Heston Stock Price", xlabel="Time", ylabel="Stock Price", lw=2)
heston_plot_var = plot!(sol_heston.t, [u[2] for u in sol_heston.u], label="Variance", lw=2)

display(gbm_plot)
display(heston_plot)
display(heston_plot_var)