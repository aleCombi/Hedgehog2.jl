using Revise, Hedgehog, StochasticDiffEqs, Plots

process = GBMProcess(0.05, 0.2)  # μ = 5%, σ = 20%
sde_function_1 = get_sde_function(process)
sde_function_2 = get_sde_function(process)
combined_sde_function = combine_sde_functions((sde_function_1, sde_function_2))

ρ = -0.8
# Correlation matrix
Γ = [
    1.0 ρ
    ρ 1.0
]

u0 = [1.0, 1.0]  # Initial conditions
tspan = (0.0, 2.0)  # Time span

noise = CorrelatedWienerProcess(Γ, tspan[1], [0.0, 0.0])
# Create multi-dimensional SDE problem
prob = SDEProblem(combined_sde_function, u0, tspan, "p", noise = noise)
prob2 = get_sde_problem((process, process), Γ, u0, tspan, "p")

sol = solve(prob)
plot(sol, plot_analytic = true)

sol2 = solve(prob2)
plot!(sol, plot_analytic = true)
