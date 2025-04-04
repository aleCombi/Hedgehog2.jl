using Revise, Hedgehog2, DifferentialEquations, Plots

# Define a constant volatility function and its integral
σ_const = t -> 0.2  # Constant volatility (σ_0 = 0.2)
function mean_σ²_const(t)
    σ1² = 0.2^2  # σ_1^2
    σ2² = 0.4^2  # σ_2^2

    if t < 0.5
        return σ1²
    else
        return (σ1² * 0.5 + σ2² * (t - 0.5)) / t
    end
end


# Create the GBMTimeDependent process with constant volatility
process_const = GBMTimeDependent(
    0.05,          # Drift
    σ_const,       # Constant volatility function
    mean_σ²_const,  # Integral of σ²(t)
)

# Get SDE function
sde_func_const = get_sde_function(process_const)

# Initial value and time span
u0 = [1.0]
tspan = (0.0, 2.0)


sde_problem_td = SDEProblem(sde_func_const, u0, tspan, seed = 12)
sol_td = solve(sde_problem_td)
plot(sol_td)

process_gbm = GBMProcess(0.05, 0.2)
sde_func = get_sde_function(process_gbm)
sde_problem = SDEProblem(sde_func, u0, tspan, seed = 12)
sol = solve(sde_problem)
plot!(sol)
