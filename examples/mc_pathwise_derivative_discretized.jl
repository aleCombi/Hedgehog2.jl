using DifferentialEquations
using ForwardDiff
using Plots

# Parameters
μ = 0.05
σ = 0.2
t0 = 0.0
T = 1.0
W0 = 100.0
tspan = (t0, T)
N = 250
dt = (T - t0) / N
seed = 1

# --- Drift and diffusion functions
function drift!(du, u, p, t)
    du[1] = p[1] * u[1]  # p[1] = μ
end

function diffusion!(du, u, p, t)
    du[1] = p[2] * u[1]  # p[2] = σ
end

# --- Terminal value of GBM using SDEProblem
function terminal_value_sde(σ_val::Real)
    T_ = typeof(σ_val)
    μ_ = convert(T_, μ)
    W0_ = convert(T_, W0)

    u0 = [W0_]
    p = [μ_, σ_val]

    prob = SDEProblem(drift!, diffusion!, u0, tspan, p; seed = seed)
    sol = DifferentialEquations.solve(prob, EM(); dt = dt)
    return sol.u[end][1]
end

# --- Finite difference
function finite_difference(f, x; ε = 1e-5)
    (f(x + ε) - f(x - ε)) / (2ε)
end

# --- Compute values
val = terminal_value_sde(σ)
grad_ad = ForwardDiff.derivative(terminal_value_sde, σ)
grad_fd = finite_difference(terminal_value_sde, σ)

println("Terminal Value at σ = $σ: ", val)
println("∂Value/∂σ (AD): ", grad_ad)
println("∂Value/∂σ (FD): ", grad_fd)

# --- Plot
σ_bumped = σ + 0.05
prob = SDEProblem(drift!, diffusion!, [W0], tspan, [μ, σ]; seed = seed)
prob_bumped = SDEProblem(drift!, diffusion!, [W0], tspan, [μ, σ_bumped]; seed = seed)

sol = DifferentialEquations.solve(prob, EM(); dt = dt)
sol_bumped = DifferentialEquations.solve(prob_bumped, EM(); dt = dt)

plot(sol.t, first.(sol.u), label = "σ = $σ", linewidth = 2)
plot!(
    sol_bumped.t,
    first.(sol_bumped.u),
    label = "σ = $σ_bumped",
    linestyle = :dash,
    linewidth = 2,
)
xlabel!("Time")
ylabel!("GBM Value")
title!("GBM (SDE): Vol Bump and Pathwise Sensitivity")
