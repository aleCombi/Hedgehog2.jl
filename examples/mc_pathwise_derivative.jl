using DifferentialEquations
using DiffEqNoiseProcess
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

# --- Terminal value of GBM with type stability
function terminal_value(σ_val::Real)
    T_ = typeof(σ_val)
    μ_ = convert(T_, μ)
    W0_ = convert(T_, W0)
    t0_ = convert(T_, t0)

    proc = GeometricBrownianMotionProcess(μ_, σ_val, t0_, W0_)
    prob = NoiseProblem(proc, tspan; seed=seed)
    sol = DifferentialEquations.solve(prob, EM(); dt=dt)
    return sol.u[end]
end

# --- Finite difference
function finite_difference(f, x; ε=1e-5)
    (f(x + ε) - f(x - ε)) / (2ε)
end

# --- Compute values
val = terminal_value(σ)
grad_ad = ForwardDiff.derivative(terminal_value, σ)
grad_fd = finite_difference(terminal_value, σ)

println("Terminal Value at σ = $σ: ", val)
println("∂Value/∂σ (AD): ", grad_ad)
println("∂Value/∂σ (FD): ", grad_fd)

# --- Plot
σ_bumped = σ + 0.05
sol = DifferentialEquations.solve(NoiseProblem(GeometricBrownianMotionProcess(μ, σ, t0, W0), tspan; seed=seed), dt=dt)
sol_bumped = DifferentialEquations.solve(NoiseProblem(GeometricBrownianMotionProcess(μ, σ_bumped, t0, W0), tspan; seed=seed), dt=dt)

plot(sol.t, sol.u, label="σ = $σ", linewidth=2)
plot!(sol_bumped.t, sol_bumped.u, label="σ = $σ_bumped", linestyle=:dash, linewidth=2)
xlabel!("Time")
ylabel!("GBM Value")
title!("GBM: Vol Bump and Pathwise Sensitivity")
