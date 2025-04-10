using ForwardDiff, BenchmarkTools, StaticArrays

# --- Target function: f(x, y) = x² + sin(y)
f(x, y) = x^2 + sin(y)

# Wrap it for gradient interface: input is a vector
f_vec(v) = f(v[1], v[2])

# --- Evaluation point
x₀ = 1.0
y₀ = 0.5
v₀ = [x₀, y₀]
sv₀ = @SVector [x₀, y₀]  # StaticVector version

# --- 1. Scalar derivatives
∂f_∂x() = ForwardDiff.derivative(x -> f(x, y₀), x₀)
∂f_∂y() = ForwardDiff.derivative(y -> f(x₀, y), y₀)

# --- 2. Regular vector gradient
function grad_vec()
    ForwardDiff.gradient(f_vec, v₀)
end

# --- 3. StaticVector gradient
function grad_svec()
    ForwardDiff.gradient(f_vec, sv₀)
end

# --- Run all and print
println("Expected: [2x, cos(y)] = [$(2x₀), $(cos(y₀))]")

println("\n--- Results ---")
println("Scalar x: ", ∂f_∂x())
println("Scalar y: ", ∂f_∂y())
println("Gradient (Vector): ", grad_vec())
println("Gradient (StaticVector): ", grad_svec())

println("\n--- Benchmarks ---")
@btime ∂f_∂x()
@btime ∂f_∂y()
@btime grad_vec()
@btime grad_svec()
