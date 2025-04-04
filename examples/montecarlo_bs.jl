println("\n--- Repeated runs for variance, MSE, and timing ---")

n_trials = 100
prices_std = Float64[]
prices_anti = Float64[]
times_std = Float64[]
times_anti = Float64[]
trajectories = 10000

for trial = 1:n_trials
    seed = 1_000_000 + trial

    strat_std = BlackScholesExact(2 * trajectories, seed = seed)
    strat_anti = BlackScholesExact(trajectories, seed = seed, antithetic = true)

    method_std = MonteCarlo(dynamics, strat_std)
    method_anti = MonteCarlo(dynamics, strat_anti)

    time_std = @elapsed begin
        price_std = solve(prob, method_std).price
        push!(prices_std, price_std)
    end

    time_anti = @elapsed begin
        price_anti = solve(prob, method_anti).price
        push!(prices_anti, price_anti)
    end

    push!(times_std, time_std)
    push!(times_anti, time_anti)
end

# --- Compute Metrics
true_price = solution_analytic.price

var_std = var(prices_std)
var_anti = var(prices_anti)

mse_std = mean((prices_std .- true_price) .^ 2)
mse_anti = mean((prices_anti .- true_price) .^ 2)

mean_time_std = mean(times_std)
mean_time_anti = mean(times_anti)

vr_ratio = var_std / var_anti
mse_ratio = mse_std / mse_anti
time_ratio = mean_time_std / mean_time_anti

# --- Print Report
println("Variance (standard MC):    ", round(var_std, digits = 8))
println("Variance (antithetic MC):  ", round(var_anti, digits = 8))
println("Variance reduction:        ", round(vr_ratio, digits = 3), "×\n")

println("MSE (standard MC):         ", round(mse_std, digits = 8))
println("MSE (antithetic MC):       ", round(mse_anti, digits = 8))
println("MSE reduction:             ", round(mse_ratio, digits = 3), "×\n")

println(
    "Mean execution time (standard MC):   ",
    round(mean_time_std * 1000, digits = 3),
    " ms",
)
println(
    "Mean execution time (antithetic MC): ",
    round(mean_time_anti * 1000, digits = 3),
    " ms",
)
println("Time ratio (std / antithetic):       ", round(time_ratio, digits = 3), "×")
