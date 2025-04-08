using Revise, Distributions, Random, BenchmarkTools, Roots, SpecialFunctions
using Hedgehog2

println("\n🧾 Heston Parameters")
S0, V0 = 100.0, 0.04
κ, θ, σ, ρ = 2.0, 0.04, 0.3, -0.7
r, T = 0.03, 1.0
d = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)
rng = MersenneTwister(1234)

println("\n📈 Step 1: sample_V_T")
@btime Hedgehog2.sample_V_T($rng, $d)
VT = Hedgehog2.sample_V_T(rng, d)
println("VT = "); show(VT); println()

println("\n🧱 Step 2: Construct HestonCFIterator")
@btime Hedgehog2.HestonCFIterator($VT, $d)
iter = Hedgehog2.HestonCFIterator(VT, d)
println("Iterator = "); show(iter); println()

println("\n📊 Step 3: Estimate moments from CF")
@btime Hedgehog2.moments_from_cf($iter)
mean_h, variance = Hedgehog2.moments_from_cf(iter)
println("mean = $mean_h, variance = $variance")

println("\n🔮 Step 4: Evaluate characteristic function")
@btime Hedgehog2.evaluate_chf($iter, 0.5, nothing)
ϕ_val, _ = Hedgehog2.evaluate_chf(iter, 0.5, nothing)
println("ϕ(0.5) = "); show(ϕ_val); println()

println("\n📐 Step 5: Evaluate CDF from CF")
x = mean_h
h = π / (mean_h + 5 * sqrt(variance))
@btime Hedgehog2.cdf_from_cf($iter, $x, $h)
cdf_val = Hedgehog2.cdf_from_cf(iter, x, h)
println("cdf(x = $x) = $cdf_val")

println("\n🔁 Step 6: Invert CDF")
cdf_func = x -> Hedgehog2.cdf_from_cf(iter, x, h)
u = rand(rng)
initial_guess = mean_h + sqrt(variance) * quantile(Normal(), u)
max_guess = mean_h + 11 * sqrt(variance)
@btime Hedgehog2.inverse_cdf($cdf_func, $u, $initial_guess, $max_guess)
sample = Hedgehog2.inverse_cdf(cdf_func, u, initial_guess, max_guess)
println("inverse_cdf(u = $u) = $sample")

println("\n🎯 Step 7: Full sample_from_cf")
ϕ = Hedgehog2.HestonCFIterator(VT, d)
@btime Hedgehog2.sample_from_cf($rng, $ϕ)
sample2 = Hedgehog2.sample_from_cf(rng, ϕ)
println("sample_from_cf = $sample2")

println("\n🎲 Step 8: Final rand(rng, d)")
@btime Hedgehog2.rand($rng, $d)
sample3 = Hedgehog2.rand(rng, d)
println("rand = "); show(sample3); println()

# using Profile, ProfileView
# @profile rand(rng, d)
# ProfileView.view()
