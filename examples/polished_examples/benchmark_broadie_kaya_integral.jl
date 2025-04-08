using Revise
using Hedgehog2
import Hedgehog2:HestonCFIterator, evaluate_chf
using Integrals, Random
using Printf
using BenchmarkTools

function broadie_kaya_integrand(a, ϕ_iter::HestonCFIterator, x, θ_ref::Ref)
    ϕ_val, new_θ = evaluate_chf(ϕ_iter, a, θ_ref[])
    θ_ref[] = new_θ
    return (2 / π) * sin(a * x) / a * real(ϕ_val)
end

function cdf_from_cf_loop(ϕ_iter::HestonCFIterator, x, h; cf_tol = 1e-5)
    if x < 0
        return 0.0
    end

    result = h * x / 2
    θ_prev = NaN
    for j = 1:10^8  # big upper bound
        a = h * j
        ϕ_val, θ_prev = evaluate_chf(ϕ_iter, a, θ_prev)
        term = sin(a * x) / j * real(ϕ_val)
        result += term
        if abs(ϕ_val) / j < π * cf_tol / 2
            break
        end
    end
    return 2 / π * result
end

function broadie_kaya_integrand(a, ϕ_iter::HestonCFIterator, x, θ_ref::Ref)
    ϕ_val, new_θ = evaluate_chf(ϕ_iter, a, θ_ref[])
    θ_ref[] = new_θ
    return (2 / π) * x * sinc((a * x) / π) * real(ϕ_val)
end

function cdf_from_cf_trapezoidal(ϕ_iter::HestonCFIterator, x;
    method = QuadGKJL(),
    abstol = 1e-6,
    reltol = 1e-6,
    max_a = 100.0)

    if x < 0
        return 0.0
    end

    θ_ref = NaN
    function integrand(a,p) 
        ϕ_val, θ_ref = evaluate_chf(ϕ_iter, a, θ_ref)
        return (2 / π) * x * sinc((a * x) / π) * real(ϕ_val)
    end

    problem = IntegralProblem(integrand, 0, Inf)
    sol = Integrals.solve(problem, method; abstol=abstol, reltol=reltol)

return sol.u
end

S0, V0 = 100.0, 0.04
κ, θ, σ, ρ = 2.0, 0.04, 0.3, -0.7
r, T = 0.03, 1.0
d = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)
VT = Hedgehog2.sample_V_T(MersenneTwister(123), d)
ϕ = Hedgehog2.HestonCFIterator(VT, d)

# x and h
mean_h, variance = Hedgehog2.moments_from_cf(ϕ)
x = mean_h
h = π / (mean_h + 5 * sqrt(variance))

# Warmup
cdf_loop = cdf_from_cf_loop(ϕ, x, h)
int_meth = QuadGKJL()
cdf_integ = cdf_from_cf_trapezoidal(ϕ, x; method=int_meth)


# ✅ Check consistency
@printf("\nLoop = %.10f\nIntegration = %.10f\n", cdf_loop, cdf_integ)
@printf("Absolute error = %.3e\n", abs(cdf_loop - cdf_integ))
println("✔️ Match: ", isapprox(cdf_loop, cdf_integ; atol=1e-6))

# ⏱️ Benchmark
println("\nBenchmarking original loop...")
@btime cdf_from_cf_loop($ϕ, $x, $h)

println("\nBenchmarking Integration.jl version...")
@btime cdf_from_cf_trapezoidal($ϕ, $x; method=$int_meth)

cdf_func(x) = cdf(Normal(), x)  # example

u = 0.75
using Distributions
initial_guess = quantile(Normal(), u)
max_guess = 10.0

Hedgehog2.inverse_cdf(cdf_func, u, initial_guess, max_guess)
