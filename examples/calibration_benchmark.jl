using Revise, Hedgehog2, Dates, BenchmarkTools

# --- True model inputs
reference_date = Date(2020, 1, 1)
S0 = 100.0
true_params = (v0 = 0.010201, κ = 6.21, θ = 0.019, σ = 0.61, ρ = -0.7)
r = 0.0319
market_inputs = HestonInputs(
    reference_date,
    r,
    S0,
    true_params.v0,
    true_params.κ,
    true_params.θ,
    true_params.σ,
    true_params.ρ,
)

# --- Payoffs
strikes = collect(60.0:5.0:140.0)
expiries = [reference_date + Day(d) for d in (90, 180, 365)]
payoffs =
    [
        VanillaOption(K, expiry, European(), Call(), Spot()) for K in strikes,
        expiry in expiries
    ] |> vec

# --- Quotes from Heston model
α, boundary = 1.0, 32.0
method_heston = CarrMadan(α, boundary, HestonDynamics())
quotes = [solve(PricingProblem(p, market_inputs), method_heston).price for p in payoffs]

# --- Calibration setup
initial_guess = [0.02, 3.0, 0.03, 0.4, -0.3]
accessors = [
    @optic(_.market_inputs.V0),
    @optic(_.market_inputs.κ),
    @optic(_.market_inputs.θ),
    @optic(_.market_inputs.σ),
    @optic(_.market_inputs.ρ)
]

basket_problem = Hedgehog2.BasketPricingProblem(payoffs, market_inputs)

calib_problem = Hedgehog2.CalibrationProblem(
    basket_problem,
    method_heston,
    accessors,
    quotes,
    initial_guess,
)

# --- Run benchmark
println("\nBenchmarking AutoForwardDiff()...")
@btime solve(
    $calib_problem,
    Hedgehog2.OptimizerAlgo(AutoForwardDiff(), Optimization.LBFGS()),
)

println("\nBenchmarking AutoFiniteDiff()...")
@btime solve(
    $calib_problem,
    Hedgehog2.OptimizerAlgo(AutoFiniteDiff(), Optimization.LBFGS()),
)
