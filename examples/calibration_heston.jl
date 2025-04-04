using Revise, Hedgehog2, Dates, Accessors

# --- True model inputs (used to generate quotes)
reference_date = Date(2020, 1, 1)
S0 = 100.0
true_params = (
    v0 = 0.010201,
    κ = 6.21,
    θ = 0.019,
    σ = 0.61,
    ρ = -0.7,
)
r = 0.0319
market_inputs = HestonInputs(
    reference_date, r, S0,
    true_params.v0, true_params.κ, true_params.θ,
    true_params.σ, true_params.ρ
)

# --- Payoffs
# --- Payoffs
# --- Strikes and multiple expiries
strikes = collect(60.0:5.0:140.0)  # 60, 65, ..., 140

expiries = [
    reference_date + Day(90),    # 3 months
    reference_date + Day(180),   # 6 months
    reference_date + Day(365)    # 1 year
]

payoffs = [
    VanillaOption(K, expiry, European(), Call(), Spot())
    for K in strikes, expiry in expiries
] |> vec


# --- Pricing method: Carr-Madan under Heston
α, boundary = 1.0, 32.0
method_heston = CarrMadan(α, boundary, HestonDynamics())

# --- True prices from Heston model
quotes = [
    solve(PricingProblem(p, market_inputs), method_heston).price
    for p in payoffs
]

# --- Calibration Setup

# Start from wrong initial parameters
initial_guess = [0.02, 3.0, 0.03, 0.4, -0.3]  # [v0, κ, θ, σ, ρ]

# Accessors for each parameter in HestonInputs
accessors = [
    @optic(_.market.V0),
    @optic(_.market.κ),
    @optic(_.market.θ),
    @optic(_.market.σ),
    @optic(_.market.ρ)
]


# CalibrationProblem
basket_problem = Hedgehog2.BasketPricingProblem(payoffs, market_inputs)

calib_problem = Hedgehog2.CalibrationProblem(
    basket_problem,
    method_heston,
    accessors,
    quotes
)

# --- Solve calibration
result = solve(calib_problem, initial_guess)

# --- Output
println("True Heston parameters:")
@show true_params
println("Calibrated parameters:")
@show result.u
println("Target prices:")
@show quotes
