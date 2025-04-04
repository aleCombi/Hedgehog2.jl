using Revise, Hedgehog2, Dates, Accessors

# -- Market Inputs (true values)
reference_date = Date(2020, 1, 1)
true_rate = 0.05
true_spot = 100.0
true_sigma = 0.4
market = BlackScholesInputs(reference_date, true_rate, true_spot, true_sigma)

# -- Define several European call options
payoffs = [
    VanillaOption(90.0, reference_date + Day(180), European(), Call(), Spot()),
    VanillaOption(100.0, reference_date + Day(365), European(), Call(), Spot()),
    VanillaOption(110.0, reference_date + Day(540), European(), Call(), Spot())
]

# -- Pricing method
pricing_method = BlackScholesAnalytic()

# -- Generate true prices
true_prices = [
    solve(PricingProblem(payoff, market), pricing_method).price
    for payoff in payoffs
]

# -- Define CalibrationProblem
basket_problem = Hedgehog2.BasketPricingProblem(payoffs, market)

# Calibrate both sigma and rate
accessors = [
    @optic _.market.sigma
]

initial_guess = [0.2]  # Start from wrong sigma and rate

calib_problem = Hedgehog2.CalibrationProblem(
    basket_problem,
    pricing_method,
    accessors,
    true_prices
)

# -- Run calibration
result = solve(calib_problem, initial_guess)

# -- Output
@show true_sigma
@show true_rate
@show result.u
@show true_prices
