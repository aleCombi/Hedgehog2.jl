using Revise, Hedgehog2, Dates

reference_date = Date(2020, 1, 1)

# Query interpolated vol
T = 0.75
K = 95.0

r = 0.02
spot = 100.0
market_inputs = BlackScholesInputs(reference_date, r, spot, 0.65)
expiry_date = reference_date + Day(365)
payoff_call = VanillaOption(80.0, expiry_date, European(), Call(), Spot())
pricing_problem = PricingProblem(payoff_call, market_inputs)
price = solve(pricing_problem, BlackScholesAnalytic()).price

calibration_problem =
    Hedgehog2.BlackScholesCalibrationProblem(pricing_problem, BlackScholesAnalytic(), price)
calibrated_vol = solve(calibration_problem)
print(calibrated_vol)
#TODO: test the inversion
