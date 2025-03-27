using Revise, Hedgehog2, Interpolations, Dates

# Grid definitions
tenors  = [0.25, 0.5, 1.0, 2.0]               # maturities in years
strikes = [80.0, 90.0, 100.0, 110.0]          # strikes

# 2D volatility grid: vols[i, j] corresponds to (tenor[i], strike[j])
vols = [
    0.22  0.21  0.20  0.19;    # T = 0.25
    0.23  0.22  0.21  0.20;    # T = 0.50
    0.25  0.24  0.23  0.22;    # T = 1.00
    0.28  0.27  0.26  0.25     # T = 2.00
]

reference_date = Date(2020,1,1)
vol_surface = RectVolSurface(reference_date, tenors, strikes, vols)

# Query interpolated vol
T = 0.75
K = 95.0
σ = get_vol(vol_surface, T, K)

println("Implied vol at T = $T, K = $K is σ = $σ")

r = 0.02
spot = 100.0
market_inputs = BlackScholesInputs(reference_date, r, spot, 0.65)
expiry_date = reference_date + Day(365)
payoff_call = VanillaOption(80.0, expiry_date, European(), Call(), Spot())
pricing_problem = PricingProblem(payoff_call, market_inputs)
price = solve(pricing_problem, BlackScholesAnalytic()).price

calibration_problem = Hedgehog2.BlackScholesCalibrationProblem(pricing_problem, BlackScholesAnalytic(), price)
calibrated_vol = solve(calibration_problem, BlackScholesAnalytic())
print(calibrated_vol)
#TODO: test the inversion