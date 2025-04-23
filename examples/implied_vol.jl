using Dates
using Accessors
using Optimization
using Hedgehog

# Inputs
reference_date = Date(2025, 1, 1)
expiry = Date(2025, 7, 1)
strike = 1.0
spot = 1.0
rate = 0.01
vol = 0.36

# Build payoff
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Build market inputs with dummy sigma
market_inputs = BlackScholesInputs(
    reference_date,
    rate,
    spot,
    vol
)


# Create the pricing problem
pricing_problem = PricingProblem(payoff, market_inputs)
market_price = Hedgehog.solve(pricing_problem, BlackScholesAnalytic()).price

# Wrap in a calibration problem
calib = CalibrationProblem(
    BasketPricingProblem([payoff], market_inputs),   # pricing problem container
    BlackScholesAnalytic(),                    # method
    [VolLens(1,1)],             # parameter to calibrate
    [market_price],                            # target price
    [0.05],                                     # initial guess
)

# Solve for implied vol using root finding
sol = Hedgehog.solve(calib, RootFinderAlgo())

# Extract implied vol
implied_vol = sol.u
println("Implied vol = ", implied_vol)
println("Real vol = ", vol)