using Dates
using Accessors  # for @lens
using Optimization
using Hedgehog2  # replace with your module name

# Inputs
reference_date = Date(2025, 1, 1)
expiry = Date(2025, 7, 1)
strike = 100.0
spot = 100.0
rate = FlatRateCurve(0.01)  # assume this gives df(t)

# Build payoff
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Build market inputs with dummy sigma
market_inputs = BlackScholesInputs(
    reference_date,
    rate,
    spot,
    0.2
)

vol = 0.3

# Create the pricing problem
pricing_problem = PricingProblem(payoff, market_inputs)
market_price = Hedgehog2.solve(pricing_problem, BlackScholesAnalytic()).price

# Wrap in a calibration problem
calib = CalibrationProblem(
    BasketPricingProblem([payoff], market_inputs),   # pricing problem container
    BlackScholesAnalytic(),                    # method
    [@optic _.market_inputs.sigma],             # parameter to calibrate
    [market_price],                            # target price
    [vol],                                     # initial guess
)

# Solve for implied vol using root finding
sol = Hedgehog2.solve(calib, RootFinderAlgo())

# Extract implied vol
implied_vol = sol.u
println("Implied vol = ", implied_vol)
println("Real vol = ", vol)