using Revise, Dates, Hedgehog2

reference_date = Date(2020, 1, 1)
r, S0, sigma = 0.05, 100.0, 0.25
market_inputs = BlackScholesInputs(reference_date, r, S0, sigma)

strikes = collect(60.0:5.0:140.0)
expiry = reference_date + Day(365)
payoffs = [VanillaOption(K, expiry, European(), Call(), Spot()) for K in strikes]

quotes = [
    solve(PricingProblem(p, market_inputs), BlackScholesAnalytic()).price for
    p in payoffs
]

accessors = [VolLens(1,1)]
initial_guess = [0.15]

basket = BasketPricingProblem(payoffs, market_inputs)
calib = CalibrationProblem(basket, BlackScholesAnalytic(), accessors, quotes, 0.7*ones(length(payoffs)))
result = solve(calib, OptimizerAlgo())
@show result.objective