using Revise, Hedgehog, BenchmarkTools, Dates

# define payoff
strike = 1.2
expiry = Date(2021, 1, 1)
underlying = Hedgehog.Forward()
american_payoff = VanillaOption(strike, expiry, Hedgehog.American(), Put(), underlying)
euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# create analytical black scholes pricer
analytical_price = solve(euro_pricing_prob, BlackScholesAnalytic())

# create Cox Ross Rubinstein pricer
steps = 800
crr = CoxRossRubinsteinMethod(steps)
crr_euro_prob = PricingProblem(euro_payoff, market_inputs)
crr_euro_price = solve(crr_euro_prob, crr)
crr_american_prob = PricingProblem(american_payoff, market_inputs)
crr_american_price = solve(crr_american_prob, crr)

# print results
println("Cox Ross Rubinstein European Price:")
println(crr_euro_price)

println("Cox Ross Rubinstein American Price:")
println(crr_american_price)

println("Analytical European price:")
println(analytical_price)
