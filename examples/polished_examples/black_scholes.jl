using Revise, Hedgehog2, BenchmarkTools, Dates

# define payoff
strike = 1.2
expiry = Date(2021, 1, 1)
underlying = Hedgehog2.Forward()
euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# create analytical black scholes pricer
@btime analytical_price = solve(euro_pricing_prob, BlackScholesAnalytic())

# # print results
# println("Analytical European price:")
# println(analytical_price)
