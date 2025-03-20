using Revise, Hedgehog2, BenchmarkTools, Dates

# define payoff
strike = 1.2
expiry = Date(2021, 1, 1)
call_put = Hedgehog2.Put()
underlying = Hedgehog2.Forward()
american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), call_put, underlying)
euro_payoff = VanillaOption(strike, expiry, Hedgehog2.European(), call_put, underlying)

# define market inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# create analytical black scholes pricer
bs_method = BlackScholesMethod()
analytical_pricer = Pricer(euro_payoff, market_inputs, bs_method)

# create Cox Ross Rubinstein pricer
steps = 800
crr = CoxRossRubinsteinMethod(steps)
crr_euro_pricer = Pricer(euro_payoff, market_inputs, crr)
crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

# print results
println("Cox Ross Rubinstein European Price:")
println(crr_euro_pricer())

println("Cox Ross Rubinstein American Price:")
println(crr_american_pricer())

println("Analytical European price:")
println(analytical_pricer())