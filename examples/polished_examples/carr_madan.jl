using Revise, Hedgehog2, BenchmarkTools, Dates, Accessors

# -- Market Inputs
reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 100.0
sigma = 0.4
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# -- Payoff
expiry = reference_date + Day(365)
strike = 100.0
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# -- Pricing problem
prob = PricingProblem(payoff, market_inputs)

# -- Methods
method_analytic = BlackScholesAnalytic()
method_carr_madan = CarrMadan(1.0, 16.0, LognormalDynamics())

# -- Solve for prices
sol_analytic = Hedgehog2.solve(prob, method_analytic)
sol_carr = Hedgehog2.solve(prob, method_carr_madan)

@btime Hedgehog2.solve($prob, $method_analytic)
@btime Hedgehog2.solve($prob, $method_carr_madan)

println("Analytic price:   ", sol_analytic.price)
println("Carr-Madan price: ", sol_carr.price)

# --- Greeks via GreekProblem

# Accessors
spot_lens = @optic _.market_inputs.spot
sigma_lens = Hedgehog2.VolLens(1,1)

# Methods
fd = FiniteDifference(1e-3)
ad = ForwardAD()

println("\n--- Greeks (Analytic Method) ---")
delta_fd = Hedgehog2.solve(GreekProblem(prob, spot_lens), fd, method_analytic).greek
vega_fd = Hedgehog2.solve(GreekProblem(prob, sigma_lens), fd, method_analytic).greek
delta_ad = Hedgehog2.solve(GreekProblem(prob, spot_lens), ad, method_analytic).greek
vega_ad = Hedgehog2.solve(GreekProblem(prob, sigma_lens), ad, method_analytic).greek

println("FD Delta (analytic): ", delta_fd)
println("AD Delta (analytic): ", delta_ad)
println("FD Vega  (analytic): ", vega_fd)
println("AD Vega  (analytic): ", vega_ad)

println("\n--- Greeks (Carr-Madan Method) ---")
delta_fd_cm = Hedgehog2.solve(GreekProblem(prob, spot_lens), fd, method_carr_madan).greek
vega_fd_cm = Hedgehog2.solve(GreekProblem(prob, sigma_lens), fd, method_carr_madan).greek
delta_ad_cm = Hedgehog2.solve(GreekProblem(prob, spot_lens), ad, method_carr_madan).greek
vega_ad_cm = Hedgehog2.solve(GreekProblem(prob, sigma_lens), ad, method_carr_madan).greek

println("FD Delta (Carr-Madan): ", delta_fd_cm)
println("AD Delta (Carr-Madan): ", delta_ad_cm)
println("FD Vega  (Carr-Madan): ", vega_fd_cm)
println("AD Vega  (Carr-Madan): ", vega_ad_cm)
