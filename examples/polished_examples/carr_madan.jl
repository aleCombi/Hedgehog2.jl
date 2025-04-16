using Revise, Hedgehog, BenchmarkTools, Dates, Accessors

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
@btime Hedgehog.solve($prob, $method_analytic)
@btime Hedgehog.solve($prob, $method_carr_madan)

sol_analytic = Hedgehog.solve(prob, method_analytic)
sol_carr = Hedgehog.solve(prob, method_carr_madan)

println("Analytic price:   ", sol_analytic.price)
println("Carr-Madan price: ", sol_carr.price)

# --- Greeks via Accessors
spot_lens = @optic _.market_inputs.spot
sigma_lens = Hedgehog.VolLens(1, 1)
lenses = (sigma_lens, spot_lens)

# -- Greek methods
fd = FiniteDifference(1e-3)
ad = ForwardAD()

println("\n--- Greeks (Analytic Method) ---")
batch_prob = BatchGreekProblem(prob, lenses)
greeks_fd = solve(batch_prob, fd, method_analytic)
greeks_ad = solve(batch_prob, ad, method_analytic)

println("FD Delta (analytic): ", greeks_fd[spot_lens])
println("AD Delta (analytic): ", greeks_ad[spot_lens])
println("FD Vega  (analytic): ", greeks_fd[sigma_lens])
println("AD Vega  (analytic): ", greeks_ad[sigma_lens])

println("\n--- Greeks (Carr-Madan Method) ---")
greeks_fd_cm = solve(batch_prob, fd, method_carr_madan)
greeks_ad_cm = solve(batch_prob, ad, method_carr_madan)

println("FD Delta (Carr-Madan): ", greeks_fd_cm[spot_lens])
println("AD Delta (Carr-Madan): ", greeks_ad_cm[spot_lens])
println("FD Vega  (Carr-Madan): ", greeks_fd_cm[sigma_lens])
println("AD Vega  (Carr-Madan): ", greeks_ad_cm[sigma_lens])

# -- Benchmarks
println("\n--- Benchmarking AD Greeks ---")
@btime solve($batch_prob, $ad, $method_carr_madan)

delta_prob = GreekProblem(prob, spot_lens)
vega_prob = GreekProblem(prob, sigma_lens)

@btime solve($delta_prob, $ad, $method_carr_madan)
@btime solve($vega_prob, $ad, $method_carr_madan)
