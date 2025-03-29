using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2020, 1, 2)
underlying = Hedgehog2.Forward()

euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# ------------------------------
# Pricing method
# ------------------------------
bs_method = BlackScholesAnalytic()

# ------------------------------
# Define lenses
# ------------------------------
vol_lens = @optic _.market.sigma
spot_lens = @optic _.market.spot

# ------------------------------
# Vega (1st order w.r.t. sigma)
# ------------------------------
gprob = Hedgehog2.GreekProblem(euro_pricing_prob, vol_lens)
vega_ad = solve(gprob, ForwardAD(), bs_method).greek
vega_fd = solve(gprob, FiniteDifference(1e-4), bs_method).greek

# ------------------------------
# Gamma (2nd order w.r.t. spot)
# ------------------------------
gammaprob = Hedgehog2.SecondOrderGreekProblem(euro_pricing_prob, spot_lens, spot_lens)
gamma_ad = solve(gammaprob, ForwardAD(), bs_method).greek
gamma_fd = solve(gammaprob, FiniteDifference(1e-4), bs_method).greek

# ------------------------------
# Volga (2nd order w.r.t. sigma)
# ------------------------------
volgaprob = Hedgehog2.SecondOrderGreekProblem(euro_pricing_prob, vol_lens, vol_lens)
volga_ad = solve(volgaprob, ForwardAD(), bs_method).greek
volga_fd = solve(volgaprob, FiniteDifference(1e-4), bs_method).greek

# ------------------------------
# Print results
# ------------------------------
println("Vega (Forward AD): $vega_ad")
println("Vega (Finite Diff): $vega_fd\n")

println("Gamma (Forward AD): $gamma_ad")
println("Gamma (Finite Diff): $gamma_fd\n")

println("Volga (Forward AD): $volga_ad")
println("Volga (Finite Diff): $volga_fd")

println("\n--- Zero Rate Deltas (per tenor) ---")

# Create a rate curve with multiple tenors
tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
dfs = @. exp(-rate * tenors)
rate_curve = RateCurve(reference_date, tenors, dfs)

# Replace scalar rate in market_inputs with the curve
market_inputs_with_curve = BlackScholesInputs(reference_date, rate_curve, spot, sigma)
curve_prob = PricingProblem(euro_payoff, market_inputs_with_curve)

# Choose method and Greek method
greek_method = ForwardAD()
pricing_method = bs_method

# Compute zero deltas (∂Price/∂zᵢ)
spine_len = length(spine_zeros(rate_curve))
zero_deltas = [
    solve(
        GreekProblem(curve_prob, ZeroRateSpineLens(i)),
        greek_method,
        pricing_method
    ).greek
    for i in 1:spine_len
]

# Print result
for (i, d) in enumerate(zero_deltas)
    t = spine_tenors(rate_curve)[i]
    println("Tenor $(t)y: ∂Price/∂z[$i] = ", round(d, sigdigits=6))
end

vega_an = solve(gprob, AnalyticGreek(), bs_method).greek

gamma_an = solve(gammaprob, AnalyticGreek(), bs_method).greek
volga_an = solve(volgaprob, AnalyticGreek(), bs_method).greek

println("Vega (Analytic): $vega_an")
println("Vega (Forward AD): $vega_ad")
println("Vega (Finite Diff): $vega_fd\n")

println("Gamma (Analytic): $gamma_an")
println("Gamma (Forward AD): $gamma_ad")
println("Gamma (Finite Diff): $gamma_fd\n")

println("Volga (Analytic): $volga_an")
println("Volga (Forward AD): $volga_ad")
println("Volga (Finite Diff): $volga_fd")

# ------------------------------
# Theta (1st order w.r.t. expiry)
# ------------------------------
thetaproblem = Hedgehog2.GreekProblem(euro_pricing_prob, @optic _.payoff.expiry)
theta_ad = solve(thetaproblem, ForwardAD(), bs_method).greek
theta_fd = solve(thetaproblem, FiniteDifference(1), bs_method).greek
# theta_an = solve(thetaproblem, AnalyticGreek(), bs_method).greek

# println("Theta (Analytic):     $theta_an")
println("Theta (Forward AD):  $theta_ad")
println("Theta (Finite Diff): $theta_fd")
