using Revise
using Hedgehog2  # Your library with RectVolSurface, PricingProblem, etc.
using Dates
using Test
using DataInterpolations
# --- Grid definitions
tenors  = [0.25, 0.5, 1.0, 2.0]               # maturities in years
strikes = [80.0, 90.0, 100.0, 110.0]          # strikes

# --- Vol grid: vols[i, j] corresponds to (tenor[i], strike[j])
vols = [
    0.22  0.21  0.20  0.19;
    0.23  0.22  0.21  0.20;
    0.25  0.24  0.23  0.22;
    0.28  0.27  0.26  0.25
]

# --- Market data
reference_date = DateTime(2020, 1, 1)
rate = 0.02
spot = 100.0

# --- Build volatility surface
vol_surface = RectVolSurface(reference_date, tenors, strikes, vols)

# --- Recompute prices from surface
nrows, ncols = size(vols)
recomputed_prices = zeros(nrows, ncols)

for i in 1:nrows, j in 1:ncols
    T = tenors[i]
    K = strikes[j]
    σ = get_vol(vol_surface, T, K)
    
    # Create expiry from year fraction
    expiry = reference_date + Millisecond(round(Int, T * 365 * 86400 * 1000))

    payoff = VanillaOption(K, expiry, European(), Call(), Spot())
    market = BlackScholesInputs(reference_date, rate, spot, σ)

    prob = PricingProblem(payoff, market)
    sol = solve(prob, BlackScholesAnalytic())
    recomputed_prices[i, j] = sol.price
end

# --- Invert surface using prices and DateTime-based expiries
expiry_datetimes = [add_yearfrac(reference_date, T) for T in tenors]
tenor_periods = [expiry - reference_date for expiry in expiry_datetimes]

inverted_surface = RectVolSurface(
    reference_date,
    rate,
    spot,
    tenor_periods,
    strikes,
    recomputed_prices;
    interp_strike = DataInterpolations.LinearInterpolation,
    interp_time = DataInterpolations.LinearInterpolation,
    extrap_strike = ExtrapolationType.Constant,
    extrap_time = ExtrapolationType.Constant,
)

# --- Test recovery of original implied volatilities
@testset "Vol Surface Inversion Consistency (DateTime-safe)" begin
    for i in 1:nrows, j in 1:ncols
        T = tenors[i]
        K = strikes[j]
        original = vols[i, j]
        recovered = get_vol(inverted_surface, T, K)
        @test isapprox(original, recovered; atol=1e-6)
    end
end
