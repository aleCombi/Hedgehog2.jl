using Revise
using Hedgehog
using Dates
using Test
using DataInterpolations
using Accessors
@testset "Implied Vol Calibration Consistency" begin
    reference_date = Date(2020, 1, 1)
    expiry_date = reference_date + Year(1)
    r = 0.02
    spot = 100.0
    true_vol = 0.65
    strike = 80.0

    # Step 1: Price from known vol
    market_inputs = BlackScholesInputs(reference_date, r, spot, true_vol)
    payoff = VanillaOption(strike, expiry_date, European(), Call(), Spot())
    pricing_problem = PricingProblem(payoff, market_inputs)
    price = Hedgehog.solve(pricing_problem, BlackScholesAnalytic()).price

    # Step 2: Recover implied vol with new-style calibration
    dummy_inputs = BlackScholesInputs(reference_date, r, spot, 0.2)  # dummy vol
    basket_prob = BasketPricingProblem([payoff], dummy_inputs)

    calib_problem = CalibrationProblem(
        basket_prob,
        BlackScholesAnalytic(),
        [VolLens(1,1)],
        [price],
        [0.2],
    )

    implied_vol = Hedgehog.solve(calib_problem, RootFinderAlgo()).u

    @test isapprox(implied_vol, true_vol; atol = 1e-8)
end


@testset "Vol Surface Inversion Consistency (New Style, DateTime-safe)" begin
    # --- Grid definitions
    tenors = [0.25, 0.5, 1.0, 2.0]
    strikes = [80.0, 90.0, 100.0, 110.0]

    vols = [
        0.22 0.21 0.20 0.19
        0.23 0.22 0.21 0.20
        0.25 0.24 0.23 0.22
        0.28 0.27 0.26 0.25
    ]

    reference_date = DateTime(2020, 1, 1)
    rate = 0.02
    spot = 100.0

    vol_surface = RectVolSurface(reference_date, tenors, strikes, vols)

    nrows, ncols = size(vols)
    recomputed_prices = zeros(nrows, ncols)

    for i = 1:nrows, j = 1:ncols
        T = tenors[i]
        K = strikes[j]
        σ = get_vol_yf(vol_surface, T, K)
        expiry = add_yearfrac(reference_date, T)

        payoff = VanillaOption(K, expiry, European(), Call(), Spot())
        market = BlackScholesInputs(reference_date, rate, spot, σ)

        prob = PricingProblem(payoff, market)
        sol = Hedgehog.solve(prob, BlackScholesAnalytic())
        recomputed_prices[i, j] = sol.price
    end

    # Convert DateTimes to tenor Periods for inversion
    expiry_datetimes = [add_yearfrac(reference_date, T) for T in tenors]
    tenor_periods = [expiry - reference_date for expiry in expiry_datetimes]

    # Invert vol surface via implied vol recovery (new-style)
    inverted_surface = RectVolSurface(
        reference_date,
        rate,
        spot,
        tenor_periods,
        strikes,
        recomputed_prices;
        interp_strike = LinearInterpolation,
        interp_time = LinearInterpolation,
        extrap_strike = ExtrapolationType.Constant,
        extrap_time = ExtrapolationType.Constant,
    )

    for i = 1:nrows, j = 1:ncols
        T = tenors[i]
        K = strikes[j]
        original = vols[i, j]
        recovered = get_vol_yf(inverted_surface, T, K)
        @test isapprox(original, recovered; atol = 1e-6)
    end
end
