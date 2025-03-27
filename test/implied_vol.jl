using Revise, Hedgehog2, Interpolations, Dates, Test

@testset "Implied Vol Calibration Consistency" begin
    reference_date = Date(2020, 1, 1)
    expiry_date = reference_date + Day(365)
    r = 0.02
    spot = 100.0
    true_vol = 0.65
    strike = 80.0

    # Step 1: Generate market price from known vol
    market_inputs = BlackScholesInputs(reference_date, r, spot, true_vol)
    payoff = VanillaOption(strike, expiry_date, European(), Call(), Spot())
    pricing_problem = PricingProblem(payoff, market_inputs)
    price = solve(pricing_problem, BlackScholesAnalytic()).price

    # Step 2: Calibrate vol to recover implied from price
    market_inputs_new = BlackScholesInputs(reference_date, r, spot, 9000.8) # the vol value is not used
    pricing_problem_new = PricingProblem(payoff, market_inputs_new)
    calib_problem = Hedgehog2.BlackScholesCalibrationProblem(pricing_problem, BlackScholesAnalytic(), price)
    calib_solution = solve(calib_problem)
    implied_vol = calib_solution.u

    @test isapprox(implied_vol, true_vol; atol=1e-8)
end

using Revise, Hedgehog2, Interpolations, Dates, Test

@testset "Vol Surface Inversion Consistency (DateTime-safe)" begin
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
        expiry = add_yearfrac(reference_date, T)

        payoff = VanillaOption(K, expiry, European(), Call(), Spot())
        market = BlackScholesInputs(reference_date, rate, spot, σ)

        prob = PricingProblem(payoff, market)
        sol = solve(prob, BlackScholesAnalytic())
        recomputed_prices[i, j] = sol.price
    end

    # --- Invert surface using recomputed prices
    expiry_datetimes = [add_yearfrac(reference_date, T) for T in tenors]
    tenor_periods = [expiry - reference_date for expiry in expiry_datetimes]

    inverted_surface = RectVolSurface(reference_date, rate, spot, tenor_periods, strikes, recomputed_prices)

    # --- Check recovery of original volatilities
    for i in 1:nrows, j in 1:ncols
        T = tenors[i]
        K = strikes[j]
        original = vols[i, j]
        recovered = get_vol(inverted_surface, T, K)
        @test isapprox(original, recovered; atol=1e-6)
    end
end
