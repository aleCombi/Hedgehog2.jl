using Test
using Dates
using Hedgehog2
using Optimization
using Accessors

@testset "Black-Scholes Calibration" begin
    reference_date = Date(2020, 1, 1)
    r, S0, sigma = 0.05, 100.0, 0.25
    market_inputs = BlackScholesInputs(reference_date, r, S0, sigma)

    strikes = collect(60.0:5.0:140.0)
    expiry = reference_date + Day(365)
    payoffs = [VanillaOption(K, expiry, European(), Call(), Spot()) for K in strikes]

    quotes = [solve(PricingProblem(p, market_inputs), BlackScholesAnalytic()).price for p in payoffs]

    accessors = [@optic(_.market.sigma)]
    initial_guess = [0.15]

    basket = BasketPricingProblem(payoffs, market_inputs)
    calib = CalibrationProblem(basket, BlackScholesAnalytic(), accessors, quotes)
    result = solve(calib, initial_guess)

    @test isapprox(result.u[1], sigma; atol=1e-5)
end

@testset "Heston Calibration" begin
    reference_date = Date(2020, 1, 1)
    true_params = (v0 = 0.010201, κ = 6.21, θ = 0.019, σ = 0.61, ρ = -0.7)
    r, S0 = 0.0319, 100.0

    market_inputs = HestonInputs(reference_date, r, S0,
        true_params.v0, true_params.κ, true_params.θ, true_params.σ, true_params.ρ)

    strikes = collect(60.0:5.0:140.0)
    expiries = [reference_date + Day(d) for d in (90, 180, 365)]
    payoffs = [VanillaOption(K, expiry, European(), Call(), Spot()) for K in strikes, expiry in expiries] |> vec

    α, boundary = 1.0, 32.0
    method_heston = CarrMadan(α, boundary, HestonDynamics())

    quotes = [solve(PricingProblem(p, market_inputs), method_heston).price for p in payoffs]

    accessors = [
        @optic(_.market.v0),
        @optic(_.market.κ),
        @optic(_.market.θ),
        @optic(_.market.σ),
        @optic(_.market.ρ),
    ]

    initial_guess = [0.02, 3.0, 0.03, 0.4, -0.3]
    basket = BasketPricingProblem(payoffs, market_inputs)
    calib = CalibrationProblem(basket, method_heston, accessors, quotes)
    result = solve(calib, initial_guess; allow_f_increases = true)

    @test isapprox(result.u[1], true_params.v0; atol=1e-5)
    @test isapprox(result.u[2], true_params.κ;  atol=1e-2)
    @test isapprox(result.u[3], true_params.θ;  atol=1e-4)
    @test isapprox(result.u[4], true_params.σ;  atol=1e-3)
    @test isapprox(result.u[5], true_params.ρ;  atol=1e-4)
end
