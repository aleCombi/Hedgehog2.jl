using Test
using Hedgehog2
using Accessors
import Accessors: @optic

@testset "Greek agreement: ForwardAD vs FiniteDifference" begin
    # Setup
    strike = 1.2
    expiry = Date(2021, 1, 1)
    reference_date = Date(2020, 1, 1)
    spot = 1.0
    sigma = 0.4
    rate = 0.2

    payoff = VanillaOption(strike, expiry, European(), Put(), Forward())
    market = BlackScholesInputs(reference_date, rate, spot, sigma)
    prob = PricingProblem(payoff, market)

    method = BlackScholesAnalytic()
    ε = 1e-4

    # First-order Greeks
    @testset "First-order: Vega" begin
        lens = @optic _.market.sigma
        gprob = GreekProblem(prob, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol = 1e-5)
    end

    @testset "First-order: Delta" begin
        lens = @optic _.market.spot
        gprob = GreekProblem(prob, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol = 1e-5)
    end

    # Second-order Greeks
    @testset "Second-order: Gamma" begin
        lens = @optic _.market.spot
        gprob = SecondOrderGreekProblem(prob, lens, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol = 1e-5)
    end

    @testset "Second-order: Volga" begin
        lens = @optic _.market.sigma
        gprob = SecondOrderGreekProblem(prob, lens, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol = 1e-5)
    end
end

using Test
using Hedgehog2
using Dates
using Accessors
import Accessors: @optic

@testset "Greeks Agreement Test" begin
    strike = 1.0
    expiry = Date(2021, 1, 1)
    reference_date = Date(2020, 1, 1)
    rate = 0.03
    spot = 1.0
    sigma = 1.0

    underlying = Hedgehog2.Forward()
    payoff = VanillaOption(strike, expiry, European(), Call(), underlying)
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
    pricing_prob = PricingProblem(payoff, market_inputs)
    bs_method = BlackScholesAnalytic()

    vol_lens = @optic _.market.sigma
    spot_lens = @optic _.market.spot

    # Vega
    gprob = GreekProblem(pricing_prob, vol_lens)
    vega_ad = solve(gprob, ForwardAD(), bs_method).greek
    vega_fd = solve(gprob, FiniteDifference(1e-4), bs_method).greek
    vega_an = solve(gprob, AnalyticGreek(), bs_method).greek
    @test isapprox(vega_ad, vega_fd; rtol = 1e-5)
    @test isapprox(vega_ad, vega_an; rtol = 1e-5)

    # Gamma
    gammaprob = SecondOrderGreekProblem(pricing_prob, spot_lens, spot_lens)
    gamma_ad = solve(gammaprob, ForwardAD(), bs_method).greek
    gamma_fd = solve(gammaprob, FiniteDifference(1e-4), bs_method).greek
    gamma_an = solve(gammaprob, AnalyticGreek(), bs_method).greek
    @test isapprox(gamma_ad, gamma_fd; rtol = 1e-5)
    @test isapprox(gamma_ad, gamma_an; rtol = 1e-5)

    # Volga
    volgaprob = SecondOrderGreekProblem(pricing_prob, vol_lens, vol_lens)
    volga_ad = solve(volgaprob, ForwardAD(), bs_method).greek
    volga_fd = solve(volgaprob, FiniteDifference(1e-4), bs_method).greek
    volga_an = solve(volgaprob, AnalyticGreek(), bs_method).greek
    @test isapprox(volga_ad, volga_fd; rtol = 1e-3)
    @test isapprox(volga_ad, volga_an; rtol = 1e-5)

    # Theta (no analytic)
    thetaprob = GreekProblem(pricing_prob, @optic _.payoff.expiry)
    theta_ad = solve(thetaprob, ForwardAD(), bs_method).greek
    theta_fd = solve(thetaprob, FiniteDifference(1), bs_method).greek
    theta_analytic = solve(thetaprob, AnalyticGreek(), bs_method).greek
    @test isapprox(theta_ad, theta_fd; rtol = 5e-3)
    @test isapprox(theta_ad, theta_analytic; rtol = 1e-8)
end

using Test
using Hedgehog2
using Dates
using Accessors
import Accessors: @optic
using DataInterpolations

@testset "Zero Rate Deltas: ForwardAD vs FiniteDifference" begin
    # Define the vanilla option
    strike = 1.0
    expiry = Date(2020, 4, 2)
    underlying = Hedgehog2.Forward()
    payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

    # Reference date and market inputs
    reference_date = Date(2020, 1, 1)
    spot = 1.0
    sigma = 1.0
    rates = [0.03, 0.032, 0.07, 0.042, 0.03]

    # Create multi-tenor curve with explicit interpolation builder
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
    dfs = @. exp(-rates * tenors)
    interp_fn =
        (u, t) -> QuadraticInterpolation(u, t; extrapolation = ExtrapolationType.Constant)
    rate_curve = RateCurve(reference_date, tenors, dfs; interp = interp_fn)

    # Construct pricing problem with interpolated curve
    market_inputs = BlackScholesInputs(reference_date, rate_curve, spot, sigma)
    prob = PricingProblem(payoff, market_inputs)

    # Choose Greek and pricing methods
    pricing_method = BlackScholesAnalytic()
    ad_method = ForwardAD()
    fd_method = FiniteDifference(1e-5)

    # Loop through each zero rate pillar and compare AD vs FD
    for i = 1:length(spine_zeros(rate_curve))
        lens = ZeroRateSpineLens(i)

        g_ad = solve(GreekProblem(prob, lens), ad_method, pricing_method).greek
        g_fd = solve(GreekProblem(prob, lens), fd_method, pricing_method).greek
        @test isapprox(g_ad, g_fd; rtol = 1e-6, atol = 1e-10) ||
              @warn "Mismatch at zero rate spine index $i" ad = g_ad fd = g_fd
    end
end

