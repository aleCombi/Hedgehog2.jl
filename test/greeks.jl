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
        lens = @optic _.marketinputs.σ
        gprob = GreekProblem(prob, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol=1e-5)
    end

    @testset "First-order: Delta" begin
        lens = @optic _.marketinputs.spot
        gprob = GreekProblem(prob, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol=1e-5)
    end

    # Second-order Greeks
    @testset "Second-order: Gamma" begin
        lens = @optic _.marketinputs.spot
        gprob = SecondOrderGreekProblem(prob, lens, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol=1e-5)
    end

    @testset "Second-order: Volga" begin
        lens = @optic _.marketinputs.σ
        gprob = SecondOrderGreekProblem(prob, lens, lens)

        ad_val = solve(gprob, ForwardAD(), method).greek
        fd_val = solve(gprob, FiniteDifference(ε), method).greek

        @test isapprox(ad_val, fd_val; rtol=1e-5)
    end
end
