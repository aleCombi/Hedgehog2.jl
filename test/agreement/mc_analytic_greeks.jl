
using Test
using Hedgehog2
using Accessors
using Dates
using Random

@testset "Monte Carlo vs Analytic Greeks (Black-Scholes)" begin
    # --------------------------
    # Setup
    # --------------------------
    strike = 1.0
    expiry = Date(2021, 1, 1)
    reference_date = Date(2020, 1, 1)
    rate = 0.03
    spot = 1.0
    sigma = 1.0

    underlying = Spot()
    payoff = VanillaOption(strike, expiry, European(), Call(), underlying)
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
    prob = PricingProblem(payoff, market_inputs)

    # --------------------------
    # Lenses
    # --------------------------
    vol_lens = VolLens(1,1)
    spot_lens = @optic _.market_inputs.spot
    rate_lens = ZeroRateSpineLens(1)

    # --------------------------
    # Methods with fixed seeds
    # --------------------------
    trajectories = 100_000
    rng = MersenneTwister(42)
    seeds = rand(rng, 1:10^9, trajectories)
    mc_method =
        MonteCarlo(LognormalDynamics(), BlackScholesExact(), SimulationConfig(trajectories; seeds = seeds))
    analytic_method = BlackScholesAnalytic()

    # --------------------------
    # Price Comparison
    # --------------------------
    price_mc = solve(prob, mc_method).price
    price_an = solve(prob, analytic_method).price
    @test isapprox(price_mc, price_an; rtol = 3e-2)

    # --------------------------
    # Delta
    # --------------------------
    gprob = GreekProblem(prob, spot_lens)
    delta_mc = solve(gprob, ForwardAD(), mc_method).greek
    delta_an = solve(gprob, AnalyticGreek(), analytic_method).greek
    @test isapprox(delta_mc, delta_an; rtol = 3e-2)

    # --------------------------
    # Gamma (FD due to AD instability)
    # --------------------------
    gprob2 = SecondOrderGreekProblem(prob, spot_lens, spot_lens)
    gamma_mc = solve(gprob2, FiniteDifference(1E-1), mc_method).greek
    gamma_an = solve(gprob2, AnalyticGreek(), analytic_method).greek
    @test isapprox(gamma_mc, gamma_an; rtol = 2e-1)

    # --------------------------
    # Vega
    # --------------------------
    gprob = GreekProblem(prob, vol_lens)
    vega_mc = solve(gprob, ForwardAD(), mc_method).greek
    vega_an = solve(gprob, AnalyticGreek(), analytic_method).greek
    @test isapprox(vega_mc, vega_an; rtol = 1e-1)

    # --------------------------
    # Rho (flat curve, first zero rate)
    # --------------------------
    gprob = GreekProblem(prob, rate_lens)
    rho_mc = solve(gprob, ForwardAD(), mc_method).greek
    rho_an = solve(gprob, ForwardAD(), analytic_method).greek
    @test isapprox(rho_mc, rho_an; rtol = 1e-2)
end
