@testset "LSM regression" begin
    # define payoff
    strike = 1.0
    expiry = Date(2021, 1, 1)
    american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Call(), Spot())

    # define market inputs
    reference_date = Date(2020, 1, 1)
    rate = 0.2
    spot = 1.0
    sigma = 0.4
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # LSM pricer
    dynamics = LognormalDynamics()
    trajectories = 1
    steps = 100
    strategy = BlackScholesExact(trajectories, steps; seed=42) #we fix the seed for deterministic results
    degree = 3
    lsm = LSM(dynamics, strategy, degree)
    lsm_american_price = Pricer(american_payoff, market_inputs, lsm)()

    @test isapprox(lsm_american_price, 0.19533448129762399, atol=1E-8)
end