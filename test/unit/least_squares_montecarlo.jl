@testset "LSM regression" begin
    # Define payoff
    strike = 1.0
    expiry = Date(2021, 1, 1)
    american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Call(), Spot())

    # Define market inputs
    reference_date = Date(2020, 1, 1)
    rate = 0.2
    spot = 1.0
    sigma = 0.4
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # LSM method
    dynamics = LognormalDynamics()
    trajectories = 1
    steps = 100
    seeds = [42]
    strategy = BlackScholesExact()  # Deterministic seed
    config = SimulationConfig(trajectories, steps = steps, seeds = seeds)
    degree = 3
    method = LSM(dynamics, strategy, config, degree)

    # Define problem and solve
    prob = PricingProblem(american_payoff, market_inputs)
    sol = solve(prob, method)

    @test isapprox(sol.price, 0.22536692592893945, atol = 1e-8)
end
