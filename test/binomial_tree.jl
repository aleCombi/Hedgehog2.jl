@testset "Binomial tree vs black scholes analytical" begin
    # define payoff
    strike = 1.1
    expiry = Date(2021, 1, 1)
    euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

    # define market inputs
    reference_date = Date(2020, 1, 1)
    rate = 0.2
    spot = 1
    sigma = 0.4
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # create analytical black scholes pricer
    analytical_price = Pricer(euro_payoff, market_inputs, BlackScholesAnalytic())()

    # create Cox Ross Rubinstein pricer
    steps = 100
    crr = CoxRossRubinsteinMethod(steps)
    crr_euro_price = Pricer(euro_payoff, market_inputs, crr)()

    @test isapprox(analytical_price, crr_euro_price; atol=1E-3)
end