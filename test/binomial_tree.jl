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

@testset "American binomial tree regression" begin
    # call option of spot
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
    steps = 80
    crr_method = CoxRossRubinsteinMethod(steps)
    crr_price = Pricer(american_payoff, market_inputs, crr_method)()

    @test isapprox(crr_price, 0.25735029973418333, atol=1E-8)

    # put option of forward
    american_payoff_fwd = VanillaOption(strike, expiry, Hedgehog2.American(), Put(), Forward())
    crr_price_fwd = Pricer(american_payoff_fwd, market_inputs, crr_method)()
    @test isapprox(crr_price_fwd, 0.07410498956582845, atol=1E-8)
end