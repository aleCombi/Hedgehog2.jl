    @testset "Payoff" begin
    @testset "Vanilla Option" begin
        reference_date = Date(2020,1,1)
        expiry = reference_date + Day(365)
        call = VanillaOption(100.0, expiry, European(), Call(), Spot())
        put  = VanillaOption(100.0, expiry, European(), Put(), Spot())

        @test call(120.0) ≈ 20.0
        @test put(120.0) ≈ 0.0
        @test put(80.0) ≈ 20.0

        # Put-call parity
        call_price = call(100.0)
        parity_put = parity_transform(call_price, put, 100.0, FlatRateCurve(1.0
    ; reference_date=reference_date))
        @test parity_put ≈ call_price - 100.0 + 100.0 * exp(-1.0)

        # Type check
        @test call isa VanillaOption
        @test call.call_put() == 1.0
        @test put.call_put() == -1.0
    end
end