    @testset "Payoff" begin
    @testset "Vanilla Option" begin
        call = VanillaOption(100.0, 1.0, European(), Call(), Spot())
        put  = VanillaOption(100.0, 1.0, European(), Put(), Spot())

        @test call(120.0) ≈ 20.0
        @test put(120.0) ≈ 0.0
        @test put(80.0) ≈ 20.0

        # Put-call parity
        call_price = call(100.0)
        parity_put = parity_transform(call_price, put, 100.0, FlatRateCurve(1.0))
        @test parity_put ≈ call_price - 100.0 + 100.0 * exp(-1.0)

        # Type check
        @test call isa VanillaOption
        @test call.call_put() == 1.0
        @test put.call_put() == -1.0
    end
end