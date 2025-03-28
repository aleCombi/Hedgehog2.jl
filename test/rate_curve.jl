using Revise
using Test
using Dates
using Hedgehog2  # Replace with your module name
using DataInterpolations

@testset "RateCurve interpolation at spine points" begin
    # Input data
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dfs = [0.995, 0.990, 0.980, 0.955, 0.890, 0.750]
    ref_date = Date(2025, 1, 1)

    # Expected zero rates
    expected_zr = @. -log(dfs) / tenors

    # Build the curve
    curve = Hedgehog2.RateCurve(ref_date, tenors, dfs;
        interp = LinearInterpolation,
        extrap = ExtrapolationType.Constant,
    )

    # Test discount factors at each tenor
    for (t, df_expected) in zip(tenors, dfs)
        @test isapprox(Hedgehog2.df(curve, t), df_expected; atol=1e-12)
    end

    # Test zero rates at each tenor
    for (t, zr_expected) in zip(tenors, expected_zr)
        @test isapprox(Hedgehog2.zero_rate(curve, t), zr_expected; atol=1e-12)
    end
end

@testset "FlatRateCurve correctness" begin
    r = 0.025  # 2.5% flat rate
    curve = Hedgehog2.FlatRateCurve(r)

    # Check scalar zero_rate and df
    for t in [0.1, 1.0, 5.0]
        @test isapprox(Hedgehog2.zero_rate(curve, t), r; atol=1e-12)
        @test isapprox(Hedgehog2.df(curve, t), exp(-r * t); atol=1e-12)
    end

    # Check vectorized input
    ts = [0.25, 1.0, 3.0]
    expected_zr = fill(r, length(ts))
    expected_df = @. exp(-r * ts)

    @test Hedgehog2.zero_rate.(Ref(curve), ts) == expected_zr
    @test isapprox.(Hedgehog2.df.(Ref(curve), ts), expected_df; atol=1e-6) |> all

    # Check Date input
    t0 = Date(2025, 1, 1)
    t1 = Date(2026, 1, 1)
    τ = Dates.value(t1 - t0) / 365

    curve = Hedgehog2.FlatRateCurve(r; reference_date=t0)

    @test isapprox(Hedgehog2.zero_rate(curve, t1), r; atol=1e-12)
    @test isapprox(Hedgehog2.df(curve, t1), exp(-r * τ); atol=1e-12)
end
