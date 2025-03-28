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
