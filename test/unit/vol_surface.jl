@testset "Interpolator2D" begin
    x = [0.25, 0.5]
    y = [90.0, 100.0]
    z = [0.2 0.25; 0.22 0.27]

    itp = Interpolator2D(
        x, y, z;
        interp_y = LinearInterpolation,
        interp_x = LinearInterpolation,
        extrap_y = ExtrapolationType.Constant,
        extrap_x = ExtrapolationType.Constant,
    )

    # Interpolation inside the grid
    @test isapprox(itp[0.25, 90.0], 0.2; atol=1e-8)
    @test isapprox(itp[0.5, 100.0], 0.27; atol=1e-8)

    # Interpolation at midpoint
    @test isapprox(itp[0.375, 95.0], 0.235; atol=1e-4)

    # Extrapolation (constant)
    @test isapprox(itp[0.1, 85.0], 0.2; atol=1e-8)     # bottom-left corner
    @test isapprox(itp[0.6, 105.0], 0.27; atol=1e-8)   # top-right corner

    @test isa(itp, Interpolator2D)
end


using Test
using Dates
using DataInterpolations

@testset "RectVolSurface construction from interpolator" begin
    ticks = to_ticks(Date(2024, 1, 1))
    x = [0.25, 0.5]
    y = [90.0, 100.0]
    vols = [0.2 0.25; 0.22 0.27]
    itp = Interpolator2D(x, y, vols)

    surf = RectVolSurface(ticks, itp)
    @test isa(surf, RectVolSurface)
    @test isapprox(get_vol_yf(surf, 0.5, 100.0), 0.27; atol=1e-8)
end

@testset "RectVolSurface construction from Date" begin
    ref_date = Date(2024, 1, 1)
    x = [0.25, 0.5]
    y = [90.0, 100.0]
    vols = [0.2 0.25; 0.22 0.27]
    itp = Interpolator2D(x, y, vols)

    surf = RectVolSurface(ref_date, itp)
    T = yearfrac(to_ticks(ref_date), to_ticks(ref_date)) + 0.5
    @test isa(surf, RectVolSurface)
    @test isapprox(get_vol(surf, to_ticks(ref_date) + round(Int, T * 365.25 * 24 * 60 * 60 * 1000), 100.0), 0.27; atol=1e-8)
end

@testset "get_vol by Date and by ticks" begin
    ref_date = Date(2024, 1, 1)
    x = [0.25, 0.5]
    y = [90.0, 100.0]
    vols = [0.2 0.25; 0.22 0.27]
    itp = Interpolator2D(x, y, vols)
    surf = RectVolSurface(ref_date, itp)

    expiry_date = ref_date + Year(1)
    @test isapprox(get_vol(surf, expiry_date, 100.0), surf.interpolator[1.0, 100.0]; atol=1e-8)

    ticks = to_ticks(expiry_date)
    @test isapprox(get_vol(surf, ticks, 100.0), surf.interpolator[1.0, 100.0]; atol=1e-8)
end
