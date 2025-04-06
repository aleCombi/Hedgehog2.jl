using Test
using Dates

@testset "Time Conversion and ACT/365 Utilities" begin
    # Setup
    d1 = Date(2020, 1, 1)
    d2 = d1 + Day(365)
    dt1 = DateTime(d1)
    dt2 = DateTime(d2)
    ticks1 = to_ticks(d1)
    ticks2 = to_ticks(d2)
    
    # to_ticks
    @test to_ticks(d1) == Dates.datetime2epochms(DateTime(d1))
    @test to_ticks(dt1) == Dates.datetime2epochms(dt1)
    @test to_ticks(123456789.0) === 123456789.0

    # yearfrac between dates
    yf = yearfrac(d1, d2)
    @test isapprox(yf, 1.0; atol=1e-8)

    # yearfrac with ticks
    @test isapprox(yearfrac(ticks1, ticks2), 1.0; atol=1e-8)

    # yearfrac with Period
    @test isapprox(yearfrac(Month(12)), 1.0; atol=1e-8)

    # add_yearfrac (Real, Real)
    t0 = 1_580_000_000_000.0  # some ms timestamp
    t1 = add_yearfrac(t0, 1.0)
    @test isapprox((t1 - t0), 365 * 86400 * 1000; atol=1e-3)

    # add_yearfrac (DateTime)
    ref_date = DateTime(2020, 1, 1)
    shifted = add_yearfrac(ref_date, 1.0)
    expected = ref_date + Day(365)
    @test Date(shifted) == Date(expected)
end
