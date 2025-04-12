using Hedgehog2: SECONDS_IN_YEAR_365, MILLISECONDS_IN_YEAR_365, MILLISECONDS_IN_DAY, to_ticks, yearfrac, add_yearfrac

# --- Test Constants ---
@testset "Constants Definition" begin
    # These remain unchanged
    @test SECONDS_IN_YEAR_365 == 365 * 86400
    @test SECONDS_IN_YEAR_365 == 31536000

    @test MILLISECONDS_IN_YEAR_365 == SECONDS_IN_YEAR_365 * 1000
    @test MILLISECONDS_IN_YEAR_365 == 31536000000

    @test MILLISECONDS_IN_DAY == 86400 * 1000
    @test MILLISECONDS_IN_DAY == 86400000
end

# --- Test to_ticks Conversions (Epoch: 0000-01-01T00:00:00) ---
@testset "to_ticks Conversions (Epoch: 0000-01-01)" begin
    # Define the actual offset for the Unix epoch relative to Julia's epoch
    const UNIX_EPOCH_DAYS_OFFSET = Dates.date2epochdays(Date(1970, 1, 1)) # Should be 719163
    const UNIX_EPOCH_MS_OFFSET = Dates.datetime2epochms(DateTime(1970, 1, 1)) # Should be 62135596800000

    @testset "to_ticks(::Date)" begin
        # Julia epoch (Year 0000) starts day 0
        @test to_ticks(Date(0, 1, 1)) == 0
        @test to_ticks(Date(0, 1, 2)) == 1 * MILLISECONDS_IN_DAY

        # Test the date corresponding to the Unix epoch
        unix_epoch_date = Date(1970, 1, 1)
        @test Dates.date2epochdays(unix_epoch_date) == UNIX_EPOCH_DAYS_OFFSET
        @test to_ticks(unix_epoch_date) == UNIX_EPOCH_DAYS_OFFSET * MILLISECONDS_IN_DAY

        # A specific date (this test remains valid as it uses Dates internal directly)
        d = Date(2023, 10, 27)
        expected_days = Dates.date2epochdays(d)
        @test to_ticks(d) == expected_days * MILLISECONDS_IN_DAY
    end

    @testset "to_ticks(::DateTime)" begin
        # Julia epoch (Year 0000)
        @test to_ticks(DateTime(0, 1, 1, 0, 0, 0, 0)) == 0

        # One second after Julia epoch
        @test to_ticks(DateTime(0, 1, 1, 0, 0, 1, 0)) == 1000

        # One day after Julia epoch
        @test to_ticks(DateTime(0, 1, 2, 0, 0, 0, 0)) == MILLISECONDS_IN_DAY

        # Test the DateTime corresponding to the Unix epoch
        unix_epoch_datetime = DateTime(1970, 1, 1, 0, 0, 0, 0)
        @test Dates.datetime2epochms(unix_epoch_datetime) == UNIX_EPOCH_MS_OFFSET
        @test to_ticks(unix_epoch_datetime) == UNIX_EPOCH_MS_OFFSET

        # Specific DateTime (this test remains valid)
        dt = DateTime(2023, 10, 27, 12, 30, 15, 500)
        expected_ms = Dates.datetime2epochms(dt)
        @test to_ticks(dt) == expected_ms
    end

    @testset "to_ticks(::Real)" begin
        # This subsection remains unchanged as it's epoch-independent
        @test to_ticks(123456789) == 123456789
        @test to_ticks(0) == 0 # Note: A tick value of 0 now corresponds to 0000-01-01
        @test to_ticks(-1000) == -1000 # Corresponds to slightly before 0000-01-01

        @test to_ticks(12345.678) ≈ 12345.678
        @test to_ticks(0.0) ≈ 0.0
        @test to_ticks(-987.65) ≈ -987.65
    end
end

# --- Test yearfrac Calculations ---
@testset "yearfrac Calculations" begin
    # This whole section should remain unchanged.
    # yearfrac calculates differences or uses differences based on Period,
    # which are independent of the absolute epoch zero point.
    @testset "yearfrac(start, stop)" begin
        d1 = Date(2023, 1, 1)
        d2 = Date(2024, 1, 1) # Exactly 365 days later
        dt1 = DateTime(2023, 1, 1, 12, 0, 0)
        dt2 = DateTime(2024, 1, 1, 12, 0, 0) # Exactly 365 days later

        # Zero duration
        @test yearfrac(d1, d1) ≈ 0.0
        @test yearfrac(dt1, dt1) ≈ 0.0

        # Exactly one non-leap year (365 days)
        @test yearfrac(d1, d2) ≈ 1.0
        @test yearfrac(dt1, dt2) ≈ 1.0

        # Mixed types (Date, DateTime)
        # This calculation relies on the underlying Dates functions producing consistent ticks
        @test yearfrac(d1, dt2) ≈ (Dates.datetime2epochms(dt2) - Dates.date2epochdays(d1) * MILLISECONDS_IN_DAY) / MILLISECONDS_IN_YEAR_365
        @test yearfrac(d1, DateTime(2024, 1, 1, 0, 0, 0)) ≈ 1.0 # Easier check

        # Half year (182.5 days in ACT/365)
        d_half = Date(2023, 7, 2) # 182 days after Jan 1st
        dt_half = DateTime(2023, 7, 2, 12, 0, 0) # 182.5 days after Jan 1st 00:00
        @test yearfrac(d1, d_half) ≈ 182 / 365.0
        @test yearfrac(dt1, dt_half) ≈ (182 * MILLISECONDS_IN_DAY + Dates.datetime2epochms(DateTime(0,1,1,12,0,0))) / MILLISECONDS_IN_YEAR_365 # Check calculation carefully
        # Alternative check for dt_half:
        expected_yf_dt_half = (to_ticks(dt_half) - to_ticks(dt1)) / MILLISECONDS_IN_YEAR_365
        @test yearfrac(dt1, dt_half) ≈ expected_yf_dt_half


        # Using ticks directly (epoch doesn't matter for difference)
        t1 = to_ticks(dt1) # Now based on Year 0000 epoch
        t2 = to_ticks(dt2) # Now based on Year 0000 epoch
        @test yearfrac(t1, t2) ≈ 1.0
        @test yearfrac(t1, t1) ≈ 0.0

        # Negative duration
        @test yearfrac(d2, d1) ≈ -1.0
        @test yearfrac(dt2, dt1) ≈ -1.0
        @test yearfrac(t2, t1) ≈ -1.0
    end

    @testset "yearfrac(::Period)" begin
        # These tests remain valid as they calculate the duration of the period
        @test yearfrac(Year(1)) ≈ 1.0 # Relies on yearfrac(ref, ref + p) difference
        @test yearfrac(Day(365)) ≈ 1.0
        @test yearfrac(Day(1)) ≈ 1.0 / 365.0
        @test yearfrac(Day(0)) ≈ 0.0
        @test yearfrac(Millisecond(MILLISECONDS_IN_YEAR_365)) ≈ 1.0
        @test yearfrac(Millisecond(MILLISECONDS_IN_DAY)) ≈ 1.0 / 365.0

        p = Year(1) + Day(182)
        @test yearfrac(p) ≈ 547.0 / 365.0

        @test yearfrac(Day(-365)) ≈ -1.0
    end
end

# --- Test add_yearfrac ---
@testset "add_yearfrac Operations" begin
    # This section remains unchanged.
    # It adds a duration (calculated from yf) to an absolute tick value.
    # The operation is independent of where the tick '0' is located.
    @testset "add_yearfrac(::Real, ::Real)" begin
        t0 = to_ticks(DateTime(2023, 1, 1, 0, 0, 0)) # Tick value is now relative to Year 0000

        @test add_yearfrac(t0, 0.0) ≈ t0
        @test add_yearfrac(t0, 1.0) ≈ t0 + MILLISECONDS_IN_YEAR_365
        @test add_yearfrac(t0, 0.5) ≈ t0 + 0.5 * MILLISECONDS_IN_YEAR_365
        @test add_yearfrac(t0, -1.0) ≈ t0 - MILLISECONDS_IN_YEAR_365

        @test isa(add_yearfrac(t0, 1.0), Float64)
        @test isa(add_yearfrac(Float64(t0), 0.5), Float64)
    end

    @testset "add_yearfrac(::TimeType, ::Real)" begin
        d1 = Date(2023, 1, 1)
        dt1 = DateTime(2023, 1, 1, 12, 0, 0)

        # All these calculations are relative and remain correct
        @test add_yearfrac(d1, 0.0) == DateTime(2023, 1, 1, 0, 0, 0)
        @test add_yearfrac(dt1, 0.0) == dt1

        @test add_yearfrac(d1, 1.0) == DateTime(2024, 1, 1, 0, 0, 0)
        @test add_yearfrac(dt1, 1.0) == DateTime(2024, 1, 1, 12, 0, 0)

        d_leap_start = Date(2024, 1, 1)
        dt_leap_start = DateTime(2024, 1, 1, 6, 0, 0)
        @test add_yearfrac(d_leap_start, 1.0) == DateTime(2025, 1, 1, 0, 0, 0)
        @test add_yearfrac(dt_leap_start, 1.0) == DateTime(2025, 1, 1, 6, 0, 0)

        expected_ms_d1_half = to_ticks(d1) + 0.5 * MILLISECONDS_IN_YEAR_365
        @test add_yearfrac(d1, 0.5) == Dates.epochms2datetime(expected_ms_d1_half)
        @test add_yearfrac(d1, 0.5) == DateTime(2023, 7, 2, 12, 0, 0)

        expected_ms_dt1_half = to_ticks(dt1) + 0.5 * MILLISECONDS_IN_YEAR_365
        @test add_yearfrac(dt1, 0.5) == Dates.epochms2datetime(expected_ms_dt1_half)
        @test add_yearfrac(dt1, 0.5) == DateTime(2023, 7, 3, 0, 0, 0)

        @test add_yearfrac(d1, -1.0) == DateTime(2022, 1, 1, 0, 0, 0)
        @test add_yearfrac(dt1, -0.5) == DateTime(2022, 7, 3, 0, 0, 0)

        @test isa(add_yearfrac(d1, 0.1), DateTime)
        @test isa(add_yearfrac(dt1, 0.1), DateTime)
    end
end