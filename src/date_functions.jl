const SECONDS_IN_YEAR_365 = 365 * 86400
const MILLISECONDS_IN_YEAR_365 = SECONDS_IN_YEAR_365 * 1000
const MILLISECONDS_IN_DAY = 86400000

# --- Time Conversions ---

"""
    to_ticks(x::Date)

Convert a `Date` to milliseconds since the Julia `Dates` module epoch (0000-01-01).

Note: This is calculated by converting the `Date` to days since epoch 
(`Dates.date2epochdays`) and multiplying by `MILLISECONDS_IN_DAY`.
"""
function to_ticks(x::Date)
    return Dates.date2epochdays(x) * MILLISECONDS_IN_DAY
end

"""
    to_ticks(x::DateTime)

Convert a `DateTime` to milliseconds since the Julia `Dates` module epoch (0000-01-01T00:00:00).

Uses `Dates.datetime2epochms`.
"""
function to_ticks(x::DateTime)
    return Dates.datetime2epochms(x)
end

"""
    to_ticks(x::Real)

Assume `x` is already a timestamp in milliseconds since the Julia `Dates` module epoch
(0000-01-01T00:00:00) and return it unchanged.

Used to normalize mixed inputs (e.g., `Date`, `DateTime`, `Real`) to a common
tick representation for calculations like `yearfrac` or `add_yearfrac`.
"""
function to_ticks(x::Real)
    return x
end

# --- ACT/365 Year Fractions ---

"""
    yearfrac(start, stop)

Compute the ACT/365 year fraction between two time points.

Inputs `start` and `stop` can be `Date`, `DateTime`, or ticks (`Int` or `Float64`).
If ticks are provided, they are assumed to be milliseconds since the Julia `Dates`
module epoch (0000-01-01T00:00:00), consistent with the output of `to_ticks`.
"""
function yearfrac(start, stop)
    ms_start = to_ticks(start)
    ms_stop = to_ticks(stop)
    return (ms_stop - ms_start) / MILLISECONDS_IN_YEAR_365
end

"""
    yearfrac(p::AbstractTime)

Compute the ACT/365 year fraction from a `Period` object (e.g., `Year(1)`, `Day(180)`).
It allows also for `CompoundPeriod`, and even Dates, interpreting them as time periods from year 0 of the Gregorian calendar.
Calculates the duration represented by the period in milliseconds and divides by
`MILLISECONDS_IN_YEAR_365`.
"""
function yearfrac(p::Dates.AbstractTime)
    # The choice of reference date here is arbitrary and does not affect the result,
    # as only the difference (duration) matters.
    ref = DateTime(1970, 1, 1) 
    return yearfrac(ref, ref + p)
end

# --- Add Year Fraction to a Date or DateTime ---

"""
    add_yearfrac(t::Real, yf::Real) -> Real

Add a fractional number of years (`yf`, computed as ACT/365) to a timestamp `t`.

The timestamp `t` is assumed to be in milliseconds since the Julia `Dates` module
epoch (0000-01-01T00:00:00). Returns the updated timestamp in milliseconds as `Float64`.

This version is AD-compatible.
"""
function add_yearfrac(t::Real, yf::Real)
    return t + yf * MILLISECONDS_IN_YEAR_365
end

"""
    add_yearfrac(t::TimeType, yf::Real) -> DateTime

Add a fractional number of years (`yf`, computed as ACT/365) to a `Date` or `DateTime` `t`.

Converts `t` to milliseconds since the Julia `Dates` module epoch, adds the
duration corresponding to `yf`, and converts the result back to a `DateTime` object.
"""
function add_yearfrac(t::TimeType, yf::Real)
    # `to_ticks(t)` uses the Julia Dates epoch (Year 0000)
    # `add_yearfrac(Real,Real)` adds the correct millisecond duration
    # `epochms2datetime` converts back from the Julia Dates epoch (Year 0000)
    return Dates.epochms2datetime(add_yearfrac(to_ticks(t), yf))
end