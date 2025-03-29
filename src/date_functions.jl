export yearfrac, add_yearfrac

const SECONDS_IN_YEAR_365 = 365 * 86400
const MILLISECONDS_IN_YEAR_365 = SECONDS_IN_YEAR_365 * 1000

# --- Time Conversions ---

"Convert a `Date`, `DateTime`, or ticks (Int/Float64) to milliseconds since epoch."
to_ticks(x::Date) = Dates.datetime2epochms(DateTime(x))
to_ticks(x::DateTime) = Dates.datetime2epochms(x)
to_ticks(x::Real) = x  # Already Float64-compatible

# --- ACT/365 Year Fractions ---

"""
    yearfrac(start, stop)

Compute the ACT/365 year fraction between two time points.

Supports `Date`, `DateTime`, or ticks (`Int` or `Float64`).
"""
function yearfrac(start, stop)
    ms_start = to_ticks(start)
    ms_stop = to_ticks(stop)
    return (ms_stop - ms_start) / MILLISECONDS_IN_YEAR_365
end

"""
    yearfrac(p::Period)

Compute the ACT/365 year fraction from a `Period` object.
"""
function yearfrac(p::Period)
    ref = DateTime(1970, 1, 1)
    return yearfrac(ref, ref + p)
end

# --- Add Year Fraction to a Date or DateTime ---

"""
    add_yearfrac(t::Real, yf::Real) -> Real

Add a fractional number of years (ACT/365) to a timestamp in milliseconds since epoch.
Returns the updated timestamp in milliseconds as `Float64`.

This version is AD-compatible.
"""
add_yearfrac(t::Real, yf::Real) = t + yf * MILLISECONDS_IN_YEAR_365

# Date-based (returns DateTime)
add_yearfrac(t::TimeType, yf::Real) =
    Dates.epochms2datetime(add_yearfrac(to_ticks(t), yf))
