const SECONDS_IN_YEAR_365 = 365 * 86400
const MILLISECONDS_IN_YEAR_365 = SECONDS_IN_YEAR_365 * 1000

# --- Time Conversions ---

"""
    to_ticks(x::Date)

Convert a `Date` to milliseconds since the Unix epoch.
"""
function to_ticks(x::Date)
    return Dates.datetime2epochms(DateTime(x))
end

"""
    to_ticks(x::DateTime)

Convert a `DateTime` to milliseconds since the Unix epoch.
"""
function to_ticks(x::DateTime)
    return Dates.datetime2epochms(x)
end

"""
    to_ticks(x::Real)

Assume `x` is already a timestamp in milliseconds (e.g., `Float64` or `Int`).
Used to normalize mixed inputs.
"""
function to_ticks(x::Real)
    return x
end

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
function add_yearfrac(t::Real, yf::Real)
    return t + yf * MILLISECONDS_IN_YEAR_365
end

"""
    add_yearfrac(t::TimeType, yf::Real) -> DateTime

Add a fractional number of years (ACT/365) to a `Date` or `DateTime`.
Returns a `DateTime` object.
"""
function add_yearfrac(t::TimeType, yf::Real)
    return Dates.epochms2datetime(add_yearfrac(to_ticks(t), yf))
end
