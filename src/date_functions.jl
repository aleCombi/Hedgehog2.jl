export yearfrac, add_yearfrac

const SECONDS_IN_YEAR_365 = 365 * 86400

function yearfrac(start::S, stop::E) where {S<:TimeType, E<:TimeType}
    ms_start = Dates.datetime2epochms(DateTime(start))
    ms_stop = Dates.datetime2epochms(DateTime(stop))
    return (ms_stop - ms_start) / (365 * 86400 * 1000)
end

const MILLISECONDS_IN_YEAR_365 = 365 * 86400 * 1000

"Add a fractional number of years (ACT/365) to a DateTime."
function add_yearfrac(dt::D, yf::Real) where D<:TimeType
    ms = Dates.datetime2epochms(dt)
    new_ms = ms + round(Int, yf * MILLISECONDS_IN_YEAR_365)
    return Dates.epochms2datetime(new_ms)
end

"Compute ACT/365 year fraction from a Period (e.g., Day, Month, CompoundPeriod)."
function yearfrac(p::Period)
    ref = DateTime(1970, 1, 1)
    return yearfrac(ref, ref + p)
end