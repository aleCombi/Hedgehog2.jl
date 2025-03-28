using DataInterpolations
import Dates: Date, value
import Base: getindex
export RateCurve, df, zero_rate, forward_rate, spine_tenors, spine_zeros, FlatRateCurve, is_flat

# -- Curve struct --
struct RateCurve{I}
    reference_date::Date
    interpolator::I  # Should be a callable interpolating function
end

# -- Constructor from discount factors (interpolate in zero rates) --
function RateCurve(
    reference_date::Date,
    tenors,
    dfs;
    interp = LinearInterpolation,
    extrap = ExtrapolationType.Constant
)
    @assert length(tenors) == length(dfs) "Mismatched tenor/DF lengths"
    @assert issorted(tenors) "Tenors must be sorted"

    zr = @. -log(dfs) / tenors  # continuous zero rate
    itp = interp(zr, tenors; extrapolation=extrap)
    return RateCurve(reference_date, itp)
end

# -- Accessors --
df(curve::RateCurve, t::Real) = exp(-zero_rate(curve, t) * t)
df(curve::RateCurve, t::Date) = df(curve, yearfrac(curve.reference_date, t))

zero_rate(curve::RateCurve, t::Real) = curve.interpolator(t)
zero_rate(curve::RateCurve, t::Date) = zero_rate(curve, yearfrac(curve.reference_date, t))

# -- Forward rate between two times --
function forward_rate(curve::RateCurve, t1::Real, t2::Real)
    df1 = df(curve, t1)
    df2 = df(curve, t2)
    return log(df1 / df2) / (t2 - t1)
end

forward_rate(curve::RateCurve, d1::Date, d2::Date) =
    forward_rate(curve, yearfrac(curve.reference_date, d1), yearfrac(curve.reference_date, d2))

# -- Diagnostic accessors --
spine_tenors(curve::RateCurve) = curve.interpolator.t
spine_zeros(curve::RateCurve) = curve.interpolator.u

function FlatRateCurve(r; reference_date=Date(0)) 
    itp = DataInterpolations.ConstantInterpolation([r], [0]; extrapolation=ExtrapolationType.Constant)
    return RateCurve(reference_date, itp)
end

function is_flat(curve::RateCurve)
    length(spine_tenors(curve)) == 1
end
