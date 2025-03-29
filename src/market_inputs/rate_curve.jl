using DataInterpolations
import Dates: Date, value
import Base: getindex
export RateCurve, df, zero_rate, forward_rate, spine_tenors, spine_zeros, FlatRateCurve, is_flat, ZeroRateSpineLens

# -- Curve struct --
struct RateCurve{I}
    reference_date::Real #ticks
    interpolator::I  # Should be a callable interpolating function
end

function RateCurve(reference_date::TimeType, interpolator)
    return RateCurve(to_ticks(reference_date), interpolator)
end

# -- Constructor from discount factors (interpolate in zero rates) --
function RateCurve(
    reference_date::Real,
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

function RateCurve(
    reference_date::TimeType,
    tenors,
    dfs;
    interp = LinearInterpolation,
    extrap = ExtrapolationType.Constant
) 
    return RateCurve(to_ticks(reference_date), tenors, dfs; interp=interp, extrap=extrap)
end

# -- Accessors --
# Accepts ticks (ms since epoch)
df_ticks(curve::RateCurve, ticks::Real) =
    exp(-zero_rate_ticks(curve, ticks) * yearfrac(curve.reference_date, ticks))

# Accepts Date, routes to tick-based version
df(curve::RateCurve, t::Date) =
    df_ticks(curve, to_ticks(t))

# Accepts ticks (ms since epoch)
zero_rate_ticks(curve::RateCurve, ticks::Real) =
    curve.interpolator(yearfrac(curve.reference_date, ticks))

# Accepts daycounts (already in year fractions)
zero_rate(curve::RateCurve, t::Date) = zero_rate_ticks(curve, to_ticks(t))

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

struct ZeroRateSpineLens
    i::Int
end

# accessors for greeks
import Accessors: set, @optic

# Getter
function (lens::ZeroRateSpineLens)(prob)
    return spine_zeros(prob.market.rate)[lens.i]
end

# Setter (rebuilds the rate curve with updated zero rate at index `i`)
function set(prob, lens::ZeroRateSpineLens, new_zᵢ)
    curve = prob.market.rate
    t = spine_tenors(curve)
    z = spine_zeros(curve)
    dfs = @. exp(-z * t)
    
    # Update only the i-th discount factor with new_zᵢ
    dfs_bumped = @set dfs[lens.i] = exp(-new_zᵢ * t[lens.i])

    new_curve = RateCurve(curve.reference_date, t, dfs_bumped)
    return @set prob.market.rate = new_curve
end
