using DataInterpolations
import Dates: Date, value
import Base: getindex
import Accessors: set, @optic

export RateCurve,
    df,
    zero_rate,
    forward_rate,
    spine_tenors,
    spine_zeros,
    FlatRateCurve,
    is_flat,
    ZeroRateSpineLens

# -- Structs --

struct RateCurve{F}
    reference_date::Real # ticks
    interpolator::DataInterpolations.AbstractInterpolation     # callable interpolation function
    builder::F           # function (u, t) -> interpolator
end

struct ZeroRateSpineLens
    i::Int
end


# -- Constructors --

function RateCurve(
    reference_date::Real,
    tenors::AbstractVector,
    dfs;
    interp = (u, t) ->
        LinearInterpolation(u, t; extrapolation = ExtrapolationType.Constant),
)
    @assert length(tenors) == length(dfs) "Mismatched tenor/DF lengths"
    @assert issorted(tenors) "Tenors must be sorted"

    zr = @. -log(dfs) / tenors  # continuous zero rate
    itp = interp(zr, tenors)
    return RateCurve(reference_date, itp, interp)
end

function RateCurve(
    reference_date::Date,
    tenors::AbstractVector,
    dfs;
    interp = (u, t) ->
        LinearInterpolation(u, t; extrapolation = ExtrapolationType.Constant),
)
    return RateCurve(to_ticks(reference_date), tenors, dfs; interp = interp)
end

function RateCurve(reference_date::Date, itp::I, builder::F) where {I,F}
    return RateCurve(to_ticks(reference_date), itp, builder)
end

# -- Accessors --

df(curve::RateCurve, ticks::Real) =
    exp(-zero_rate(curve, ticks) * yearfrac(curve.reference_date, ticks))

df(curve::RateCurve, t::Date) = df(curve, to_ticks(t))

df_yf(curve::RateCurve, yf::Real) = exp(-zero_rate_yf(curve, yf) * yf)

zero_rate(curve::RateCurve, ticks::Real) =
    curve.interpolator(yearfrac(curve.reference_date, ticks))

zero_rate(curve::RateCurve, t::Date) = zero_rate(curve, to_ticks(t))

zero_rate_yf(curve::RateCurve, yf::Real) = curve.interpolator(yf)

# -- Forward Rates --

function forward_rate(curve::RateCurve, t1::Real, t2::Real)
    df1 = df(curve, t1)
    df2 = df(curve, t2)
    return log(df1 / df2) / (t2 - t1)
end

forward_rate(curve::RateCurve, d1::Date, d2::Date) = forward_rate(
    curve,
    yearfrac(curve.reference_date, d1),
    yearfrac(curve.reference_date, d2),
)

# -- Spine Access --

spine_tenors(curve::RateCurve) = curve.interpolator.t
spine_zeros(curve::RateCurve) = curve.interpolator.u

# -- Flat Curve --

function FlatRateCurve(r; reference_date = Date(0))
    builder =
        (u, t) -> ConstantInterpolation(u, t; extrapolation = ExtrapolationType.Constant)
    itp = builder([r], [0.0])
    return RateCurve(to_ticks(reference_date), itp, builder)
end

function is_flat(curve::RateCurve)
    length(spine_tenors(curve)) == 1
end

# -- Lens for bumping --

function (lens::ZeroRateSpineLens)(prob)
    return spine_zeros(prob.market.rate)[lens.i]
end

function set(prob, lens::ZeroRateSpineLens, new_zᵢ)
    curve = prob.market.rate
    t = spine_tenors(curve)
    z = spine_zeros(curve)

    # Rebuild bumped zero rates
    z_bumped = @set z[lens.i] = new_zᵢ
    new_itp = curve.builder(z_bumped, t)
    new_curve = RateCurve(curve.reference_date, new_itp, curve.builder)

    return @set prob.market.rate = new_curve
end
