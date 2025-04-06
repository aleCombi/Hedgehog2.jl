# -- Structs --
abstract type AbstractRateCurve end

struct RateCurve{F, R <:Real, I <:DataInterpolations.AbstractInterpolation } <: AbstractRateCurve
    reference_date::R # ticks
    interpolator::I    # callable interpolation function
    builder::F           # function (u, t) -> interpolator
end

struct FlatRateCurve{R <: Number, S <: Number} <: AbstractRateCurve
    reference_date::R
    rate::S
end

# -- Flat Curve --

function FlatRateCurve(rate; reference_date = Date(0))
    return FlatRateCurve(to_ticks(reference_date), rate)
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

df(curve::R, ticks::T) where {T <: Number, R <: AbstractRateCurve} =
    exp(-zero_rate(curve, ticks) * yearfrac(curve.reference_date, ticks))

df(curve::R, t::D) where {R <: AbstractRateCurve, D <:TimeType} = df(curve, to_ticks(t))

df_yf(curve::R, yf::T) where {R <: AbstractRateCurve, T <: Number} = exp(-zero_rate_yf(curve, yf) * yf)

zero_rate(curve::RateCurve, ticks::T) where T <: Number =
    curve.interpolator(yearfrac(curve.reference_date, ticks))

zero_rate(curve::FlatRateCurve, ticks::T) where T <: Number =
    curve.rate

zero_rate(curve::R, t::Date) where R <: AbstractRateCurve = zero_rate(curve, to_ticks(t))

zero_rate_yf(curve::RateCurve, yf::R) where R <: Number = curve.interpolator(yf)
zero_rate_yf(curve::FlatRateCurve, yf::R) where R <: Number = curve.rate

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

# -- Lens for bumping --

struct ZeroRateSpineLens
    i::Int
end

function (lens::ZeroRateSpineLens)(prob::PricingProblem{<:Any, <:BlackScholesInputs{<:FlatRateCurve}})
    return prob.market_inputs.rate.rate
end

function (lens::ZeroRateSpineLens)(prob::PricingProblem{<:Any, <:BlackScholesInputs{<:RateCurve}})
    return spine_zeros(prob.market_inputs.rate)[lens.i]
end


function set(prob, lens::ZeroRateSpineLens, new_zᵢ)
    return _set_rate_curve(prob, lens, new_zᵢ, prob.market_inputs.rate)
end

# Interpolated curve: mutate zero vector and rebuild
function _set_rate_curve(prob, lens::ZeroRateSpineLens, new_zᵢ, curve::RateCurve)
    t = spine_tenors(curve)
    z = spine_zeros(curve)
    z_bumped = @set z[lens.i] = new_zᵢ
    new_itp = curve.builder(z_bumped, t)
    new_curve = InterpolatedRateCurve(curve.reference_date, new_itp, curve.builder)
    return @set prob.market_inputs.rate = new_curve
end

# Flat curve: just return new FlatRateCurve with new value
function _set_rate_curve(prob, lens::ZeroRateSpineLens, new_zᵢ, curve::FlatRateCurve)
    new_curve = FlatRateCurve(curve.reference_date, new_zᵢ)
    return @set prob.market_inputs.rate = new_curve
end

spine_zeros(curve::FlatRateCurve) = [curve.rate]
spine_tenors(curve::FlatRateCurve) = [0.0]  # Or however you define the time axis
