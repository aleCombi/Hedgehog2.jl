### --- 2D Interpolator ---

"""
    Interpolator2D{T, S, U, F}

A 2D interpolation structure where:
- `x_vals`: grid in the x-direction (e.g., expiry),
- `y_vals`: grid in the y-direction (e.g., strike),
- `y_interps`: interpolators in the y-direction, one per x,
- `build_x_interp`: a function `y ↦ x-interpolator` built from applying all y-interpolators at `y`.
"""
struct Interpolator2D{T, S, U, F}
    x_vals::T
    y_vals::S
    y_interps::U
    build_x_interp::F
end

"""
    Interpolator2D(x_vals, y_vals, values; kwargs...)

Returns an `Interpolator2D` using interpolation and extrapolation logic along each axis.
Used for reconstructing updated surfaces after calibration or bumping.
"""
function Interpolator2D(
    x_vals::AbstractVector,
    y_vals::AbstractVector,
    values::AbstractMatrix;
    interp_y = LinearInterpolation,
    interp_x = LinearInterpolation,
    extrap_y = ExtrapolationType.Constant,
    extrap_x = ExtrapolationType.Constant,
)
    @assert size(values, 1) == length(x_vals)
    @assert size(values, 2) == length(y_vals)

    y_interps = [
        interp_y(view(values, i, :), y_vals; extrapolation = extrap_y) for i in eachindex(x_vals)
    ]

    function build_x_interp(y)
        vals_at_y = [itp(y) for itp in y_interps]
        return interp_x(vals_at_y, x_vals; extrapolation = extrap_x)
    end

    return Interpolator2D(x_vals, y_vals, y_interps, build_x_interp)
end

"""
    Base.getindex(itp::Interpolator2D, x, y)

Evaluate the 2D interpolator `itp` at point `(x, y)` using nested interpolation.
"""
function Base.getindex(itp::Interpolator2D, x::Real, y::Real)
    x_interp = itp.build_x_interp(y)
    return x_interp(x)
end

### --- Vol Surfaces ---

"""
    AbstractVolSurface

Abstract supertype for volatility surface representations.
"""
abstract type AbstractVolSurface end

"""
    FlatVolSurface(σ::Real)

Volatility surface with a constant volatility `σ`.
"""
struct FlatVolSurface{T <: Real} <: AbstractVolSurface
    σ::T
end

"""
    get_vol(surf::FlatVolSurface, _, _)

Returns the constant volatility from a `FlatVolSurface`.
"""
function get_vol(surf::FlatVolSurface, ::Any, ::Any)
    return surf.σ
end

"""
    get_vol_yf(surf::FlatVolSurface, _, _)

Returns the constant volatility from a `FlatVolSurface`, used when year fraction is precomputed.
"""
function get_vol_yf(surf::FlatVolSurface, ::Any, ::Any)
    return surf.σ
end

"""
    RectVolSurface(reference_date, interpolator, vols, builder)

Internal constructor for a volatility surface using rectangular interpolation over a grid.
"""
struct RectVolSurface{T<:Real, V<:Real} <: AbstractVolSurface
    reference_date::T
    interpolator::Interpolator2D
    vols::Matrix{V}
    builder::Function
end

"""
    RectVolSurface(reference_date::Date, interpolator::Interpolator2D, vols, builder)

Wraps a tick-based surface into a `RectVolSurface` from a calendar `reference_date`.
"""
function RectVolSurface(
    reference_date::Union{Date, DateTime},
    interpolator::Interpolator2D,
    vols::Matrix,
    builder::Function,
)
    return RectVolSurface(to_ticks(reference_date), interpolator, vols, builder)
end

"""
    RectVolSurface(reference_date, tenors, strikes, vols; kwargs...)

Constructs a rectangular volatility surface by interpolating across `tenors` and `strikes`.
"""
function RectVolSurface(
    reference_date,
    tenors::Vector{T},
    strikes::Vector{S},
    vols::Matrix{V};
    interp_strike = LinearInterpolation,
    interp_time = LinearInterpolation,
    extrap_strike = ExtrapolationType.Constant,
    extrap_time = ExtrapolationType.Constant,
) where {T<:Real, S<:Real, V<:Real}
    builder = function (v, t, k)
        return Interpolator2D(t, k, v;
            interp_y = interp_strike,
            interp_x = interp_time,
            extrap_y = extrap_strike,
            extrap_x = extrap_time,
        )
    end
    itp = builder(vols, tenors, strikes)
    return RectVolSurface(to_ticks(reference_date), itp, vols, builder)
end

"""
    get_vol(surf::RectVolSurface, expiry_date::Date, strike)

Interpolates the implied volatility for a given `expiry_date` and `strike`.
"""
function get_vol(surf::RectVolSurface, expiry_date::Date, strike::Real)
    T = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator[T, strike]
end

"""
    get_vol(surf::RectVolSurface, expiry_time::Real, strike)

Interpolates the implied volatility for a given time to expiry `expiry_time` and `strike`.
"""
function get_vol(surf::RectVolSurface, expiry_date::E, strike) where E <: Real
    T = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator[T, strike]
end

"""
    get_vol_yf(surf::RectVolSurface, t, strike)

Like `get_vol`, but assumes time to expiry `t` is already in year-fraction format.
"""
function get_vol_yf(surf::RectVolSurface, t::Real, strike::Real)
    return surf.interpolator[t, strike]
end

"""
    RectVolSurface(reference_date, rate, spot, tenors, strikes, prices; kwargs...)

Calibrates a `RectVolSurface` from observed option prices.
Handles optional `call_put_matrix` and allows customization of interpolation/extrapolation.
"""
function RectVolSurface(
    reference_date,
    rate,
    spot,
    tenors::AbstractVector,
    strikes::AbstractVector,
    prices::AbstractMatrix;
    call_put_matrix::Union{Nothing, AbstractMatrix} = nothing,
    interp_strike = LinearInterpolation,
    interp_time = LinearInterpolation,
    extrap_strike = ExtrapolationType.Constant,
    extrap_time = ExtrapolationType.Constant,
    initial_guess = 0.02,
    root_finding_algo = nothing,
    kwargs...,
)
    nrows, ncols = length(tenors), length(strikes)
    @assert size(prices) == (nrows, ncols) "Price matrix size must match (length(tenors), length(strikes))"

    if call_put_matrix === nothing
        call_put_matrix = fill(Call(), nrows, ncols)
    else
        @assert size(call_put_matrix) == (nrows, ncols)
    end

    vols = Matrix{Float64}(undef, nrows, ncols)
    accessor = @optic _.market_inputs.sigma
    for i = 1:nrows, j = 1:ncols
        expiry = reference_date + tenors[i]
        strike = strikes[j]
        cp = call_put_matrix[i, j]
        price = prices[i, j]
        payoff = VanillaOption(strike, expiry, European(), cp, Spot())
        market = BlackScholesInputs(reference_date, rate, spot, initial_guess)

        prob = CalibrationProblem(
            BasketPricingProblem([payoff], market),
            BlackScholesAnalytic(),
            [accessor],
            [price],
            [initial_guess]
        )

        sol = solve(prob, RootFinderAlgo(root_finding_algo); kwargs...)
        vols[i, j] = sol.u
        println(sol.u)
    end

    times = [yearfrac(reference_date, reference_date + τ) for τ in tenors]
    return RectVolSurface(to_ticks(reference_date), times, strikes, vols;
        interp_strike = interp_strike,
        interp_time = interp_time,
        extrap_strike = extrap_strike,
        extrap_time = extrap_time,
    )
end

"""
    VolLens(strike, expiry)

Lens structure for accessing and mutating specific vol entries by strike and expiry.
"""
struct VolLens{S, T}
    strike::S
    expiry::T
end

"""
    (lens::VolLens)(prob)

Reads the volatility at a specific strike and expiry from a `CalibrationProblem`.
"""
function (lens::VolLens)(prob)
    sigma = prob.market_inputs.sigma
    return _get_vol_lens(sigma, lens.expiry, lens.strike)
end

"""
    set(prob, lens::VolLens, new_val)

Returns a modified `CalibrationProblem` with the volatility at the lens location updated to `new_val`.
"""
function set(prob, lens::VolLens{S, T}, new_val) where {S, T}
    sigma = prob.market_inputs.sigma
    sigma′ = _set_vol_lens(sigma, lens.expiry, lens.strike, new_val)
    return @set prob.market_inputs.sigma = sigma′
end

"""
    _get_vol_lens(sigma::RectVolSurface, expiry, strike)

Internal helper to extract volatility from a rectangular surface at a given point.
Throws an error if exact match is not found.
"""
function _get_vol_lens(sigma::RectVolSurface, T::Real, K::Real)
    i = findfirst(==(T), sigma.interpolator.x_vals)
    j = findfirst(==(K), sigma.interpolator.y_vals)
    if i === nothing || j === nothing
        error("VolLens: no exact match found for expiry=$T and strike=$K in RectVolSurface.")
    end
    return sigma.vols[i, j]
end

"""
    _set_vol_lens(sigma::RectVolSurface, expiry, strike, new_val)

Internal helper to return a new `RectVolSurface` with one vol entry updated.
Rebuilds the interpolator.
"""
function _set_vol_lens(sigma::RectVolSurface, T::Real, K::Real, new_val)
    i = findfirst(==(T), sigma.interpolator.x_vals)
    j = findfirst(==(K), sigma.interpolator.y_vals)
    if i === nothing || j === nothing
        error("VolLens: cannot set value — expiry=$T or strike=$K not found in RectVolSurface grid.")
    end
    bumped_vols = @set sigma.vols[i, j] = new_val
    new_itp = sigma.builder(bumped_vols, sigma.interpolator.x_vals, sigma.interpolator.y_vals)
    return RectVolSurface(sigma.reference_date, new_itp, bumped_vols, sigma.builder)
end

"""
    _get_vol_lens(sigma::FlatVolSurface, _, _)

Returns the constant volatility for a flat surface.
"""
function _get_vol_lens(sigma::FlatVolSurface, T, K)
    return sigma.σ
end

"""
    _set_vol_lens(sigma::FlatVolSurface, _, _, new_val)

Returns a new `FlatVolSurface` with updated constant volatility.
"""
function _set_vol_lens(sigma::FlatVolSurface, T, K, new_val)
    return FlatVolSurface(new_val)
end
