using Dates
using Accessors
using Roots 

export RectVolSurface, spine_strikes, spine_tenors, spine_vols, get_vol, Interpolator2D

import DataInterpolations: LinearInterpolation, ExtrapolationType

struct Interpolator2D{XVec, YVec, YInterp, XInterpFunc}
    x_vals::XVec                        # Grid in x-direction (e.g., expiry)
    y_vals::YVec                        # Grid in y-direction (e.g., strike)
    y_interps::Vector{YInterp}         # Interpolators in y, one per x
    build_x_interp::XInterpFunc        # Callable: y ↦ x-interpolator
end

function Interpolator2D(
    x_vals::Vector{<:Real},
    y_vals::Vector{<:Real},
    values::Matrix{<:Real};
    interp_y = LinearInterpolation,
    interp_x = LinearInterpolation,
    extrap_y = ExtrapolationType.Constant,
    extrap_x = ExtrapolationType.Constant,
)
    @assert size(values, 1) == length(x_vals)
    @assert size(values, 2) == length(y_vals)

    # Interpolators along y-direction (row-wise)
    y_interps = [
        interp_y(view(values, i, :), y_vals; extrapolation=extrap_y)
        for i in eachindex(x_vals)
    ]

    # Closure: build interpolator along x-direction at a given y
    function build_x_interp(y::Real)
        vals_at_y = [itp(y) for itp in y_interps]
        return interp_x(vals_at_y, x_vals; extrapolation=extrap_x)
    end

    return Interpolator2D(x_vals, y_vals, y_interps, build_x_interp)
end

# Accessor: surf[x, y]
function Base.getindex(itp::Interpolator2D, x::Real, y::Real)
    x_interp = itp.build_x_interp(y)
    return x_interp(x)
end

struct RectVolSurface
    reference_date
    interpolator::Interpolator2D
end

function RectVolSurface(
    reference_date,
    tenors::AbstractVector{<:Real},
    strikes::AbstractVector{<:Real},
    vols::AbstractMatrix{<:Real};
    interp_strike = LinearInterpolation,
    interp_time = LinearInterpolation,
    extrap_strike = ExtrapolationType.Constant,
    extrap_time = ExtrapolationType.Constant,
)
    itp = Interpolator2D(
        tenors,
        strikes,
        vols;
        interp_y = interp_strike,
        interp_x = interp_time,
        extrap_y = extrap_strike,
        extrap_x = extrap_time,
    )
    return RectVolSurface(reference_date, itp)
end

# Lookup using expiry date
function get_vol(surf::RectVolSurface, expiry_date::Date, strike::Real)
    T = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator[T, strike]
end

# Lookup using year fraction
function get_vol(surf::RectVolSurface, t::Real, strike::Real)
    return surf.interpolator[t, strike]
end
