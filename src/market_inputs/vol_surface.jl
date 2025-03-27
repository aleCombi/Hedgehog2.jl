using Interpolations
using Dates
using Accessors
using Roots 

export RectVolSurface, spine_strikes, spine_tenors, spine_vols, get_vol
struct RectVolSurface{I <: AbstractInterpolation}
    reference_date
    interpolator::I
end

# Accessors for grid info
function spine_tenors(surf::RectVolSurface)
    return surf.interpolator.itp.knots[1]
end

function spine_strikes(surf::RectVolSurface)
    return surf.interpolator.itp.knots[2]
end

function spine_vols(surf::RectVolSurface)
    return surf.interpolator.itp.coefs
end

# Constructor with interpolation options
function RectVolSurface(
    reference_date,
    tenors::AbstractVector,
    strikes::AbstractVector,
    vols::AbstractMatrix;
    interp_type = Gridded(Linear()),
    extrap_type = Flat()
)
    itp = interpolate((tenors, strikes), vols, interp_type)
    ext = extrapolate(itp, extrap_type)
    return RectVolSurface(reference_date, ext)
end

# Lookup by date
function get_vol(surf::RectVolSurface, expiry_date, strike)
    Tfrac = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator(Tfrac, strike)
end

# Lookup by year fraction
function get_vol(surf::RectVolSurface, t_fraction::Number, strike)
    return surf.interpolator(t_fraction, strike)
end