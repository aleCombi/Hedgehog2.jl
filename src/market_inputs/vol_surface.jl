using Interpolations
using Dates

struct RectVolSurface{I <: AbstractInterpolation}
    reference_date::Date
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
    reference_date::Date,
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
function getvol(surf::RectVolSurface, expiry_date::Date, strike)
    Tfrac = Dates.value(expiry_date - surf.reference_date) / 365.0
    return surf.interpolator(Tfrac, strike)
end

# Lookup by year fraction
function getvol(surf::RectVolSurface, t_fraction::Number, strike)
    return surf.interpolator(t_fraction, strike)
end
