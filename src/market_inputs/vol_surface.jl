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
    Interpolator2D(x_vals, y_vals, values; ...)

Constructs a 2D interpolator by first interpolating along y for each row of `values`,
then creating an x-interpolator from the resulting values.

Interpolation and extrapolation methods can be specified for both dimensions.
"""
function Interpolator2D(
    x_vals::AbstractVector{X},
    y_vals::AbstractVector{Y},
    values::AbstractMatrix{V};
    interp_y = LinearInterpolation,
    interp_x = LinearInterpolation,
    extrap_y = ExtrapolationType.Constant,
    extrap_x = ExtrapolationType.Constant,
) where {X,Y,V}
    @assert size(values, 1) == length(x_vals)
    @assert size(values, 2) == length(y_vals)

    y_interps = [
        interp_y(view(values, i, :), y_vals; extrapolation = extrap_y) for
        i in eachindex(x_vals)
    ]

    # Closure: build interpolator along x-direction at a given y
    function build_x_interp(y::Y)
        vals_at_y = [itp(y) for itp in y_interps]
        return interp_x(vals_at_y, x_vals; extrapolation = extrap_x)
    end

    return Interpolator2D(
        x_vals, y_vals, y_interps, build_x_interp
    )
end

"""
    Base.getindex(itp::Interpolator2D, x, y)

Evaluate the 2D interpolator at point `(x, y)`.
"""
function Base.getindex(itp::Interpolator2D, x::Real, y::Real)
    x_interp = itp.build_x_interp(y)
    return x_interp(x)
end

"""
    RectVolSurface{T}

A volatility surface parameterized by expiry (x) and strike (y), internally backed
by an `Interpolator2D`. The reference date is stored in tick format.
"""
struct RectVolSurface{T<:Real}
    reference_date::T
    interpolator::Interpolator2D
end

"""
    RectVolSurface(reference_date::Date, interpolator::Interpolator2D)

Convenience constructor that accepts a `Date` or `DateTime` as `reference_date`,
and converts it to internal tick format.
"""
function RectVolSurface(
    reference_date::Union{Date, DateTime},
    interpolator::Interpolator2D
)
    return RectVolSurface(to_ticks(reference_date), interpolator)
end

"""
    RectVolSurface(reference_date, tenors, strikes, vols; ...)

Construct a volatility surface from a grid of `vols` given:

- `reference_date`: in ticks or date-like (user responsibility),
- `tenors`: grid in time direction (e.g., time-to-maturity),
- `strikes`: grid in strike direction,
- `vols`: implied volatility values.

Interpolation and extrapolation methods can be specified per axis.
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

"""
    get_vol(surf::RectVolSurface, expiry_date::Date, strike::Real)

Retrieve the interpolated implied volatility at a calendar expiry date and strike.
"""
function get_vol(surf::RectVolSurface, expiry_date::Date, strike::Real)
    T = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator[T, strike]
end

"""
    get_vol(surf::RectVolSurface, expiry_ticks::Real, strike)

Retrieve the interpolated implied volatility at expiry given in ticks and strike.
"""
function get_vol(surf::RectVolSurface, expiry_date::E, strike) where E <: Real
    T = yearfrac(surf.reference_date, expiry_date)
    return surf.interpolator[T, strike]
end

"""
    get_vol_yf(surf::RectVolSurface, t::Real, strike::Real)

Retrieve the interpolated implied volatility using a time-to-expiry (year fraction) and strike.
"""
function get_vol_yf(surf::RectVolSurface, t::Real, strike::Real)
    return surf.interpolator[t, strike]
end

"""
    RectVolSurface(reference_date, rate, spot, tenors, strikes, prices; ...)

Construct a volatility surface by calibrating implied volatilities from observed prices.

Steps:
1. If `call_put_matrix` is not provided, defaults to calls.
2. Builds a matrix of implied volatilities using root-finding.
3. Converts `tenors` to year fractions.
4. Constructs a `RectVolSurface` using interpolated vols.

# Arguments
- `reference_date`: Can be date or ticks.
- `rate`: Flat rate used in calibration.
- `spot`: Spot price of the underlying.
- `tenors`: Vector of `Period`s.
- `strikes`: Vector of strikes.
- `prices`: Matrix of observed option prices.
- `call_put_matrix`: Optional matrix of `Call()`/`Put()` types.
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
        @assert size(call_put_matrix) == (nrows, ncols) "Call/Put matrix must match price matrix size"
    end

    # Step 1: Calibrate implied vols
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
    end

    # Step 2: Convert tenors to year fractions
    times = [yearfrac(reference_date, reference_date + τ) for τ in tenors]

    # Step 3: Construct RectVolSurface with interpolated vols
    return RectVolSurface(
        reference_date,
        times,
        strikes,
        vols;
        interp_strike = interp_strike,
        interp_time = interp_time,
        extrap_strike = extrap_strike,
        extrap_time = extrap_time,
    )
end
