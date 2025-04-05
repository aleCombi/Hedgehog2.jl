struct Interpolator2D{XVec,YVec,YInterp,XInterpFunc}
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
        interp_y(view(values, i, :), y_vals; extrapolation = extrap_y) for
        i in eachindex(x_vals)
    ]

    # Closure: build interpolator along x-direction at a given y
    function build_x_interp(y::Real)
        vals_at_y = [itp(y) for itp in y_interps]
        return interp_x(vals_at_y, x_vals; extrapolation = extrap_x)
    end

    return Interpolator2D(x_vals, y_vals, y_interps, build_x_interp)
end

# Accessor: surf[x, y]
function Base.getindex(itp::Interpolator2D, x::Real, y::Real)
    x_interp = itp.build_x_interp(y)
    return x_interp(x)
end

struct RectVolSurface
    reference_date::Any
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

function RectVolSurface(
    reference_date,
    rate::Real,
    spot::Real,
    tenors::Vector{<:Period},
    strikes::Vector{<:Real},
    prices::Matrix{<:Real};
    call_put_matrix::Union{Nothing,AbstractMatrix} = nothing,
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

    # Fill call/put matrix if not provided
    if call_put_matrix === nothing
        call_put_matrix = fill(Call(), nrows, ncols)
    else
        @assert size(call_put_matrix) == (nrows, ncols) "Call/Put matrix must match price matrix size"
    end

    # Calibrate implied vols
    vols = Matrix{Float64}(undef, nrows, ncols)
    accessor = @optic _.market_inputs.sigma
    for i = 1:nrows, j = 1:ncols
        expiry = reference_date + tenors[i]
        strike = strikes[j]
        cp = call_put_matrix[i, j]
        price = prices[i, j]
        payoff = VanillaOption(strike, expiry, European(), cp, Spot())
        market = BlackScholesInputs(reference_date, rate, spot, initial_guess)
        @show payoff.expiry

        println(solve(PricingProblem(payoff, market), BlackScholesAnalytic()).price)
        prob = CalibrationProblem(
            BasketPricingProblem([payoff], market),
            BlackScholesAnalytic(),
            [accessor],
            [price],
            [initial_guess]
        )

        sol = solve(prob, RootFinderAlgo(root_finding_algo); kwargs...)

        println(solve(PricingProblem(payoff, @set market.sigma = 0.22), BlackScholesAnalytic()).price)
        println(solve(PricingProblem(payoff, @set market.sigma = sol.u), BlackScholesAnalytic()).price)

        @show sol.u
        vols[i, j] = sol.u
    end

    # Convert periods to year fractions
    times = [yearfrac(reference_date, reference_date + τ) for τ in tenors]

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
