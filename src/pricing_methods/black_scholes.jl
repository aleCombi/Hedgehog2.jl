"""
    BlackScholesAnalytic <: AbstractPricingMethod

Analytical Black-Scholes pricing method.

Represents the closed-form Black-Scholes model for pricing European-style vanilla options.
Assumes lognormal dynamics for the underlying asset with constant volatility and interest rate.
"""
struct BlackScholesAnalytic <: AbstractPricingMethod end

"""
    log_dynamics(::BlackScholesAnalytic) -> LognormalDynamics

Returns the assumed price dynamics under the Black-Scholes model.

This corresponds to lognormal dynamics with constant volatility and risk-free rate.
"""
function log_dynamics(::BlackScholesAnalytic)
    return LognormalDynamics()
end

"""
    solve(prob::PricingProblem{VanillaOption{European}}, ::BlackScholesAnalytic) -> AnalyticSolution

Computes the price of a European vanilla option under the Black-Scholes model.

# Arguments
- `prob::PricingProblem`: The pricing problem, including the payoff and market inputs.
- `BlackScholesAnalytic`: Marker for the analytic pricing method.

# Returns
- `AnalyticSolution`: The priced solution under Black-Scholes assumptions.

# Notes
- Uses the forward measure formulation.
- Falls back to intrinsic value if volatility is zero.
"""
function solve(
    prob::PricingProblem{VanillaOption{TS,TE,European,B,C}, BlackScholesInputs},
    ::BlackScholesAnalytic,
) where {TS,TE,B,C}

    payoff = prob.payoff
    market = prob.market_inputs

    K = payoff.strike
    σ = market.sigma
    cp = payoff.call_put()
    T = yearfrac(market.referenceDate, payoff.expiry)
    D = df(market.rate, payoff.expiry)
    F = market.spot / D

    price = if σ == 0
        D * payoff(F)
    else
        sqrtT = sqrt(T)
        d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
        d2 = d1 - σ * sqrtT
        N = Normal()
        D * cp * (F * cdf(N, cp * d1) - K * cdf(N, cp * d2))
    end

    return AnalyticSolution(price)
end

# Define the calibration problem
struct BlackScholesCalibrationProblem{P<:PricingProblem,M}
    prob::P
    method::M
    price_target::Any
end

function solve(
    calib::BlackScholesCalibrationProblem;
    initial_guess = 0.2,
    lower = 1e-6,
    upper = 5.0,
    kwargs...,
)
    function calibration_function(σ, p)
        market = calib.prob.market_inputs
        new_market = @set market.sigma = σ
        new_prob = PricingProblem(calib.prob.payoff, new_market)
        sol = solve(new_prob, calib.method)
        return sol.price - calib.price_target
    end

    problem = NonlinearProblem(calibration_function, initial_guess)
    solution = NonlinearSolve.solve(problem; kwargs...)

    return solution
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

    for i = 1:nrows, j = 1:ncols
        expiry = reference_date + tenors[i]
        strike = strikes[j]
        cp = call_put_matrix[i, j]
        price = prices[i, j]

        payoff = VanillaOption(strike, expiry, European(), cp, Spot())
        market = BlackScholesInputs(reference_date, rate, spot, 0.2)

        prob = BlackScholesCalibrationProblem(
            PricingProblem(payoff, market),
            BlackScholesAnalytic(),
            price,
        )

        sol = solve(prob; kwargs...)
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
