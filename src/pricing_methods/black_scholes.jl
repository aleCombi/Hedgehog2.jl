# Black-Scholes pricing function for European vanilla options
import DifferentialEquations
using NonlinearSolve, Distributions, Roots

export BlackScholesAnalytic, implied_vol

"""
    BlackScholesAnalytic <: AbstractPricingMethod

The Black-Scholes pricing method.

Represents the analytical Black-Scholes model for pricing European-style vanilla options. 
Assumes the underlying follows lognormal dynamics with constant volatility and interest rate.
"""
struct BlackScholesAnalytic <: AbstractPricingMethod end

"""
    log_dynamics(::BlackScholesAnalytic) -> LognormalDynamics

Returns the assumed lognormal price dynamics for the Black-Scholes model.
"""
function log_dynamics(::BlackScholesAnalytic)
  return LognormalDynamics()
end

"""
    solve(prob::PricingProblem, ::BlackScholesAnalytic) -> AnalyticSolution

Computes the price of a European vanilla option under Black-Scholes.
Returns an `AnalyticSolution` with the price.
"""
function solve(
    prob::PricingProblem{VanillaOption{European, A, B}, BlackScholesInputs},
    ::BlackScholesAnalytic
) where {A, B}

    K = prob.payoff.strike
    r = prob.market.rate
    σ = prob.market.sigma
    cp = prob.payoff.call_put()
    T = yearfrac(prob.market.referenceDate, prob.payoff.expiry)
    F = prob.market.spot * exp(r * T)

    price = if σ == 0
        exp(-r * T) * prob.payoff(prob.market.spot * exp(r * T))
    else
        d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrt(T))
        d2 = d1 - σ * sqrt(T)
        exp(-r * T) * cp * (F * cdf(Normal(), cp * d1) - K * cdf(Normal(), cp * d2))
    end

    return AnalyticSolution(price)
end

# Define the calibration problem
struct BlackScholesCalibrationProblem{P<:PricingProblem, M}
    prob::P
    method::M
    price_target::Float64
end

function solve(calib::BlackScholesCalibrationProblem; initial_guess=0.2, lower=1e-6, upper=5.0, kwargs...)
    function calibration_function(σ, p)
        market = calib.prob.market
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
    call_put_matrix::Union{Nothing, AbstractMatrix} = nothing,
    interp_strike = LinearInterpolation,
    interp_time = LinearInterpolation,
    extrap_strike = ExtrapolationType.Constant,
    extrap_time = ExtrapolationType.Constant,
    kwargs...
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

    for i in 1:nrows, j in 1:ncols
        expiry = reference_date + tenors[i]
        strike = strikes[j]
        cp     = call_put_matrix[i, j]
        price  = prices[i, j]

        payoff = VanillaOption(strike, expiry, European(), cp, Spot())
        market = BlackScholesInputs(reference_date, rate, spot, 0.2)

        prob = BlackScholesCalibrationProblem(
            PricingProblem(payoff, market),
            BlackScholesAnalytic(),
            price
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
        interp_time   = interp_time,
        extrap_strike = extrap_strike,
        extrap_time   = extrap_time,
    )
end
