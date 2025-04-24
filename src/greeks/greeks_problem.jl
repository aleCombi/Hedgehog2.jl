"""
    GreekLens

Abstract supertype for all lens types used to extract or modify model inputs
(e.g., spot, volatilities, rates) for sensitivity analysis or automatic differentiation.

Concrete subtypes define how to access and mutate a particular model input from a `PricingProblem`.
"""
abstract type GreekLens end

"""
    SpotLens <: GreekLens

Lens for accessing and modifying the spot price within a `PricingProblem` object.

Used to compute sensitivities (e.g., delta) with respect to the spot.
"""
struct SpotLens <: GreekLens end

"""
    (::SpotLens)(p::PricingProblem)

Extract the spot value from the `market_inputs` field of the pricing problem `p`.

# Arguments
- `p`: A `PricingProblem` that contains `market_inputs.spot`.

# Returns
- The current spot value.
"""
function (::SpotLens)(p)
    return p.market_inputs.spot
end

"""
    set(p::PricingProblem, ::SpotLens, newval)

Return a modified copy of the pricing problem `p` with its spot value updated to `newval`.

# Arguments
- `p`: A `PricingProblem` with a `market_inputs` field containing `spot`.
- `newval`: The new value to assign to `spot`.

# Returns
- A new `PricingProblem` instance with updated spot value.
"""
function set(p, ::SpotLens, newval)
    return @set p.market_inputs.spot = newval
end

"""
    VolLens(strike, expiry)

Lens structure for accessing and mutating specific vol entries by strike and expiry.
"""
struct VolLens{S, T} <: GreekLens
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


# Method types
"""
    GreekMethod

Abstract type representing different methods for Greek calculation.
"""
abstract type GreekMethod end

"""
    GreekResult{T}

Container for the result of a Greek calculation.

# Fields
- `greek`: The calculated Greek value.
"""
struct GreekResult{T}
    greek::T
end

"""
    FDScheme

Abstract type representing different finite difference schemes.
"""
abstract type FDScheme end

"""
    FDForward <: FDScheme

Finite difference scheme using forward differencing.
"""
struct FDForward <: FDScheme end

"""
    FDBackward <: FDScheme

Finite difference scheme using backward differencing.
"""
struct FDBackward <: FDScheme end

"""
    FDCentral <: FDScheme

Finite difference scheme using central differencing.
"""
struct FDCentral <: FDScheme end

"""
    AnalyticGreek <: GreekMethod

Greek calculation method using closed-form analytic formulas.
"""
struct AnalyticGreek <: GreekMethod end

"""
    ForwardAD <: GreekMethod

Greek calculation method using forward-mode automatic differentiation.
"""
struct ForwardAD <: GreekMethod end

"""
    FiniteDifference{S<:FDScheme, A <: Number} <: GreekMethod

Greek calculation method using finite differences.

# Fields
- `bump`: The bump size for finite difference.
- `scheme`: The finite difference scheme to use.
"""
struct FiniteDifference{S<:FDScheme, A <: Number} <: GreekMethod
    bump::A
    scheme::S
end

"""
    FiniteDifference(bump)

Convenience constructor for central finite differencing with a specified bump size.

# Arguments
- `bump`: The bump size for finite difference.

# Returns
- A `FiniteDifference` object with a central scheme.
"""
FiniteDifference(bump) = FiniteDifference(bump, FDCentral())

"""
    GreekProblem{P,L}

Problem definition for calculating a first-order Greek.

# Fields
- `pricing_problem`: The underlying pricing problem.
- `wrt`: The lens specifying which parameter to differentiate with respect to.
"""
struct GreekProblem{P,L}
    pricing_problem::P
    wrt::L  # accessor (from Accessors.jl)
end

"""
    solve(gprob::GreekProblem, ::ForwardAD, pricing_method::P) where {P<:AbstractPricingMethod}

Solve a first-order Greek problem using automatic differentiation.

# Arguments
- `gprob`: The Greek problem to solve.
- `::ForwardAD`: The method to use (automatic differentiation).
- `pricing_method`: The method to use for pricing.

# Returns
- A named tuple containing the calculated Greek.
"""
function solve(
    gprob::GreekProblem,
    ::ForwardAD,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens = gprob.wrt

    x₀ = lens(prob)
    f = x -> solve(set(prob, lens, x), pricing_method).price

    deriv = ForwardDiff.derivative(f, x₀)
    return (greek = deriv,)
end

"""
    compute_fd_derivative(scheme::FDScheme, prob, lens, ε, pricing_method)

Calculate numerical derivative using finite difference schemes.

# Arguments
- `scheme`: The finite difference scheme to use (Forward, Backward, or Central).
- `prob`: The pricing problem.
- `lens`: The lens to access and modify the relevant parameter.
- `ε`: The bump size for finite difference.
- `pricing_method`: The method to price the derivative.

# Returns
- The calculated finite difference derivative.
"""
function compute_fd_derivative(::FDForward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ * (1 + ε))

    v_up = solve(prob_up, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v_up - v₀) / (x₀ * ε)
end

function compute_fd_derivative(::FDBackward, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_down = set(prob, lens, x₀ * (1 - ε))
    v_down = solve(prob_down, pricing_method).price
    v₀ = solve(prob, pricing_method).price
    return (v₀ - v_down) / (x₀ * ε)
end

function compute_fd_derivative(::FDCentral, prob, lens, ε, pricing_method)
    x₀ = lens(prob)
    prob_up = set(prob, lens, x₀ * (1 + ε))
    prob_down = set(prob, lens, x₀ * (1 - ε))
    v_up = solve(prob_up, pricing_method).price
    v_down = solve(prob_down, pricing_method).price
    return (v_up - v_down) / (2ε * x₀)
end

"""
    solve(gprob::GreekProblem, method::FiniteDifference{S,A}, pricing_method::P) where {S<:FDScheme,P<:AbstractPricingMethod,A}

Solve a first-order Greek problem using finite differences.

# Arguments
- `gprob`: The Greek problem to solve.
- `method`: Finite difference method configuration.
- `pricing_method`: The method to use for pricing.

# Returns
- A `GreekResult` containing the calculated Greek.
"""
function solve(
    gprob::GreekProblem,
    method::FiniteDifference{S,A},
    pricing_method::P,
) where {S<:FDScheme,P<:AbstractPricingMethod,A}
    prob = gprob.pricing_problem
    lens = gprob.wrt
    ε = method.bump
    scheme = method.scheme
    deriv = compute_fd_derivative(scheme, prob, lens, ε, pricing_method)
    return GreekResult(deriv)
end

"""
    SecondOrderGreekProblem{P,L1,L2}

Problem definition for calculating a second-order Greek.

# Fields
- `pricing_problem`: The underlying pricing problem.
- `wrt1`: The first lens specifying the first parameter to differentiate with respect to.
- `wrt2`: The second lens specifying the second parameter to differentiate with respect to.
"""
struct SecondOrderGreekProblem{P,L1,L2}
    pricing_problem::P
    wrt1::L1
    wrt2::L2
end

"""
    solve(gprob::SecondOrderGreekProblem, ::ForwardAD, pricing_method::P) where {P<:AbstractPricingMethod}

Solve a second-order Greek problem using automatic differentiation.

# Arguments
- `gprob`: The second-order Greek problem to solve.
- `::ForwardAD`: The method to use (automatic differentiation).
- `pricing_method`: The method to use for pricing.

# Returns
- A `GreekResult` containing the calculated second-order Greek.
"""
function solve(
    gprob::SecondOrderGreekProblem,
    ::ForwardAD,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens1, lens2 = gprob.wrt1, gprob.wrt2
    x₀, y₀ = lens1(prob), lens2(prob)

    f = (x, y) -> solve(set(set(prob, lens1, x), lens2, y), pricing_method).price

    if lens1 === lens2
        ∂f = x -> ForwardDiff.derivative(z -> f(x, z), x)
        deriv = ForwardDiff.derivative(∂f, x₀)
    else
        ∂f = x -> ForwardDiff.derivative(y -> f(x, y), y₀)
        deriv = ForwardDiff.derivative(∂f, x₀)
    end

    return GreekResult(deriv)
end

"""
    solve(gprob::SecondOrderGreekProblem, method::FiniteDifference, pricing_method::P) where {P<:AbstractPricingMethod}

Solve a second-order Greek problem using finite differences.

# Arguments
- `gprob`: The second-order Greek problem to solve.
- `method`: Finite difference method configuration.
- `pricing_method`: The method to use for pricing.

# Returns
- A `GreekResult` containing the calculated second-order Greek.
"""
function solve(
    gprob::SecondOrderGreekProblem,
    method::FiniteDifference,
    pricing_method::P,
) where {P<:AbstractPricingMethod}
    prob = gprob.pricing_problem
    lens1, lens2 = gprob.wrt1, gprob.wrt2
    ε = method.bump

    x₀, y₀ = lens1(prob), lens2(prob)

    f = (x, y) -> solve(set(set(prob, lens1, x), lens2, y), pricing_method).price

    if lens1 === lens2
        f_plus = f(x₀ + ε, y₀ + ε)
        f_0 = f(x₀, y₀)
        f_minus = f(x₀ - ε, y₀ - ε)
        deriv = (f_plus - 2f_0 + f_minus) / (ε^2)
    else
        f_pp = f(x₀ + ε, y₀ + ε)
        f_pm = f(x₀ + ε, y₀ - ε)
        f_mp = f(x₀ - ε, y₀ + ε)
        f_mm = f(x₀ - ε, y₀ - ε)
        deriv = (f_pp - f_pm - f_mp + f_mm) / (4ε^2)
    end

    return GreekResult(deriv)
end

"""
    solve(gprob::GreekProblem{PricingProblem{VanillaOption{TS,TE,European,B,C},I}, L}, ::AnalyticGreek, ::BlackScholesAnalytic) where {TS,TE,B,C,L, I<:BlackScholesInputs}

Solve a first-order Greek problem for European vanilla options using closed-form Black-Scholes formulas.

# Arguments
- `gprob`: The Greek problem to solve.
- `::AnalyticGreek`: The method to use (analytic formulas).
- `::BlackScholesAnalytic`: The pricing method (Black-Scholes analytic).

# Returns
- A `GreekResult` containing the calculated Greek.
"""
function solve(
    gprob::GreekProblem{
        PricingProblem{VanillaOption{TS,TE,European,B,C},I},
        L,
    },
    ::AnalyticGreek,
    ::BlackScholesAnalytic,
) where {TS,TE,B,C,L, I<:BlackScholesInputs}
    prob = gprob.pricing_problem
    lens = gprob.wrt
    inputs = prob.market_inputs
    cp = prob.payoff.call_put()

    S = inputs.spot
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike
    σ = get_vol_yf(inputs.sigma, T, K)

    D = df(prob.market_inputs.rate, prob.payoff.expiry)
    F = prob.market_inputs.spot / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    Φ = x -> cdf(Normal(), x)
    ϕ = x -> pdf(Normal(), x)

    greek = if lens === @optic _.market_inputs.spot
        # Delta = ∂V/∂S = ∂V/∂F * ∂F/∂S = (cp * N(cp·d1)) * (1/D)
        cp * Φ(cp * d1)

    elseif lens === VolLens(1,1) #TODO: add logic
        # Vega = ∂V/∂σ = D · F · φ(d1) · √T
        D * F * ϕ(d1) * √T

    elseif lens === @optic _.payoff.expiry
        # Assume flat rate: z(T) = r ⇒ D(T) = exp(-rT), F(T) = S / D(T)
        r = zero_rate_yf(prob.market_inputs.rate, T)
        (r * prob.payoff.strike * D * Φ(d2) + F * D * σ * ϕ(d1) / (2√T)) / (MILLISECONDS_IN_YEAR_365) #against ticks, to match AD and FD. Observe that the sign is counterintuitive as it is a derivative against expiry in tticks, not against time-to-maturity in yearfrac

    else
        error("Unsupported lens for analytic Greek")
    end

    return GreekResult(greek)
end

"""
    solve(gprob::SecondOrderGreekProblem, ::AnalyticGreek, ::BlackScholesAnalytic)

Solve a second-order Greek problem using closed-form Black-Scholes formulas.

# Arguments
- `gprob`: The second-order Greek problem to solve.
- `::AnalyticGreek`: The method to use (analytic formulas).
- `::BlackScholesAnalytic`: The pricing method (Black-Scholes analytic).

# Returns
- A `GreekResult` containing the calculated second-order Greek.
"""
function solve(gprob::SecondOrderGreekProblem, ::AnalyticGreek, ::BlackScholesAnalytic)
    prob = gprob.pricing_problem
    lens1 = gprob.wrt1
    lens2 = gprob.wrt2
    inputs = prob.market_inputs

    S = inputs.spot
    T = yearfrac(inputs.referenceDate, prob.payoff.expiry)
    K = prob.payoff.strike
    σ = get_vol_yf(inputs.sigma, T, K)

    D = df(inputs.rate, prob.payoff.expiry)
    F = S / D
    √T = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * √T)
    d2 = d1 - σ * √T

    ϕ = x -> pdf(Normal(), x)

    greek = if (lens1 === @optic _.market_inputs.spot) && (lens2 === @optic _.market_inputs.spot)
        # Gamma = ∂²V/∂S² = φ(d1) / (Sσ√T)
        ϕ(d1) / (S * σ * √T)

    elseif (lens1 === VolLens(1,1)) && (lens2 === VolLens(1,1)) #TODO: introduce logic for sigma
        # Volga = Vega * d1 * d2 / σ
        vega = D * F * ϕ(d1) * √T
        vega * d1 * d2 / σ

    else
        error("Unsupported second-order analytic Greek")
    end

    return GreekResult(greek)
end

"""
    BatchGreekProblem{P,L}

Problem definition for calculating multiple Greeks at once.

# Fields
- `pricing_problem`: The underlying pricing problem.
- `lenses`: A collection of lenses, each specifying a parameter to differentiate with respect to.
"""
struct BatchGreekProblem{P,L}
    pricing_problem::P
    lenses::L
end

"""
    solve(gprob::BatchGreekProblem{P,L}, method::GreekMethod, pricing_method::AbstractPricingMethod) where {P,L}

Solve multiple Greek problems simultaneously.

# Arguments
- `gprob`: The batch Greek problem to solve.
- `method`: The method to use for Greek calculation.
- `pricing_method`: The method to use for pricing.

# Returns
- A dictionary mapping lenses to their corresponding Greeks.
"""
function solve(
    gprob::BatchGreekProblem{P,L},
    method::GreekMethod,
    pricing_method::AbstracstPricingMethod
) where {P,L}
    lenses = gprob.lenses
    prob = gprob.pricing_problem
   
    Dict(lens => solve(GreekProblem(prob, lens), method, pricing_method).greek for lens in lenses)
end