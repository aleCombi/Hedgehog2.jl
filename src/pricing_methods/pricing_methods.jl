"""
An abstract type representing a pricing method.

All pricing methods should inherit from this type.
"""
abstract type AbstractPricingMethod end

"""
A `PricingProblem` bundles the payoff and the market inputs required to price a derivative.

# Type Parameters
- `P<:AbstractPayoff`: The payoff type (e.g., VanillaOption).
- `M<:AbstractMarketInputs`: The market data type (e.g., volatility surface, interest rate curve).

# Fields
- `payoff::P`: The payoff object describing the contract to be priced.
- `market::M`: The market data needed for pricing.
"""
struct PricingProblem{P<:AbstractPayoff,M<:AbstractMarketInputs}
    payoff::P
    market_inputs::M
end

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
    new_curve = RateCurve(curve.reference_date, new_itp, curve.builder)
    return @set prob.market_inputs.rate = new_curve
end

# Flat curve: just return new FlatRateCurve with new value
function _set_rate_curve(prob, lens::ZeroRateSpineLens, new_zᵢ, curve::FlatRateCurve)
    new_curve = FlatRateCurve(curve.reference_date, new_zᵢ)
    return @set prob.market_inputs.rate = new_curve
end

spine_zeros(curve::FlatRateCurve) = [curve.rate]
spine_tenors(curve::FlatRateCurve) = [0.0]  # Or however you define the time axis
