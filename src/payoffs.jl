"""An abstract type representing a financial payoff, such as a vanilla European option, an Asian option, or a forward."""
abstract type AbstractPayoff end

"""A vanilla European call option payoff.

# Fields
- `strike`: The strike price of the option.
- `time`: The time to maturity of the option.

This struct represents a European call option, which provides a payoff of `max(spot - strike, 0.0)`.
"""
struct VanillaEuropeanCall <: AbstractPayoff
    strike
    time
end

"""Computes the payoff of a vanilla European call option given a spot price.

# Arguments
- `payoff::VanillaEuropeanCall`: The call option payoff structure.
- `spot`: The current spot price of the underlying asset.

# Returns
- The payoff value, calculated as `max(spot - payoff.strike, 0.0)`.
"""
function (payoff::VanillaEuropeanCall)(spot)
    return max(spot - payoff.strike, 0.0)
end
