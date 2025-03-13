export VanillaOption, AbstractPayoff

abstract type AbstractExerciseStyle end

struct European <: AbstractExerciseStyle end
struct American <: AbstractExerciseStyle end

"""An abstract type representing a financial payoff, such as a vanilla European option, an Asian option, or a forward."""
abstract type AbstractPayoff end

"""A vanilla European call option payoff.

# Fields
- `strike`: The strike price of the option.
- `expiry`: The time to maturity of the option.

This struct represents a European call option, which provides a payoff of `max(spot - strike, 0.0)`.
"""
struct VanillaOption{E,C} <: AbstractPayoff where E<:AbstractExerciseStyle
    strike
    expiry
    exercise_style::E
    call_put::C
end

abstract type AbstractCallPut end
struct Put <: AbstractCallPut end
struct Call <: AbstractCallPut end

(call_put::Call)() = 1.0
(call_put::Put)() = -1.0

"""Computes the payoff of a vanilla European call option given a spot price.

# Arguments
- `payoff::VanillaEuropeanCall`: The call option payoff structure.
- `spot`: The current spot price of the underlying asset.

# Returns
- The payoff value, calculated as `max(spot - payoff.strike, 0.0)`.
"""
function (payoff::VanillaOption)(spot)
    return max.(payoff.call_put() .* (spot .- payoff.strike), 0.0)
end
