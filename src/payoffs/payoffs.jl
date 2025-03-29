export VanillaOption, AbstractPayoff, European, Spot, Forward, Call, Put

"""
    AbstractExerciseStyle

Abstract type representing the exercise style of an option (e.g., European or American).
"""
abstract type AbstractExerciseStyle end

"""
    European <: AbstractExerciseStyle

Represents a European-style option that can only be exercised at expiry.
"""
struct European <: AbstractExerciseStyle end

"""
    American <: AbstractExerciseStyle

Represents an American-style option that can be exercised at any time before or at expiry.
"""
struct American <: AbstractExerciseStyle end

"""
    AbstractPayoff

An abstract type representing a financial payoff, such as a vanilla European option, an Asian option, or a forward.
"""
abstract type AbstractPayoff end

"""
    Underlying

Abstract type representing the nature of the underlying asset.
"""
abstract type Underlying end

"""
    Spot <: Underlying

Represents an underlying defined in terms of spot price.
"""
struct Spot <: Underlying end

"""
    Forward <: Underlying

Represents an underlying defined in terms of forward price.
"""
struct Forward <: Underlying end

"""
    AbstractCallPut

Abstract type representing whether an option is a call or a put.
"""
abstract type AbstractCallPut end

"""
    Put <: AbstractCallPut

Represents a put option.
"""
struct Put <: AbstractCallPut end

"""
    Call <: AbstractCallPut

Represents a call option.
"""
struct Call <: AbstractCallPut end

"""
    (call_put::Call)() -> 1.0

Returns the call-put indicator for a call option (+1).
"""
function (call_put::Call)() 
    return 1.0
end

"""
    (call_put::Put)() -> -1.0

Returns the call-put indicator for a put option (-1).
"""
function (call_put::Put)() 
    return -1.0
end

"""
    VanillaOption{E,C,U} <: AbstractPayoff

A vanilla option with specified exercise style, call/put type, and underlying type.

# Fields
- `strike`: The strike price of the option.
- `expiry`: The expiry (maturity) time.
- `exercise_style`: Instance of `European` or `American`.
- `call_put`: Instance of `Call` or `Put`.
- `underlying`: Either `Spot` or `Forward`.
"""
struct VanillaOption{E,C,U} <: AbstractPayoff where {E<:AbstractExerciseStyle, C <: AbstractCallPut, U<: Underlying}
    strike
    expiry::Real
    exercise_style::E
    call_put::C
    underlying::U
end

function VanillaOption(
    strike,
    expiry_date::TimeType,
    exercise_style::E,
    call_put::C,
    underlying::U
) where {E<:AbstractExerciseStyle, C<:AbstractCallPut, U<:Underlying}
    expiry_ticks = to_ticks(expiry_date)
    return VanillaOption{E, C, U}(strike, expiry_ticks, exercise_style, call_put, underlying)
end

"""
    (payoff::VanillaOption)(spot) -> Float64

Computes the payoff of a vanilla option for a given spot price.

# Arguments
- `payoff`: A `VanillaOption` instance.
- `spot`: The spot price or array of spot prices.

# Returns
- The intrinsic value(s), i.e. `max(cp * (S - K), 0)`.
"""
function (payoff::VanillaOption)(spot)
    return max.(payoff.call_put() .* (spot .- payoff.strike), 0.0)
end

"""
    parity_transform(call_price, opt::VanillaOption{E, Call, U}, S, T) -> Float64

Returns the call price unchanged (no transformation needed).

# Arguments
- `call_price`: Price of the call option.
- `opt`: Vanilla call option.
- `S`: Spot price.
- `T`: Time to expiry.
"""
function parity_transform(call_price, opt::VanillaOption{E, Call, U}, S, T) where {E, U}
    return call_price
end

"""
    parity_transform(call_price, opt::VanillaOption{E, Put, U}, S, T) -> Float64

Applies put-call parity to derive the put price from the call price.

# Arguments
- `call_price`: Price of the call option.
- `opt`: Vanilla put option.
- `S`: Spot price.
- `T`: Time to expiry.

# Returns
- The corresponding put price using: `put = call - S + K * exp(-rT)`.
"""
function parity_transform(call_price, opt::VanillaOption{E, Put, U}, S, T) where {E, U}
    return call_price - S + opt.strike * exp(-opt.expiry)
end