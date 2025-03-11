"""A payoff, such as a vanilla european option or an asian option or a forward."""
abstract type AbstractPayoff end

"""vanilla european call payoff"""
struct VanillaEuropeanCall <: AbstractPayoff
    strike
    time
end

"""vanilla european option callable to get the payoff given a spot price."""
function (payoff::VanillaEuropeanCall)(spot)
    return max(spot - payoff.strike, 0.0)
end