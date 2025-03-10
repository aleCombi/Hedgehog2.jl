module Hedgehog2

using BenchmarkTools, ForwardDiff, Distributions, Accessors

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

"""market data inputs for pricers"""
abstract type AbstractMarketInputs end

"""Inputs for black scholes model"""
struct BlackScholesInputs <: AbstractMarketInputs
    rate
    spot
    sigma
end

"""A pricing method"""
abstract type AbstractPricingMethod end

"""# the whole algorithm to price a specific derivative, using specific market inputs and a pricing method.
# it should be a callable made up of all the ingredients: a payoff, market data, a pricing model"""
struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S<:AbstractPricingMethod}
    marketInputs::M
    payoff::P
    pricingMethod::S
end

"""Black scholes method"""
struct BlackScholesMethod <: AbstractPricingMethod end

"""Dispatch of pricer for call black scholes pricing"""
function (pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod})() 
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end

"""A method for delta calculation"""
abstract type AbstractDeltaMethod end

"""A method for delta calculation analytically using black scholes"""
struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end

"""Delta calculator"""
struct DeltaCalculator{D<:AbstractDeltaMethod, P<:AbstractPayoff, I<:AbstractMarketInputs, S<:AbstractPricingMethod}
    pricer::Pricer{P, I, S}
    deltaMethod::D
end

"""Callable struct: Computes delta when called, using black scholes on a call."""
function (delta_calc::DeltaCalculator{BlackScholesAnalyticalDelta, VanillaEuropeanCall, BlackScholesInputs, BlackScholesMethod})()
    S = delta_calc.pricer.marketInputs.spot
    K = delta_calc.pricer.payoff.strike
    r = delta_calc.pricer.marketInputs.rate
    σ = delta_calc.pricer.marketInputs.sigma
    T = delta_calc.pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)  # Black-Scholes delta for calls
end


"""Delta with AD"""
struct ADDelta <: AbstractDeltaMethod end

"""Delta with AD callable"""
function (delta_calc::DeltaCalculator{ADDelta, P, BlackScholesInputs, S})() where {P,S}
    pricer = delta_calc.pricer
    return ForwardDiff.derivative(
        S -> begin
            new_pricer = @set pricer.marketInputs.spot = S
            new_pricer()
        end,
        pricer.marketInputs.spot
    )
end

"""Example code with benchmarks"""
function example()
    # Define market data and payoff
    market_inputs = BlackScholesInputs(0.01, 1, 0.4)
    payoff = VanillaEuropeanCall(1, 1)
    pricer = Pricer(market_inputs, payoff, BlackScholesMethod())
    println(pricer())
    # Analytical Delta
    analytical_delta_calc = DeltaCalculator(pricer, BlackScholesAnalyticalDelta())
    println(analytical_delta_calc())

    # AD Delta
    ad_delta_calc = DeltaCalculator(pricer, ADDelta())
    println(ad_delta_calc())

    println("Benchmarking pricer:")
    @btime pricer()

    # Run benchmarks
    println("Benchmarking Analytical Delta:")
    @btime analytical_delta_calc()

    println("Benchmarking AD Delta:")
    @btime ad_delta_calc()
end

example() 

end