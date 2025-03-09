module Hedgehog2

# A payoff, such as a vanilla european option or an asian option or a forward.
abstract type AbstractPayoff end

struct VanillaEuropeanCall <: AbstractPayoff
    strike
    time
end

# vanilla european option callable to get the payoff given a spot price.
(payoff::VanillaEuropeanCall)(spot) = max(spot - payoff.strike, 0.0)

# market data inputs for pricers
abstract type AbstractMarketInputs end

struct BlackScholesInputs <: AbstractMarketInputs
    rate
    spot
    sigma
end


# # MODELS
# the whole algorithm to price a specific derivative.
# it should be a callable made up of all the ingredients: a payoff, market data, a pricing model
abstract type AbstractPricingStrategy end

using Distributions

struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S<:AbstractPricingStrategy}
    marketInputs::M
    payoff::P
    pricingStrategy::S
end


struct BlackScholesStrategy <: AbstractPricingStrategy end

function (pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesStrategy})() 
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end


abstract type AbstractDeltaMethod end
struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end
struct DeltaCalculator{D<:AbstractDeltaMethod, P<:AbstractPayoff, I<:AbstractMarketInputs, S<:AbstractPricingStrategy}
    pricer::Pricer{P, I, S}
    deltaMethod::D
end

# Callable struct: Computes delta when called
function (delta_calc::DeltaCalculator{BlackScholesAnalyticalDelta, VanillaEuropeanCall, BlackScholesInputs, BlackScholesStrategy})()
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)  # Black-Scholes delta for calls
end

using ForwardDiff

struct ADDelta <: AbstractDeltaMethod end

using Accessors, ForwardDiff

function (delta_calc::DeltaCalculator{ADDelta, P, BlackScholesInputs, S})() where {P,S}
    return ForwardDiff.derivative(
        S -> begin
            new_pricer = @set pricer.marketInputs.spot = S
            new_pricer()
        end,
        pricer.marketInputs.spot
    )
end


using Accessors

using BenchmarkTools, ForwardDiff, Distributions

# Define market data and payoff
market_inputs = BlackScholesInputs(0.01, 1, 0.4)
payoff = VanillaEuropeanCall(1, 1)
pricer = Pricer(market_inputs, payoff, BlackScholesStrategy())
println(pricer())
# Analytical Delta
analytical_delta_calc = DeltaCalculator(pricer, BlackScholesAnalyticalDelta())
println(analytical_delta_calc())

# AD Delta
ad_delta_calc = DeltaCalculator(pricer, ADDelta())
println(ad_delta_calc())

# Run benchmarks
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()


end
