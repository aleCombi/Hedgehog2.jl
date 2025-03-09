Hedgehog2 Design Overview

This document provides a concise architectural overview of the Hedgehog2 derivatives pricing library. It explains the core abstractions, how they fit together, and the recommended approach for extending the library.

--------------------------------------------------------------------------------
1. Payoffs

   AbstractPayoff
   --------------
   An abstract type representing the concept of an option or other payoff. It
   defines what is being priced, mathematically:

       abstract type AbstractPayoff end

   Concrete Payoff Example: VanillaEuropeanCall
   --------------------------------------------
   A plain vanilla European call option. It stores:
     - strike
     - time (time to maturity)

       struct VanillaEuropeanCall <: AbstractPayoff
           strike
           time
       end

       (payoff::VanillaEuropeanCall)(spot) = max(spot - payoff.strike, 0.0)

   Implementation Note:
   Each payoff type should be callable, computing the payoff given the spot
   price (or other underlying state). This makes it easy to plug into numerical
   methods such as automatic differentiation.

--------------------------------------------------------------------------------
2. Market Inputs

   AbstractMarketInputs
   --------------------
   An abstract type representing the market data required for pricing:

       abstract type AbstractMarketInputs end

   Concrete Market Data Example: BlackScholesInputs
   ------------------------------------------------
       struct BlackScholesInputs <: AbstractMarketInputs
           rate
           spot
           sigma
       end

   Implementation Note:
   New pricing models might need extra inputs (e.g., yield curves, local vol
   parameters). Extend AbstractMarketInputs accordingly.

--------------------------------------------------------------------------------
3. Pricing Strategies and the Pricer Structure

   AbstractPricingStrategy
   -----------------------
   An abstract type that defines the method of computing the price for a given
   payoff under certain market inputs:

       abstract type AbstractPricingStrategy end

   Pricer
   ------
   The core container that combines:
     - A concrete AbstractPayoff
     - A concrete AbstractMarketInputs
     - A concrete AbstractPricingStrategy

       struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S<:AbstractPricingStrategy}
           marketInputs::M
           payoff::P
           pricingStrategy::S
       end

   This design avoids the “God object” anti-pattern. Instead, each model-specific
   pricing routine is placed in a Pricer, specialized for the payoff, market data
   type, and strategy.

   Example Strategy: BlackScholesStrategy
   --------------------------------------
   Implements the closed-form Black-Scholes formula for a European call. We
   define a callable method on the Pricer itself:

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

   Implementation Note:
   When adding a new pricing algorithm (e.g., a Monte Carlo method or Binomial
   Tree), create a new type subtyping AbstractPricingStrategy and define an
   appropriate callable method on Pricer{PayoffType, MarketType, NewStrategyType}.

--------------------------------------------------------------------------------
4. Greeks Calculation

   AbstractDeltaMethod
   -------------------
   An abstract type for defining methods to calculate the option’s delta or
   other Greeks:

       abstract type AbstractDeltaMethod end

   DeltaCalculator
   ---------------
   A container that references:
     - A Pricer
     - A chosen AbstractDeltaMethod

   The calculator is then called to compute the delta.

       struct DeltaCalculator{DM <: AbstractDeltaMethod, P <: AbstractPayoff, M <: AbstractMarketInputs, S <: AbstractPricingStrategy}
           pricer::Pricer{P, M, S}
           deltaMethod::DM
       end

   Example: Black-Scholes Analytical Delta
   ---------------------------------------
   Uses the Black-Scholes closed-form delta formula:

       struct BlackScholesAnalyticalDelta <: AbstractDeltaMethod end

       function (delta_calc::DeltaCalculator{BlackScholesAnalyticalDelta, VanillaEuropeanCall, BlackScholesInputs, BlackScholesStrategy})()
           pricer = delta_calc.pricer
           S = pricer.marketInputs.spot
           K = pricer.payoff.strike
           r = pricer.marketInputs.rate
           σ = pricer.marketInputs.sigma
           T = pricer.payoff.time
           d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
           return cdf(Normal(), d1)
       end

   Example: Automatic Differentiation (AD) Delta
   ---------------------------------------------
   Uses ForwardDiff to compute the derivative of the pricing function with
   respect to spot:

       struct ADDelta <: AbstractDeltaMethod end

       function (delta_calc::DeltaCalculator{ADDelta, P, BlackScholesInputs, S})() where {P, S}
           pricer = delta_calc.pricer
           return ForwardDiff.derivative(
               S -> begin
                   new_pricer = @set pricer.marketInputs.spot = S
                   new_pricer()
               end,
               pricer.marketInputs.spot
           )
       end

   Implementation Note:
   Additional Greeks (Gamma, Vega, Rho, etc.) follow the same pattern. Simply
   define the appropriate abstract method type or reuse AbstractDeltaMethod if
   closely related, and implement the callable method on DeltaCalculator or a
   specialized GreekCalculator.

--------------------------------------------------------------------------------
5. Example Usage

   using BenchmarkTools, ForwardDiff, Distributions

   # 1. Define the market data and payoff
   market_inputs = BlackScholesInputs(0.01, 1, 0.4)
   payoff = VanillaEuropeanCall(1, 1)

   # 2. Create a Pricer
   pricer = Pricer(market_inputs, payoff, BlackScholesStrategy())

   # 3. Call the Pricer to get the option price
   println("Option price: ", pricer())

   # 4. Analytical Delta
   analytical_delta_calc = DeltaCalculator(pricer, BlackScholesAnalyticalDelta())
   println("Analytical Delta: ", analytical_delta_calc())

   # 5. Automatic Differentiation Delta
   ad_delta_calc = DeltaCalculator(pricer, ADDelta())
   println("AD Delta: ", ad_delta_calc())

   # 6. Benchmark the delta methods
   println("Benchmarking Analytical Delta:")
   @btime analytical_delta_calc()

   println("Benchmarking AD Delta:")
   @btime ad_delta_calc()

--------------------------------------------------------------------------------
6. Extending the Library

   1. New Payoff
      ----------
      - Subtype AbstractPayoff.
      - Store relevant parameters (strike, barrier level, etc.).
      - Define its call operator to compute the payoff.

   2. New Market Inputs
      -----------------
      - Subtype AbstractMarketInputs.
      - Store extra market parameters (yield curves, local volatility surfaces).

   3. New Pricing Strategy
      --------------------
      - Subtype AbstractPricingStrategy.
      - Overload the callable method for Pricer{YourPayoff, YourMarketInputs, YourStrategy}.

   4. New Greeks Calculation
      ----------------------
      - Subtype AbstractDeltaMethod (or a more general AbstractGreekMethod).
      - Implement a callable method on DeltaCalculator{NewMethod, ...}.

--------------------------------------------------------------------------------

Conclusion
----------
Hedgehog2 organizes option payoffs, market data, and pricing methods into simple,
composable abstractions. Each component is clearly separated, making it
straightforward to add new instruments, new models, or new numerical techniques
without disrupting existing code. When extending Hedgehog2, follow the patterns
above and implement well-scoped methods on the relevant abstract types.
