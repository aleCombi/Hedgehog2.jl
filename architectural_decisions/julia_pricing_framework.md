We have a modular derivatives pricing framework in Julia, organized as follows:

1. An `AbstractPayoff` type with concrete payoffs like `VanillaEuropeanCall`, which is callable on a spot price.

2. `AbstractMarketInputs` to represent market data. We have a `BlackScholesInputs` concrete type storing rate, spot, and sigma. We can extend with more complex market inputs, e.g., local volatility parameters, yield curves, or multiple underlying assets.

3. An `AbstractPricingStrategy` type for the pricing model. We currently have a `BlackScholesStrategy` with an analytical formula for vanilla European options. We can add Monte Carlo or PDE-based methods by subtyping `AbstractPricingStrategy`.

4. A `Pricer` struct that ties (payoff, market inputs, strategy) together into a callable. It uses the logic within its `pricingStrategy` to return a price.

5. A `DeltaCalculator` struct that references a `Pricer` and a `deltaMethod` (subtype of `AbstractDeltaMethod`). We have two methods: 
   - `BlackScholesAnalyticalDelta` for the closed-form expression.
   - `ADDelta` using automatic differentiation via ForwardDiff.

New pricing should always be done by using a Pricer, with appropriate field types. If a pricer work on many different payoffs, the payoff type could be a parameter itself. 
They should always be created defining a callable Pricer method.