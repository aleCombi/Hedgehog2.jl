# Greeks Calculation

Hedgehog implements various methods for computing sensitivities (Greeks) of derivative prices, using a unified [`solve`](@ref) interface with the `GreekProblem` type.

---

## Greek Methods

### [`ForwardAD`](@ref)

The [`ForwardAD`](@ref) method uses forward-mode automatic differentiation to compute sensitivities. It typically provides precise results with good performance, especially for first-order Greeks.

```julia
# Calculate delta using automatic differentiation
spot_lens = @optic _.market_inputs.spot
delta_problem = GreekProblem(pricing_problem, spot_lens)
solution = solve(delta_problem, ForwardAD(), BlackScholesAnalytic())
delta = solution.greek
```

---

### [`FiniteDifference`](@ref)

The [`FiniteDifference`](@ref) method approximates derivatives using numerical differencing. It provides flexibility in choice of scheme (forward, backward, central) and step size.

```julia
# Calculate delta using central finite differences
fd_method = FiniteDifference(1e-4)  # 1e-4 is the step size
solution = solve(delta_problem, fd_method, BlackScholesAnalytic())
delta = solution.greek
```

---

### [`AnalyticGreek`](@ref)

The [`AnalyticGreek`](@ref) method uses closed-form solutions for sensitivities when available. Currently supports Black-Scholes model Greeks (delta, gamma, vega, theta, rho).

```julia
# Calculate delta using analytical formulas
solution = solve(delta_problem, AnalyticGreek(), BlackScholesAnalytic())
delta = solution.greek
```

---

## Problem Types

### [`GreekProblem`](@ref)

A `GreekProblem` represents a first-order sensitivity calculation with respect to a specific parameter.

```julia
GreekProblem(pricing_problem, lens)
```

where `lens` is an accessor (from Accessors.jl) that specifies which parameter to differentiate with respect to.

---

### [`SecondOrderGreekProblem`](@ref)

A `SecondOrderGreekProblem` represents a second-order sensitivity (e.g., gamma, volga) calculation.

```julia
SecondOrderGreekProblem(pricing_problem, lens1, lens2)
```

When `lens1` and `lens2` are identical, this computes a pure second derivative. When they differ, it computes a mixed partial derivative.

---

### [`BatchGreekProblem`](@ref)

The [`BatchGreekProblem`](@ref) allows calculating multiple Greeks in a single operation for efficiency.

```julia
spot_lens = @optic _.market_inputs.spot
vol_lens = VolLens(1,1)
batch_problem = BatchGreekProblem(pricing_problem, (spot_lens, vol_lens))
greeks = solve(batch_problem, ForwardAD(), BlackScholesAnalytic())
delta = greeks[spot_lens]
vega = greeks[vol_lens]
```

---

## Common Parameter Lenses

Hedgehog provides several accessor lenses for targeting specific parameters:

- `SpotLens()`: Accesses the spot price (for delta and gamma)
- `VolLens(i,j)`: Accesses volatility at index `(i,j)` in a surface (for vega and volga)
- `@optic _.market_inputs.spot`: Alternative for spot price using the Accessors.jl macro
- `@optic _.payoff.expiry`: For theta (time sensitivity)
- `ZeroRateSpineLens(i)`: For rho (interest rate sensitivity) for spine point `i`

## Examples

### Calculating Delta and Gamma

```julia
using Hedgehog
using Accessors  # For @optic macro
using Dates

# Define option and market inputs
strike = 100.0
expiry = Date(2023, 12, 31)
spot = 100.0
vol = 0.20
rate = 0.05
reference_date = Date(2023, 1, 1)

payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
market = BlackScholesInputs(reference_date, rate, spot, vol)
prob = PricingProblem(payoff, market)

# Define lens for spot price
spot_lens = @optic _.market_inputs.spot

# Calculate delta using various methods
delta_fd = solve(GreekProblem(prob, spot_lens), FiniteDifference(1e-4), BlackScholesAnalytic()).greek
delta_ad = solve(GreekProblem(prob, spot_lens), ForwardAD(), BlackScholesAnalytic()).greek
delta_an = solve(GreekProblem(prob, spot_lens), AnalyticGreek(), BlackScholesAnalytic()).greek

# Calculate gamma (second-order sensitivity to spot)
gamma_prob = SecondOrderGreekProblem(prob, spot_lens, spot_lens)
gamma = solve(gamma_prob, ForwardAD(), BlackScholesAnalytic()).greek
```

### Calculating Vega

```julia
# Define lens for volatility
vol_lens = VolLens(1, 1)

# Calculate vega
vega_prob = GreekProblem(prob, vol_lens)
vega = solve(vega_prob, ForwardAD(), BlackScholesAnalytic()).greek
```

### Batch Greek Calculation

```julia
# Calculate multiple Greeks efficiently
rate_lens = ZeroRateSpineLens(1) 
batch_prob = BatchGreekProblem(prob, (spot_lens, vol_lens, rate_lens))
all_greeks = solve(batch_prob, ForwardAD(), BlackScholesAnalytic())

# Extract individual Greeks
delta = all_greeks[spot_lens]
vega = all_greeks[vol_lens]
rho = all_greeks[rate_lens]
```

### Monte Carlo Greeks

Greeks can also be calculated for Monte Carlo pricing methods:

```julia
# Define Monte Carlo method
mc_method = MonteCarlo(
    LognormalDynamics(),
    BlackScholesExact(),
    SimulationConfig(10_000)
)

# Calculate delta using AD with Monte Carlo
delta_mc = solve(GreekProblem(prob, spot_lens), ForwardAD(), mc_method).greek
```
