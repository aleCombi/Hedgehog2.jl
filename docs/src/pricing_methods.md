# Pricing Methods

Hedgehog implements various pricing methods as subtypes of `AbstractPricingMethod` [See API](api.md#AbstractPricingMethod). Each method can be used with the unified `solve(problem, method)` [See API](api.md#solve) interface.

## BlackScholesAnalytic

The `BlackScholesAnalytic` [See API](api.md#BlackScholesAnalytic) method implements the closed-form Black-Scholes formula for European vanilla options. It provides exact pricing for call and put options under lognormal dynamics with constant volatility and interest rate.

```julia
solution = solve(problem, BlackScholesAnalytic())
```

## CoxRossRubinsteinMethod

The `CoxRossRubinsteinMethod` [See API](api.md#CoxRossRubinsteinMethod) implements a binomial tree approach for pricing both European and American options. It discretizes the underlying price process into a lattice of possible future prices and uses backward induction to determine the option value.

```julia
solution = solve(problem, CoxRossRubinsteinMethod(800))  # 800 time steps
```

## MonteCarlo

The `MonteCarlo` [See API](api.md#MonteCarlo) method simulates multiple paths of the underlying price process and averages the resulting payoffs to estimate the option price. It's highly flexible and can be configured with different dynamics, simulation strategies, and variance reduction techniques.

```julia
mc_method = MonteCarlo(
    LognormalDynamics(),
    BlackScholesExact(),
    SimulationConfig(10_000)  # 10,000 paths
)
solution = solve(problem, mc_method)
```

For more details on configuration options, see:
- `LognormalDynamics` [See API](api.md#LognormalDynamics)
- `BlackScholesExact` [See API](api.md#BlackScholesExact)
- `SimulationConfig` [See API](api.md#SimulationConfig)

## LSM

The `LSM` [See API](api.md#LSM) (Least Squares Monte Carlo) method implements the Longstaff-Schwartz algorithm for pricing American options via Monte Carlo simulation. It uses polynomial regression to estimate continuation values at each exercise opportunity.

```julia
lsm_method = LSM(
    LognormalDynamics(),
    BlackScholesExact(),
    SimulationConfig(10_000),
    5  # Polynomial degree for regression
)
solution = solve(problem, lsm_method)
```

## CarrMadan

The `CarrMadan` [See API](api.md#CarrMadan) method uses Fourier transform techniques to price options under general exponential Lévy models. It's particularly useful for stochastic volatility models like Heston, where the characteristic function is known in closed form.

```julia
carr_madan_method = CarrMadan(1.0, 32.0, HestonDynamics())
solution = solve(problem, carr_madan_method)
```

For more details on Heston dynamics, see `HestonDynamics` [See API](api.md#HestonDynamics).
