# Hedgehog.jl

Welcome to **Hedgehog.jl** — a modular, high-performance derivatives pricing library written in Julia.

Hedgehog is designed with composability and clarity in mind, targeting quantitative finance professionals, academics, and advanced users who need full control over models, calibration, and pricing algorithms.

---

## 🚀 Features

- ✅ Modular pricers with support for Monte Carlo, PDE, Fourier, and analytic methods  
- ✅ Support for volatility surfaces, rate curves, and calibration infrastructure  
- ✅ Composable payoffs and flexible pricing problems  
- ✅ Automatic and finite difference Greeks using lens-based access  
- ✅ Simulation strategies and exact SDE schemes (e.g., Broadie-Kaya for Heston)  
- ✅ Performance-aware design with full AD support  
- ✅ Full test suite and example scripts

---

## 🧠 Philosophy

Hedgehog treats models, payoffs, and pricing methods as **first-class, composable objects**, allowing:
- easy experimentation with new models,
- separation of concerns (dynamics, simulation, pricing),
- and fast iteration on complex structures like baskets or multi-curve setups.

---

## 💡 Getting Started

Install from source:

```julia
using Pkg
Pkg.add(url="https://github.com/aleCombi/Hedgehog.jl")
```

Basic example:

```julia
using Hedgehog
using Dates
using Accessors  # For @optic macro

# Define the option parameters
strike = 100.0
reference_date = Date(2023, 1, 1)
expiry_date = reference_date + Year(1)
rate = 0.05
spot = 100.0
sigma = 0.20

# Create the payoff and market inputs
payoff = VanillaOption(strike, expiry_date, European(), Call(), Spot())
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Create pricing problem and solve it
problem = PricingProblem(payoff, market_inputs)
solution = solve(problem, BlackScholesAnalytic())

# Access the price
println("Option price: ", solution.price)

# Calculate delta (sensitivity to spot price)
spot_lens = @optic _.market_inputs.spot
delta = solve(GreekProblem(problem, spot_lens), ForwardAD(), BlackScholesAnalytic()).greek
println("Delta: ", delta)
```

---

## 🛣️ Roadmap

Hedgehog development follows a structured roadmap with clearly defined phases:

1. Core pricing engines
2. Volatility surfaces and yield curves
3. Calibration and sensitivities
4. PDE solvers and early exercise methods
5. Advanced stochastic volatility and local volatility models
6. Portfolio and basket pricing
7. Performance and production features
8. Final polish and publication

Check the internal ADRs for more details on the design and implementation plan.

---

## 📖 Documentation Sections

- **API Reference**: Full function and type documentation
- **Examples**: Example scripts to reproduce test cases and pricing results
- **Design Notes**: Detailed internal architecture
- **Pricing Methods**: Overview of available pricing algorithms
