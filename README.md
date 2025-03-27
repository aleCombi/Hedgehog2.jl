# Hedgehog2

**Hedgehog2** is a modular, SciML-inspired derivatives pricing library in Julia.

[![Build Status](https://github.com/aleCombi/Hedgehog2.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/Hedgehog2.jl/actions/workflows/CI.yml?query=branch%3Amaster)

## 📐 Design Overview

All pricing and calibration workflows follow a SciML-inspired `solve(problem, method)` interface.

To compute a price, you define a `PricingProblem` using:

- A **payoff** (e.g. European call, American put)
- A set of **market inputs** (e.g. Black-Scholes inputs, Heston inputs)
- Then solve it with a **pricing method** (e.g. Monte Carlo, analytical formula, Fourier inversion)

```julia
payoff = VanillaOption(...)
market = BlackScholesInputs(...)
problem = PricingProblem(payoff, market)

sol = solve(problem, BlackScholesAnalytic())
price = sol.price
```

## ✅ Supported Payoffs

- European Call / Put
- American Call / Put
- Digital and Barrier (in progress)
- Asian and Path-Dependent (planned)

## 🧠 Supported Models (Price Dynamics)

- Black-Scholes (`LognormalDynamics`)
- Heston
- Hull-White (short-rate)
- Planned: Variance Gamma, Rough Bergomi

## ⚙️ Pricing Methods

- Analytical formulas (Black-Scholes)
- Binomial Trees (Cox–Ross–Rubinstein)
- Monte Carlo:
  - Euler–Maruyama
  - Exact simulation (Black-Scholes)
  - Broadie–Kaya for Heston
- Fourier methods (Carr–Madan; COS coming soon)
- PDE methods (Crank–Nicolson, in progress)

## 📊 Calibration

Hedgehog2 supports calibration via a unified nonlinear solver interface:

- Solve for implied volatility using `CalibrationProblem`
- Invert volatility surfaces
- Build fully calibrated `RectVolSurface` objects from price matrices

## 🧮 Sensitivities

- Greeks supported via:
  - Finite differences
  - Automatic differentiation (planned)
- Extensible `GreekProblem` interface is under development

## 🚀 Highlights

- Modular by construction: models, payoffs, and methods are swappable
- Unified `solve(problem, method)` interface across pricing and calibration
- Inspired by the SciML architecture and ecosystem
- Built on top of SciML components (DifferentialEquations.jl, NonlinearSolve.jl, Integrals.jl)
- Open-source and focused on prototyping cutting-edge methods

## 📄 License

MIT
