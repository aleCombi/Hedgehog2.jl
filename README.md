# Hedgehog

**Hedgehog** is a modular, SciML-inspired derivatives pricing library in Julia.

[![Build Status](https://github.com/aleCombi/Hedgehog.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/Hedgehog.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alecombi.github.io/Hedgehog.jl/)

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
- Planned: Digital and Barrier, Asian

## 🧠 Supported Models (Price Dynamics)

- Black-Scholes (`LognormalDynamics`)
- Heston (`HestonDynamics`)
- Planned: Hull-White (short-rate), Variance Gamma, Rough Bergomi

## ⚙️ Pricing Methods

- Analytical formulas (Black-Scholes)
- Binomial Trees (Cox–Ross–Rubinstein)
- Monte Carlo:
  - Euler–Maruyama
  - Exact simulation (Black-Scholes, Broadie–Kaya for Heston)
- Fourier methods (Carr–Madan; COS coming soon)
- PDE methods (Crank–Nicolson, in progress)

## 📊 Calibration

Hedgehog supports calibration via a unified interface:

- Solve for implied volatility using `CalibrationProblem`
- Invert volatility surfaces
- Build fully calibrated `RectVolSurface` objects from price matrices

## 🧮 Sensitivities

- Greeks supported via a `GreekProblem` interface:
  - Finite differences
  - Automatic differentiation
  - `BatchGreekProblem` to compute a full gradient of sensitivities

## 🚀 Highlights

- Modular by construction: models, payoffs, and methods are swappable
- Unified `solve(problem, method)` interface across pricing and calibration
- Inspired by the SciML architecture and ecosystem
- Built on top of SciML components (StochasticDiffEq.jl, NonlinearSolve.jl, Integrals.jl)
- Open-source and focused on prototyping cutting-edge methods

## 📦 Dependencies

Hedgehog is built on several high-quality Julia packages:

- [SciML Ecosystem](https://sciml.ai/): 
  - [StochasticDiffEq.jl](https://github.com/SciML/StochasticDiffEq.jl) - For stochastic simulation
  - [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) - For calibration and implied volatility
  - [Integrals.jl](https://github.com/SciML/Integrals.jl) - For Fourier-based pricing methods
  - [Optimization.jl](https://github.com/SciML/Optimization.jl) - For advanced calibration
  - [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) - For volatility surfaces and rate curves

- [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) - Functional lens-based access for Greeks and calibration
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) - For statistical distributions and sampling

## 📖 Documentation

Complete documentation is available at [https://alecombi.github.io/Hedgehog.jl/](https://alecombi.github.io/Hedgehog.jl/)