**Hedgehog2** is a modular derivatives pricing library in Julia.

[![Build Status](https://github.com/aleCombi/Hedgehog2.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/Hedgehog2.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Typical usage: define a price by combining the following three components, each chosen to be compatible with the others:

- A **payoff** (e.g. European Call, American Put)
- A set of **market inputs** (e.g. BlackScholes inputs, Heston inputs, later Discount Curves and Vol Surfaces)
- A **pricing method** (e.g. analytical, Monte Carlo, PDE, Fourier)

## Supported Payoffs

- EuropeanCall / EuropeanPut
- AmericanCall / AmericanPut
- Digital, Barrier, Asian (in progress)

## Supported Models

- Black-Scholes (LognormalDynamics)
- Heston
- Hull-White (short rates)
- More models planned: Variance Gamma, Rough Bergomi

## Pricing Methods

- Analytical formulas (Black-Scholes)
- Binomial trees (CRR)
- Monte Carlo (Euler, exact simulations of BlackScholes and Heston using Broadie-Kaya method)
- PDE (Crank-Nicolson), in progress
- Fourier (Carr-Madan), COS in progress

## Notes

- Sensitivities (Greeks) supported via AD or finite differences (in progress)
- Components are swappable and extensible by design
- The goal is to prototype, test, and extend new models and methods

## License

MIT