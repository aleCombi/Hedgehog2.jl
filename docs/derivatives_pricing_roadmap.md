
# ✅ Derivatives Pricing Library – Roadmap Checklist (Updated)

## PHASE 1 — Core Pricing Engine [✅ Completed]

- [x] Black-Scholes pricing (analytic, MC: exact + Euler)
- [x] Heston pricing (Euler + Broadie–Kaya)
- [x] Binomial tree (European & American)
- [x] Least Squares Monte Carlo (LSM)
- [x] Carr–Madan method (FFT)
- [x] Pricer structure with method, marketinputs, payoff
- [x] Test suite with unit tests and basic price agreements

## PHASE 2 — Market Inputs & Calibration Basics [✅ Completed]

- [x] `BSVolSurface` with rectangular grid + interpolation and `get_vol(t)` accessor
- [x] Implied vol inversion for vanilla options
- [x] `calibrate_vol_surface(quotes)` from market prices
- [x] `DiscountCurve` (flat, piecewise) and `df(t)` accessor
- [x] Unit tests: vol/curve access, interpolation, smoke pricing
- [x] Rate Curves: Usage instead of constant rates, convenience constructors.

## PHASE 3 — Greeks & Calibration Infrastructure (2–3 weeks)

- [x] Implement `GreekProblem` and `solve` with support for:
- [x] Finite differences (forward, backward, central)
- [x] Forward AD
- [x] Analytic Greeks for Black-Scholes
- [x] Optional: AD for Monte Carlo methods
- [x] Plug Greeks into pricing pipeline with consistent API
- [ ] Develop modular calibration system:
- [ ] Residual-based calibration engine
- [ ] Objective functions from market quotes
- [ ] Support for different pricer-model combinations
- [ ] Heston calibration to implied volatility surface
- [ ] Unit tests: Greeks accuracy vs known formulas, calibration residual shapes

## PHASE 3.5 — Monte Carlo Enhancements (1 week)

- [ ] Add antithetic variates toggle to `MonteCarlo` method (Exact Black Scholes [x])
- [ ] Implement control variates with BS analytical formulas
- [ ] Add reproducibility features: seeded RNG + control panel
- [ ] Refactor MC framework to support pluggable variance reduction via `MCStrategy`
- [ ] Optional: add stratified / quasi-random sampling hooks
- [ ] Unit tests: check variance reduction and correctness vs analytic prices

## PHASE 4 — Structured Payoffs (2–3 weeks)

- [ ] Extend `Payoff` system to support path-dependent and monitored options
- [ ] Arithmetic Asian pricing via Monte Carlo
- [ ] Geometric Asian pricing via closed-form solution
- [ ] Cash-or-nothing and asset-or-nothing digital options
- [ ] Barrier options (up/down, knock-in/out, discrete monitoring)
- [ ] Add `Monitoring` and `Averaging` traits or modifiers
- [ ] Unit tests: price agreement across methods, monitoring edge cases

## PHASE 5 — Interest Rate Products & Models (3–4 weeks)

- [ ] Implement zero-coupon and fixed-coupon bond pricing
- [ ] Build FRA and IRS support with schedule engine and stub logic
- [ ] Cap/floor pricing using Black formula
- [ ] Hull–White short rate model
- [ ] Swaption pricing via Jamshidian decomposition or PDE method
- [ ] Basic CDS pricing with flat hazard rate model
- [ ] Unit tests: replication of swaps via bond strips, curve usage, date handling

## PHASE 6 — Multi-Curve Support & Calibration (3 weeks)

- [ ] Introduce multi-curve framework (OIS + forwarding curves)
- [ ] Curve bootstrapping: deposit, futures, swaps
- [ ] Curve interpolation: ZC rates, discount factors, log-linear extrapolation
- [ ] Integrate multi-curve into FRA, IRS, caps/floors pricing
- [ ] Unit tests: bootstrapping accuracy, multi-curve vs single-curve comparisons

## PHASE 6.5 — PDE Framework (2–3 weeks)

- [ ] Implement Crank–Nicolson solver for Black-Scholes PDE
- [ ] Define `PDEProblem` type with boundary conditions and solver settings
- [ ] Generalize solver infrastructure for future PDE models
- [ ] Unit tests: convergence checks, price agreement with analytic/MC methods

## PHASE 7 — SABR, Local Vol, Rough Vol (4–5 weeks)

- [ ] SABR model implementation (Hagan approximation)
- [ ] Calibration to vanilla vol surface
- [ ] Dupire local vol generation from calibrated surface
- [ ] Rough Heston or rBergomi sampling engine
- [ ] Fourier or Monte Carlo pricing for rough volatility models
- [ ] Unit tests: calibration error metrics, simulation path sanity checks

## PHASE 8 — Robustness & Performance (3–4 weeks)

- [ ] Edge case test suite: deep ITM/OTM, short maturity, extreme vol
- [ ] Sensitivity exploration tools (Greeks wrt model parameters)
- [ ] Allocation-free Monte Carlo and LSM refactoring
- [ ] Batch pricing support for performance testing
- [ ] Parallel/multithreaded MC and calibration support
- [ ] Optional: GPU backend for simulation
- [ ] CI: Full unit test and regression test coverage, GitHub Actions setup

## PHASE 9 — Model-Free Pricing & Arbitrage Detection (2–3 weeks)

- [ ] Risk-neutral density extraction from call spread surface
- [ ] Variance swap pricing via replication
- [ ] Digital option replication using spreads
- [ ] Arbitrage checks: calendar spreads, butterfly arbitrage
- [ ] Super-replication bounds for exotic payoffs
- [ ] Unit tests: convexity and arbitrage consistency, replication sanity

## PHASE 10 — Documentation & Launch Prep (2 weeks)

- [ ] Full docstrings for all exported types and functions
- [ ] Notebooks covering pricing, Greeks, calibration, and structured payoffs
- [ ] CONTRIBUTING.md and internal dev onboarding guide
- [ ] Smoke-tested example scripts folder
- [ ] Test coverage badge and CI status reporting
- [ ] Prepare for public release: package registry, landing README
- [ ] Soft launch with trusted peers; optionally post to LinkedIn, Discourse

## PHASE 11 — Optional Extensions

- [ ] REST API or service wrapper for pricing/calibration endpoints
- [ ] Pluto.jl-based interactive pricing UI
- [ ] Real-time batch pricing and risk engine prototype
- [ ] Signature-based models (rough paths theory)
- [ ] Structured product builder (range accruals, callable notes)
- [ ] QUAD method revisitation and benchmarking
