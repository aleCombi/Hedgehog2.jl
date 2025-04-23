# ✅ Derivatives Pricing Library — Roadmap Checklist (Updated)

---

## PHASE 1 — Core Pricing Engine [✅ Completed]

- [x] Black-Scholes pricing (analytic, MC: exact + Euler)
- [x] Heston pricing (Euler + Broadie–Kaya)
- [x] Binomial tree (European & American)
- [x] Least Squares Monte Carlo (LSM)
- [x] Carr–Madan method (FFT)
- [x] Pricer structure with method, marketinputs, payoff
- [x] Test suite with unit tests and basic price agreements

---

## PHASE 2 — Market Inputs & Calibration Basics [✅ Completed]

- [x] `BSVolSurface` with rectangular grid + interpolation and `get_vol(t)` accessor
- [x] Implied vol inversion for vanilla options
- [x] `calibrate_vol_surface(quotes)` from market prices
- [x] `DiscountCurve` (flat, piecewise) and `df(t)` accessor
- [x] Unit tests: vol/curve access, interpolation, smoke pricing
- [x] Rate Curves: Usage instead of constant rates, convenience constructors

---

## PHASE 3 — Greeks & Calibration Infrastructure [✅ Completed]

- [x] Implement `GreekProblem` and `solve` with support for:
  - [x] Finite differences (forward, backward, central)
  - [x] Forward AD
  - [x] Analytic Greeks for Black-Scholes
  - [x] Optional: AD for Monte Carlo methods
- [x] Plug Greeks into pricing pipeline with consistent API
- [x] Develop modular calibration system:
  - [x] Residual-based calibration engine
  - [x] Objective functions from market quotes
  - [x] Support for different pricer-model combinations
  - [x] Heston calibration to implied volatility surface
- [x] Unit tests: Greeks accuracy vs known formulas, calibration residual shapes

---

## PHASE 3.5 — Monte Carlo Enhancements [✅ Completed]

- [x] Add antithetic variates toggle to `MonteCarlo` method (Discrete and Exact Black-Scholes [x], Discrete and Exact Heston [x])
- [x] Unit tests: check variance reduction and correctness vs analytic prices
- [x] Seed reproducibility: allow full control over RNG seeds (for all MC types)
- [x] Refactor MC framework to support seed injection and custom ensemble configs
- [x] Optional: design placeholder for `MCStrategy`, but defer advanced variance reduction

---

## 🛠 PHASE 4 — Polishing & Pre-Release Prep [✅ Completed]

**🌟 Goal**: Registry submission with clean, reproducible functionality and docs

- [x] Public API audit and finalization
- [x] Add docstrings for all exported types/functions
- [x] Create `/docs/` folder with basic Documenter.jl setup
- [x] Include example scripts: pricing, Greeks, calibration
- [x] Add Project.toml metadata (UUID, compat)
- [x] Ensure full test suite runs via `Pkg.test()`
- [x] CI: GitHub Actions with coverage and doc build
- [x] Tag `v0.1.0` and submit to Julia registry
- [ ] Soft launch: share with trusted peers, hold off on broad announcement

---

## PHASE 5 — Structured Payoffs

- [ ] Arithmetic Asian pricing via Monte Carlo
- [ ] Geometric Asian pricing via closed-form
- [ ] Digital options (cash-or-nothing, asset-or-nothing)
- [ ] Barrier options (up/down, knock-in/out, discrete monitoring)
- [ ] Extend `Payoff` with `Monitoring` / `Averaging` modifiers
- [ ] Unit tests: price agreement across methods, monitoring edge cases

---

## PHASE 6 — Interest Rate Products & Models

- [ ] Implement zero-coupon and fixed-coupon bond pricing
- [ ] Build FRA and IRS support with schedule engine and stub logic
- [ ] Cap/floor pricing using Black formula
- [ ] Hull–White short rate model
- [ ] Swaption pricing via Jamshidian decomposition or PDE method
- [ ] Basic CDS pricing with flat hazard rate model
- [ ] Unit tests: replication of swaps via bond strips, curve usage, date handling

---

## PHASE 7 — Multi-Curve Support & Calibration

- [ ] Introduce multi-curve framework (OIS + forwarding curves)
- [ ] Curve bootstrapping: deposit, futures, swaps
- [ ] Curve interpolation: ZC rates, discount factors, log-linear extrapolation
- [ ] Integrate multi-curve into FRA, IRS, caps/floors pricing
- [ ] Unit tests: bootstrapping accuracy, multi-curve vs single-curve comparisons

---

## PHASE 7.5 — PDE Framework

- [ ] Implement Crank–Nicolson solver for Black-Scholes PDE
- [ ] Define `PDEProblem` type with boundary conditions and solver settings
- [ ] Generalize solver infrastructure for future PDE models
- [ ] Unit tests: convergence checks, price agreement with analytic/MC methods

---

## PHASE 8 — SABR, Local Vol, Rough Vol

- [ ] SABR model implementation (Hagan approximation)
- [ ] Calibration to vanilla vol surface
- [ ] Dupire local vol generation from calibrated surface
- [ ] Rough Heston or rBergomi sampling engine
- [ ] Fourier or Monte Carlo pricing for rough volatility models
- [ ] Unit tests: calibration error metrics, simulation path sanity checks

---

## PHASE 9 — Robustness & Performance

- [ ] Edge case test suite: deep ITM/OTM, short maturity, extreme vol
- [ ] Sensitivity exploration tools (Greeks wrt model parameters)
- [ ] Allocation-free Monte Carlo and LSM refactoring
- [ ] Batch pricing support for performance testing
- [ ] Parallel/multithreaded MC and calibration support
- [ ] Optional: GPU backend for simulation
- [ ] CI: Full unit test and regression test coverage, GitHub Actions setup

---

## PHASE 10 — Model-Free Pricing & Arbitrage Detection

- [ ] Risk-neutral density extraction from call spread surface
- [ ] Variance swap pricing via replication
- [ ] Digital option replication using spreads
- [ ] Arbitrage checks: calendar spreads, butterfly arbitrage
- [ ] Super-replication bounds for exotic payoffs
- [ ] Unit tests: convexity and arbitrage consistency, replication sanity

---

## PHASE 11 — Documentation & Launch Prep

- [ ] Full docstrings for all exported types and functions
- [ ] Notebooks covering pricing, Greeks, calibration, and structured payoffs
- [ ] CONTRIBUTING.md and internal dev onboarding guide
- [ ] Smoke-tested example scripts folder
- [ ] Test coverage badge and CI status reporting
- [ ] Prepare for public release: package registry, landing README
- [ ] Soft launch with trusted peers; optionally post to LinkedIn, Discourse

---

## PHASE 12 — Optional Extensions

- [ ] Control variates using Black-Scholes
- [ ] Stratified sampling / quasi-random generation
- [ ] REST API or service wrapper for pricing/calibration endpoints
- [ ] Pluto.jl-based interactive pricing UI
- [ ] Real-time batch pricing and risk engine prototype
- [ ] Signature-based models (rough paths theory)
- [ ] Structured product builder (range accruals, callable notes)
- [ ] QUAD method revisitation and benchmarking
