# ✅ Derivatives Pricing Library – Roadmap Checklist

## PHASE 1 — Core Pricing Engine [✅ Completed]

- [x] Black-Scholes pricing (analytic, MC: exact + Euler)
- [x] Heston pricing (Euler + Broadie–Kaya)
- [x] Binomial tree (European & American)
- [x] Least Squares Monte Carlo (LSM)
- [x] Carr–Madan method (FFT)
- [x] Pricer structure with method, marketinputs, payoff
- [x] Test suite with unit tests and basic price agreements

## PHASE 2 — Market Inputs & Calibration (2–3 weeks)

- [x] `BSVolSurface` with rectangular grid + interpolation and `get_vol(t)` accessor
- [x] Implied vol inversion for vanilla options
- [x] `calibrate_vol_surface(quotes)` from market prices
- [x] `DiscountCurve` (flat, piecewise) and `df(t)` accessor
- [x] Unit tests: vol/curve access, interpolation, smoke pricing
- [ ] Rate Curves: Usage instead of constant rates, convenience constructors.

## PHASE 3 — PDE, Greeks, Heston Calibration (4–5 weeks)

- [ ] Crank–Nicolson solver for BS PDE
- [ ] Greeks framework (FD + analytic for BS)
- [ ] Plug Greeks into pricing pipeline
- [ ] Heston calibration to implied vol surface
- [ ] Unit tests: PDE convergence, Greeks shape + accuracy, calibration residuals


## PHASE 4 — Structured Payoffs: Asians, Digitals, Barriers (3–4 weeks)

- [ ] Arithmetic Asian (MC)
- [ ] Geometric Asian (closed form)
- [ ] Digital options (cash-or-nothing, asset-or-nothing)
- [ ] Barrier options (knock-in/out, discrete MC)
- [ ] Payoff extensions for monitoring / averaging
- [ ] Unit tests: payoff logic, price agreement, monitoring


## PHASE 5 — IR Products & Models (4 weeks)

- [ ] Zero-coupon and coupon bond pricing
- [ ] FRA, IRS with schedule engine
- [ ] Cap/floor pricing using Black model
- [ ] Hull–White model implementation
- [ ] Swaption pricing via Jamshidian or PDE
- [ ] Basic CDS pricing (flat hazard curve)
- [ ] Unit tests: cashflow valuation, IR curve usage, swap replication


## PHASE 5.5 — Multi-Curve Support & Calibration (3 weeks)

- [ ] Multi-curve struct: OIS + forwarding curves
- [ ] Curve bootstrapping (deposits, futures, swaps)
- [ ] Curve interpolation (ZC, DF, log-linear)
- [ ] Integration into FRA, IRS, caps
- [ ] Unit tests: calibration residuals, multi-curve pricing


## PHASE 6 — SABR, Local Vol, Rough Vol (4–6 weeks)

- [ ] SABR model (Hagan) + calibration
- [ ] SABR-implied vol surface
- [ ] Local vol generation (Dupire)
- [ ] Rough Heston or rBergomi simulation
- [ ] MC or Fourier pricing for rough models
- [ ] Unit tests: calibration error, rough sim stability


## PHASE 7 — Robustness & Performance (4 weeks)

- [ ] Hedge case test suite (deep ITM, expiry, high/low vol)
- [ ] Sensitivity testing tools (Greeks vs model parameters)
- [ ] Allocation-free Monte Carlo + LSM
- [ ] Batch pricing support
- [ ] Multithreaded MC / calibration
- [ ] Optional: GPU support
- [ ] Full unit + regression test coverage
- [ ] GitHub Actions or CI setup


## PHASE 8 — Model-Free Pricing & Arbitrage Detection (2–3 weeks)

- [ ] Risk-neutral density from call spread surface
- [ ] Variance swap replication (model-free)
- [ ] Digital replication from spreads
- [ ] Arbitrage checks (calendar, butterfly)
- [ ] Super-replication bounds
- [ ] Unit tests: convexity checks, replication consistency


## PHASE 9 — Documentation & Launch Prep (2–3 weeks)

- [ ] Full docstrings and module-level documentation
- [ ] Notebooks: pricing, calibration, Greeks, structured payoffs
- [ ] CONTRIBUTING.md + dev onboarding notes
- [ ] Smoke-tested example folder
- [ ] Test coverage badge
- [ ] Prepare public repo / registry package
- [ ] Soft launch with trusted peers
- [ ] Optional public launch (LinkedIn, Discourse, blog post)


## PHASE 10 — Optional Extensions

- [ ] REST API wrapper (pricing, calibration)
- [ ] Pluto.jl-based interactive UI
- [ ] Real-time batch pricing or risk engine
- [ ] Signature-based models (rough paths theory)
- [ ] Structured product generator (range accruals, callable notes)
- [ ] QUAD method (revisit and polish implementation)
