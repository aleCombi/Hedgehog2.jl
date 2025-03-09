# Julia Pricing Framework Design

## **1. Overview**
This design implements a modular and extensible option pricing framework in Julia, leveraging **callable structs** and **multiple dispatch** for clean, type-safe pricing and Greek calculations.

## **2. Core Components**
### **2.1 Pricer**
- A callable struct that computes the price of a given **payoff** using a specific **pricing model**.
- Uses **multiple dispatch** to define pricing methods based on:
  - **Payoff type** (e.g., `VanillaEuropeanCall`)
  - **Market input type** (e.g., `BlackScholesInputs`)
  - **Pricing strategy** (e.g., `BlackScholesStrategy`, `BinomialTreeStrategy`)

#### **Example**
```julia
struct Pricer{P <: AbstractPayoff, M <: AbstractMarketInputs, S <: AbstractPricingStrategy}
    marketInputs::M
    payoff::P
    pricingStrategy::S
end

# Black-Scholes pricing method for European calls
function (pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesStrategy})()
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
end
```

---

### **2.2 DeltaCalculator**
- A callable struct that computes **Delta** (∂Price/∂Spot).
- Uses **swappable delta methods**:
  - **Analytical formula** (`BlackScholesAnalyticalDelta`)
  - **Automatic differentiation (AD)** (`ADDelta`)

#### **Example**
```julia
struct DeltaCalculator{M <: AbstractDeltaMethod, P <: AbstractPayoff, D <: AbstractMarketInputs, S <: AbstractPricingStrategy}
    pricer::Pricer{P, D, S}
    method::M
end

# Callable struct
(delta_calc::DeltaCalculator)() = compute_delta(delta_calc.method, delta_calc.pricer)
```

#### **Analytical Delta Calculation**
```julia
function compute_delta(
    ::BlackScholesAnalyticalDelta, 
    pricer::Pricer{VanillaEuropeanCall, BlackScholesInputs, BlackScholesStrategy}
)
    S = pricer.marketInputs.spot
    K = pricer.payoff.strike
    r = pricer.marketInputs.rate
    σ = pricer.marketInputs.sigma
    T = pricer.payoff.time
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return cdf(Normal(), d1)
end
```

#### **AD-Based Delta Calculation**
- Uses **Accessors.jl** to modify immutable market inputs.
- Uses **ForwardDiff.jl** to compute derivatives.
```julia
using Accessors, ForwardDiff

function compute_delta(::ADDelta, pricer::Pricer)
    ForwardDiff.derivative(
        S -> begin
            new_pricer = @set pricer.marketInputs.spot = S
            new_pricer()
        end,
        pricer.marketInputs.spot
    )
end
```

---

### **2.3 Generalized AD-Based Greeks**
- Instead of just `ADDelta`, we generalize the AD-based differentiation to **any market parameter**.
- Works for **Delta (∂Price/∂Spot)**, **Vega (∂Price/∂Sigma)**, and **Rho (∂Price/∂Rate)**.
```julia
function compute_greek(::ADGreek, pricer::Pricer, param::Symbol)
    ForwardDiff.derivative(
        x -> begin
            new_inputs = @set pricer.marketInputs.$param = x
            new_pricer = Pricer(new_inputs, pricer.payoff, pricer.pricingStrategy)
            new_pricer()
        end,
        getfield(pricer.marketInputs, param)
    )
end
```

#### **Usage**
```julia
compute_greek(ADGreek(), pricer, :spot)   # Delta
compute_greek(ADGreek(), pricer, :sigma)  # Vega
compute_greek(ADGreek(), pricer, :rate)   # Rho
```

---

## **3. Leveraging Multiple Dispatch for Generalization**
Instead of defining separate pricing methods for every payoff, we allow models like **Binomial Tree** to handle multiple payoffs:
```julia
function (pricer::Pricer{P, BlackScholesInputs, BinomialTreeStrategy}) where {P <: AbstractPayoff}
    # Binomial tree pricing logic that works for all P <: AbstractPayoff
end
```
This ensures:
✅ **Black-Scholes stays restricted to specific payoffs**  
✅ **Binomial/MCMC models work generically for many payoffs**  
✅ **Easy to extend with new pricing strategies**  

---

## **4. Performance Benchmarking**
We use `BenchmarkTools.jl` to compare analytical vs. AD-based Greeks:
```julia
println("Benchmarking Analytical Delta:")
@btime analytical_delta_calc()

println("Benchmarking AD Delta:")
@btime ad_delta_calc()
```
✅ **Analytical methods should be faster**  
✅ **AD-based methods are more flexible but may be slower**  

---

## **5. Key Takeaways**
- **Callables make the API clean** (`pricer()`, `delta_calc()`).
- **Multiple dispatch makes it easy to add new models & payoffs**.
- **Automatic Differentiation generalizes Greeks to all market inputs**.
- **Performance benchmarking ensures models scale efficiently**.

---

## **6. Next Steps**
1. **Implement Binomial Tree pricing with the same callable dispatch pattern.**
2. **Extend the framework to handle second-order Greeks (Gamma, Vanna, Volga).**
3. **Benchmark Monte Carlo pricing vs. Black-Scholes vs. Binomial Tree.**

🚀 **This framework is highly modular and ready for future extensions!**
