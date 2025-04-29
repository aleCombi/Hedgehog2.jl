# Interactive Examples

Hedgehog.jl includes several interactive [Pluto](https://github.com/fonsp/Pluto.jl) notebooks that demonstrate key concepts in derivatives pricing with rich visualizations and real-time parameter adjustment.

## Available Notebooks

- [Black-Scholes Pricing](https://github.com/aleCombi/Hedgehog.jl/blob/master/examples/notebooks/pluto_black_scholes.jl) - Interactive exploration of option prices, delta, and vega as functions of spot price and volatility
  
- [Monte Carlo Methods](https://github.com/aleCombi/Hedgehog.jl/blob/master/examples/notebooks/pluto_montecarlo.jl) - Visualization of price convergence, path generation, and variance reduction techniques

The notebooks demonstrate key features of Hedgehog.jl:

**Black-Scholes Notebook:**
- Visualization of vanilla option payoffs
- Interactive sliders to see how volatility affects option prices
- Delta and vega curves across different strike prices
- Real-time calculation using Hedgehog's analytical pricing methods

**Monte Carlo Notebook:**
- Path generation visualization using different simulation strategies
- Convergence analysis as the number of simulations increases
- Demonstration of variance reduction techniques
- Integration with automatic differentiation for Greeks calculation

## Running the Notebooks

To run these notebooks, you'll need to have Pluto.jl installed:

```julia
using Pkg
Pkg.add("Pluto")
```

Then, you can open any notebook with:

```julia
using Pluto
Pluto.run(notebook="path/to/Hedgehog.jl/examples/notebooks/pluto_black_scholes.jl")
```

For the best experience, make sure you have all dependencies installed by first running:

```julia
# Navigate to the examples directory
cd("path/to/Hedgehog.jl/examples")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

These interactive examples complement the static documentation by allowing you to experiment with different parameters and see the immediate effects on pricing and risk measures, building intuition about derivatives pricing concepts.