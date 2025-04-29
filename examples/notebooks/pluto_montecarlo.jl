### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ f091db10-24d5-11f0-1c87-db62c36b849a
begin
	import Pkg
	Pkg.activate("..")  # activate the current folder (if opened inside `examples/notebooks`)
	Pkg.instantiate()  # fetch any missing packages
end

# ╔═╡ d48916b3-36df-439a-8975-fb96d1f89d85
begin
	using Hedgehog, Dates
	# Set up option parameters
	strike = 100.0
	reference_date = Date(2023, 1, 1)
	expiry = reference_date + Year(1)
	
	# Create the payoff (European call option)
	payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
end

# ╔═╡ 385323b8-78f6-4eb9-91f0-4e18a38abf3c
md"# 📈 Monte Carlo Option Pricing in Hedgehog

In this notebook we demonstrate how Monte Carlo methods can be set up to price options using Hedgehog.jl. We'll explore:

1. Setting up option dynamics and simulation methods
2. Calculating price sensitivities (Greeks)
3. Using variance reduction techniques 
4. Ensuring reproducibility with explicit seeding

Monte Carlo is particularly valuable for complex derivatives where closed-form solutions don't exist, or when the dimensionality makes other numerical methods infeasible.
"

# ╔═╡ 715c0441-2d66-4806-871c-a9624f461a23
md"## Payoff Definition
We define a European call option with a strike of 100, expiring in one year from the reference date. This is our contract specification that will be priced.
"

# ╔═╡ 9fa72ea4-c8c8-4676-9391-01efe61921aa
md"## Price Dynamics
In order to get a price, we make assumptions on the underlying dynamics. In the Black-Scholes model, the dynamics are the following under the risk-neutral measure:
```math
dS_t = r S_t \, dt + \sigma S_t \, dW_t
```

Where:
- ``S_t`` is the price of the underlying asset at time $t$
- ``r`` is the risk-free interest rate
- ``\sigma`` is the volatility of the underlying asset
- ``W_t`` is a standard Brownian motion

We'll now define the necessary market inputs: the model parameters $r$ and $\sigma$, today's date, and the spot price.
"

# ╔═╡ 58a53919-f63a-4345-adfc-8d571b2ec6ec
begin
	rate = 0.03
	spot = 100.0
	sigma = 0.4
	market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
end

# ╔═╡ 4a547637-2652-474a-a615-8a4c992fe0ee
md"## Pricing Problem
In Hedgehog.jl, a payoff and a market inputs object are the necessary ingredients to formulate a pricing problem, which we can solve using different methods. This follows the SciML-inspired pattern of defining a problem type separately from the solution method."

# ╔═╡ a7b471c1-ff78-4562-a3e4-c5294d16928c
pricing_problem = PricingProblem(payoff, market_inputs)

# ╔═╡ 0e3e4511-4a3a-4e7f-86fd-e77c5697d921
md"## Monte Carlo Setup
We now configure our Monte Carlo simulation by defining:

1. **Dynamics**: The stochastic process model (Black-Scholes)
2. **Simulation strategy**: The numerical scheme for generating paths
3. **Configuration**: Number of paths, time steps, and variance reduction techniques

For Black-Scholes, we can use an exact simulation since the terminal distribution is known analytically. This is more efficient than Euler-Maruyama discretization for simple models.
"

# ╔═╡ 95309345-9d8f-47c7-88a5-3023a515b016
begin
	dynamics = LognormalDynamics() # Black-Scholes dynamics
	simulation_strategy = BlackScholesExact() # exact simulation using the law of S_T
	trajectories = 100_000
	steps = 1 # with an exact scheme, no need to take several steps 
	config = SimulationConfig(trajectories, steps=steps, variance_reduction=Hedgehog.Antithetic()) # Configure using antithetic variates for variance reduction
	mc_method = MonteCarlo(dynamics, simulation_strategy, config)
end

# ╔═╡ d47790fb-82f4-4910-bc84-bc179af2e63c
md"## Analytic Benchmark
To check the accuracy of our Monte Carlo method, we'll compare it to the analytic Black-Scholes formula, which provides the exact theoretical price for our European call option."

# ╔═╡ 6f603b05-f72a-40a6-8b55-49fc4cc2e48c
analytic_method = BlackScholesAnalytic()

# ╔═╡ 6e8b47b7-6a4d-40fa-bddd-e747cf9d2edc
analytic_solution = solve(pricing_problem, analytic_method).price

# ╔═╡ 6df778b3-9085-44ba-bb7f-443b0218a3ff
begin
	using Accessors
	using Plots
	default(
		linewidth = 2,
		framestyle = :box,
		grid = true,
		gridlinewidth = 1,
		gridalpha = 0.3,
		guidefontsize = 10,
		legendfontsize = 8,
		size = (700, 400),
		dpi = 300
	)
	
	# Define fixed seeds for reproducibility
	seeds = Base.rand(UInt64, 100_000) # Fixed seeds for consistent convergence behavior
	
	function price_trajectories(traj)
		mc_method_traj = Accessors.@set mc_method.config.trajectories = traj
		return solve(pricing_problem, mc_method_traj).price
	end
	
	traj_range = 1000:5_000:100_000
	prices = price_trajectories.(traj_range)
	
	# Create the convergence plot with improved styling
	plot(
		traj_range, 
		prices, 
		label="Monte Carlo Price",
		linecolor=:royalblue,
		marker=:circle,
		markersize=4,
		markerstrokewidth=0,
		markeralpha=0.7,
		markercolor=:royalblue
	)
	plot!(
		traj_range, 
		fill(analytic_solution, length(traj_range)), 
		label="Analytic Price",
		linecolor=:firebrick,
		linestyle=:dash
	)
	
	# Add annotations and enhance the plot
	title!("Monte Carlo Price Convergence")
	xlabel!("Number of Trajectories")
	ylabel!("Option Price")
	annotate!(
		traj_range[end], 
		analytic_solution + 0.8, 
		text("Analytic price: $(round(analytic_solution, digits=2))", 8, :right, :firebrick)
	)
end

# ╔═╡ 6de035ac-4656-433e-a69a-f40d05068bf5
mc_solution = solve(pricing_problem, mc_method)

# ╔═╡ 404bf520-88e0-4b33-be30-a826bea38f83
mc_solution.price

# ╔═╡ 2c458a8b-07be-4651-80b1-dd3781c75f5c
md" ## Convergence Plot
Let's visualize how the Monte Carlo price converges to the analytic price as we increase the number of simulated trajectories. This helps us understand the trade-off between computational effort and pricing accuracy."

# ╔═╡ ec594403-dbd0-4995-84a2-13ef2e5ea6e4
md" ## Variance Reduction: Antithetic Variates
Antithetic variates is a powerful variance reduction technique that works by generating pairs of negatively correlated paths. For each random path, we generate its 'mirror image' by flipping the signs of all random increments. Since the option payoff is typically a non-linear function, averaging these paired results reduces variance."

# ╔═╡ d127000b-84c7-4958-8970-1a6c6199f342
begin
	config_anti = SimulationConfig(10_000, steps=100, variance_reduction=Hedgehog.Antithetic()) # Configure with more steps to show path behavior
	mc_method_anti = MonteCarlo(LognormalDynamics(), BlackScholesExact(), config_anti)
	mc_solution_anti = solve(pricing_problem, mc_method_anti)
end

# ╔═╡ 2268a366-bec9-40d2-8365-458f3d751833
begin
	# Extract and plot an original path and its antithetic counterpart
	mc_path = mc_solution_anti.ensemble[1][12] # Regular path 
	mc_path_anti = mc_solution_anti.ensemble[2][12] # Antithetic path
	
	# Create a two-panel plot to show both price and log-price
	p1 = plot(
		mc_path.t, 
		mc_path.u, 
		label="Original Path",
		linecolor=:darkblue,
		linewidth=2,
		title="Price Space",
		ylabel="Asset Price"
	)
	plot!(
		p1,
		mc_path_anti.t, 
		mc_path_anti.u, 
		label="Antithetic Path",
		linecolor=:darkred,
		linewidth=2, 
		linestyle=:dash
	)
	
	# Add risk-free growth line for reference
	t_range = mc_path.t
	risk_free_growth = spot .* exp.(rate .* t_range)
	plot!(
		p1,
		t_range, 
		risk_free_growth, 
		label="Risk-Free Growth",
		linecolor=:forestgreen,
		linewidth=1.5,
		linestyle=:dot
	)
	
	# Second panel: log price space where mirroring is more evident
	p2 = plot(
		mc_path.t, 
		log.(mc_path.u), 
		label="Original Path (log)",
		linecolor=:darkblue,
		linewidth=2,
		title="Log-Price Space",
		ylabel="Log(Asset Price)"
	)
	plot!(
		p2,
		mc_path_anti.t, 
		log.(mc_path_anti.u), 
		label="Antithetic Path (log)",
		linecolor=:darkred,
		linewidth=2, 
		linestyle=:dash
	)
	
	# Add log of risk-free growth (straight line in log space)
	log_risk_free = log.(spot) .+ rate .* t_range
	plot!(
		p2,
		t_range, 
		log_risk_free, 
		label="Risk-Free Growth (log)",
		linecolor=:forestgreen,
		linewidth=1.5,
		linestyle=:dot
	)
	
	# Combine plots and add annotation
	plot(p1, p2, layout=(2,1), size=(800, 600), legend=:topright)
	annotate!(
		0.5, log(70), 
		text("In log-space, paths are mirrored around\nthe risk-free growth line", 8, :center),
		subplot=2
	)
	xlabel!("Time (years)")
end

# ╔═╡ 5392362f-c8ba-43ed-a9c7-8c58dc1480ba
md"The graphs above illustrate how the antithetic variates technique works:

**Top Panel (Price Space)**: The original and antithetic paths appear to follow different trajectories, but with a subtle relationship.

**Bottom Panel (Log-Price Space)**: Here we can see what's really happening - the paths are perfect mirrors of each other around the risk-free rate line. This is because antithetic sampling works by flipping the signs of the Brownian motion increments, which affects the log-price directly.

In the Black-Scholes model, the stock price follows:
```math
S_T = S_0 e^{(r - \frac{\sigma^2}{2})T + \sigma W_T}
```

When we generate the antithetic path, we replace $W_T$ with $-W_T$, resulting in a path that is symmetric in log-space.

This negative correlation between paths is extremely effective for variance reduction - when one path gives a high payoff, the other tends to give a low payoff. This reduces the variance of the price estimator without introducing bias, resulting in more precise estimates for the same computational cost."

# ╔═╡ 2b8b6e12-7035-4c4c-8df6-e720f7249e27
md"## Monte Carlo Sensitivities (Greeks)
Let's calculate the sensitivities of our option price to changes in the market parameters. We'll use pathwise differentiation through automatic differentiation with ForwardDiff.jl.

This is a powerful approach because:
1. It produces accurate Greeks without the noise of finite differences
2. It reuses paths across different Greeks for efficiency
3. It works well for complex models where analytic formulas aren't available

We'll compute vega (sensitivity to volatility) and compare with the analytic solution."

# ╔═╡ 607e775b-3576-41d5-8168-635093e4e1d0
begin
	# Function to calculate Greeks with different trajectory counts
	function greek_trajectories(traj, lens, method)
		mc_method_traj = Accessors.@set mc_method.config.trajectories = traj
		greek_problem = GreekProblem(pricing_problem, lens)
		return solve(greek_problem, method, mc_method_traj).greek
	end
	
	# Calculate analytic vega as benchmark
	analytic_vega = solve(GreekProblem(pricing_problem, VolLens(1,1)), AnalyticGreek(), analytic_method).greek
	
	# Calculate vega using automatic differentiation with various trajectory counts
	vegas_ad = [greek_trajectories(traj, VolLens(1,1), ForwardAD()) for traj in traj_range]
	
	# Create an enhanced plot showing convergence of vega
	plot(
		traj_range, 
		vegas_ad, 
		label="Monte Carlo Vega",
		linecolor=:purple,
		marker=:diamond,
		markersize=4,
		markerstrokewidth=0,
		markeralpha=0.7,
		markercolor=:purple
	)
	plot!(
		traj_range, 
		fill(analytic_vega, length(traj_range)), 
		label="Analytic Vega",
		linecolor=:orange,
		linestyle=:dash,
		linewidth=2
	)
	
	# Add annotations and enhance the plot
	title!("Monte Carlo Vega Convergence")
	xlabel!("Number of Trajectories")
	ylabel!("Vega (∂Price/∂σ)")
	annotate!(
		traj_range[end], 
		analytic_vega + 0.8, 
		text("Analytic vega: $(round(analytic_vega, digits=2))", 8, :right, :orange)
	)
end

# ╔═╡ Cell order:
# ╠═f091db10-24d5-11f0-1c87-db62c36b849a
# ╟─385323b8-78f6-4eb9-91f0-4e18a38abf3c
# ╟─715c0441-2d66-4806-871c-a9624f461a23
# ╠═d48916b3-36df-439a-8975-fb96d1f89d85
# ╟─9fa72ea4-c8c8-4676-9391-01efe61921aa
# ╠═58a53919-f63a-4345-adfc-8d571b2ec6ec
# ╟─4a547637-2652-474a-a615-8a4c992fe0ee
# ╠═a7b471c1-ff78-4562-a3e4-c5294d16928c
# ╟─0e3e4511-4a3a-4e7f-86fd-e77c5697d921
# ╠═95309345-9d8f-47c7-88a5-3023a515b016
# ╟─d47790fb-82f4-4910-bc84-bc179af2e63c
# ╠═6f603b05-f72a-40a6-8b55-49fc4cc2e48c
# ╠═6e8b47b7-6a4d-40fa-bddd-e747cf9d2edc
# ╠═6de035ac-4656-433e-a69a-f40d05068bf5
# ╠═404bf520-88e0-4b33-be30-a826bea38f83
# ╟─2c458a8b-07be-4651-80b1-dd3781c75f5c
# ╠═6df778b3-9085-44ba-bb7f-443b0218a3ff
# ╟─ec594403-dbd0-4995-84a2-13ef2e5ea6e4
# ╠═d127000b-84c7-4958-8970-1a6c6199f342
# ╟─2268a366-bec9-40d2-8365-458f3d751833
# ╟─5392362f-c8ba-43ed-a9c7-8c58dc1480ba
# ╟─2b8b6e12-7035-4c4c-8df6-e720f7249e27
# ╠═607e775b-3576-41d5-8168-635093e4e1d0
