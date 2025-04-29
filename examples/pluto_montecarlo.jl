### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ f091db10-24d5-11f0-1c87-db62c36b849a
begin
	import Pkg
	Pkg.activate(".")  # activate the current folder (if opened inside `examples/`)
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
md"# 📈 Monte Carlo option pricing in Hedgehog

In this notebook we show how can Monte Carlo methods can be setup in order to price options in Hedgehog. We will show how to choose dynamics, simulation methods and calculate sensitivities.
We will also focus on explicit seeding to obtain reproducibility.
"

# ╔═╡ 715c0441-2d66-4806-871c-a9624f461a23
md"## Payoff definition
We define a european call payoff.
"

# ╔═╡ 9fa72ea4-c8c8-4676-9391-01efe61921aa
md"## Price Dynamics
In order to get a price, we make assumptions on the underlying dynamics. In the Black-Scholes model, the dynamics are the following under the risk-neutral measure:
```math
dS_t = r S_t \, dt + \sigma S_t \, dW_t
```

We need the necessary market inputs, that is the model parameters $r$ and $\sigma$, today's date, the spot price at today.
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
In Hedgehog.jl, a payoff and a market inputs object are the necessary ingredients to formulate a pricing problem, which we can solve in different ways."

# ╔═╡ a7b471c1-ff78-4562-a3e4-c5294d16928c
pricing_problem = PricingProblem(payoff, market_inputs)

# ╔═╡ 0e3e4511-4a3a-4e7f-86fd-e77c5697d921
md"## Monte Carlo setup
We finally setup our Monte Carlo simulation, by defining our simulation scheme, the dynamics, the number of steps and of paths. We can also choose the antithetic variance-reduction scheme.
"

# ╔═╡ 95309345-9d8f-47c7-88a5-3023a515b016
begin
	dynamics = LognormalDynamics() # Black Scholes dynamics
	simulation_strategy = BlackScholesExact() # exact simulation using the law of S_T
	trajectories = 100_000
	steps = 1 # with an exact scheme, no need to take several steps 
	config = SimulationConfig(trajectories, steps=steps, variance_reduction=Hedgehog.Antithetic()) # configurate the simulation choosing  trajectories, steps, variance reduction.
	mc_method = MonteCarlo(dynamics, simulation_strategy, config)
end

# ╔═╡ d47790fb-82f4-4910-bc84-bc179af2e63c
md"## Analytic Benchmark
In order to check Monte Carlo method accuracy, we compare it to the analytic Black Scholes pricer.
"

# ╔═╡ 6f603b05-f72a-40a6-8b55-49fc4cc2e48c
analytic_method = BlackScholesAnalytic()

# ╔═╡ 6e8b47b7-6a4d-40fa-bddd-e747cf9d2edc
analytic_solution = solve(pricing_problem, analytic_method).price

# ╔═╡ 6de035ac-4656-433e-a69a-f40d05068bf5
mc_solution = solve(pricing_problem, mc_method)

# ╔═╡ 404bf520-88e0-4b33-be30-a826bea38f83
mc_solution.price

# ╔═╡ 2c458a8b-07be-4651-80b1-dd3781c75f5c
md" ## Convergence Plot
Let us now visualize the convergence of Monte Carlo price to the analytic price, varying the number of simulated trajectories.
"

# ╔═╡ ec594403-dbd0-4995-84a2-13ef2e5ea6e4
md" ## Additional insights
From the solve method we get the price together with a whole Solution object. When solving with Monte Carlo this returns the whole EnsembleSolution from DifferentialEquations.jl. This allows the user to get additional information. Here we show how the antithetic variance reduction works.
This time we need to take some steps to make the graph more meaningful.
"

# ╔═╡ d127000b-84c7-4958-8970-1a6c6199f342
begin
	config_anti = SimulationConfig(10_000, steps=100, variance_reduction=Hedgehog.Antithetic()) # configurate the simulation choosing  trajectories, steps, variance reduction.
	mc_method_anti = MonteCarlo(LognormalDynamics(), BlackScholesExact(), config_anti)
	mc_solution_anti = solve(pricing_problem, mc_method_anti)
end

# ╔═╡ 2268a366-bec9-40d2-8365-458f3d751833
begin
	using Plots
	mc_path = mc_solution_anti.ensemble[1][12] # regular path 
	plot(mc_path.t, mc_path.u)
	mc_path_anti = mc_solution_anti.ensemble[2][12] # antithetic path
	plot!(mc_path_anti.t, mc_path_anti.u)
end

# ╔═╡ 6df778b3-9085-44ba-bb7f-443b0218a3ff
begin
	using Accessors
	seeds = Base.rand(UInt64, 100_000) # we fix 100k seeds so that convergence is not noisy

	function price_trajectories(traj)
		mc_method_traj = Accessors.@set mc_method.config.trajectories = traj
		return solve(pricing_problem, mc_method_traj).price
	end

	traj_range = 1000:5_000:100_000
	plot(traj_range, price_trajectories.(traj_range))
	plot!(traj_range, fill(analytic_solution, length(traj_range)))
end

# ╔═╡ 5392362f-c8ba-43ed-a9c7-8c58dc1480ba
md"It can be seen easily from the graph above what the antithetic variates technique consist of: the second trajectory is obtained from the first one by flipping the whole Brownian Motion. This makes the solutions negatively correlated, hence reducing the price estimator variance."

# ╔═╡ 2b8b6e12-7035-4c4c-8df6-e720f7249e27
md"## Monte Carlo Sensitivities
We end by showing sensitivities of the price using the Monte Carlo methods. We do so using pathwise differentiation using Automatic Differentiation with ForwardDiff.jl. Alternatively, simple finite difference schemes can be used. Notice that this is justified as a call payoff is almost everywhere differentiable, with the only singular point being at the ATM strike. Again we can compare with the analytic solutions.
"

# ╔═╡ 607e775b-3576-41d5-8168-635093e4e1d0
begin

	function greek_trajectories(traj, lens, method)
		mc_method_traj = Accessors.@set mc_method.config.trajectories = traj
		greek_problem = GreekProblem(pricing_problem, lens)
		return solve(greek_problem, method, mc_method_traj).greek
	end

	analytic_vega = solve(GreekProblem(pricing_problem, VolLens(1,1)), AnalyticGreek(), analytic_method).greek
	vegas_ad = [greek_trajectories(traj, VolLens(1,1), ForwardAD()) for traj in traj_range]
	plot(traj_range, vegas_ad)
	plot!(traj_range, fill(analytic_vega, length(traj_range)))
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
