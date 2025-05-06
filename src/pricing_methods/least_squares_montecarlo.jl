"""
    LSM <: AbstractPricingMethod

Least Squares Monte Carlo (LSM) pricing method for American options.

Uses regression to estimate continuation values and determine optimal stopping times.

# Fields
- `mc_method`: A `MonteCarlo` method specifying dynamics and simulation strategy.
- `degree`: Degree of the polynomial basis for regression.
"""
struct LSM{M<:MonteCarlo} <: AbstractPricingMethod
    mc_method::M
    degree::Int  # degree of polynomial basis
end

"""
    LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, config::SimulationConfig, degree::Int)

Constructs an `LSM` pricing method with polynomial regression and Monte Carlo simulation.

# Arguments
- `dynamics`: Price dynamics.
- `strategy`: Simulation strategy.
- `config`: Monte Carlo configuration.
- `degree`: Degree of polynomial regression.

# Returns
- An `LSM` instance.
"""
function LSM(dynamics::PriceDynamics, strategy::SimulationStrategy, config::SimulationConfig, degree::Int)
    mc = MonteCarlo(dynamics, strategy, config)
    return LSM(mc, degree)
end

"""
    extract_spot_grid(sol::EnsembleSolution)

Extracts the simulated spot paths from an `EnsembleSolution`.

# Arguments
- `sol`: The ensemble simulation result.

# Returns
- A matrix of spot values of size `(nsteps, npaths)`.
"""
function extract_spot_grid(sol::EnsembleSolution)
    npaths = length(sol.u)
    nsteps = length(sol.u[1].t)
    spot_grid = Matrix{eltype(sol.u[1].u[1][1])}(undef, nsteps, npaths)

    @inbounds for j in 1:npaths
        @views spot_grid[:, j] = getindex.(sol.u[j].u, 1)
    end

    return spot_grid
end

"""
    extract_spot_grid(sol_anti::Tuple{EnsembleSolution, EnsembleSolution})

Extracts spot paths from a pair of ensemble simulations with antithetic variates.

# Arguments
- `sol_anti`: Tuple of original and antithetic ensemble simulations.

# Returns
- A matrix of spot values of size `(nsteps, 2 * npaths)`.
"""
function extract_spot_grid(sol_anti::Tuple{EnsembleSolution,EnsembleSolution})
    sol, antithetic = sol_anti
    npaths = length(sol.u)
    nsteps = length(sol.u[1].t)
    spot_grid = Matrix{eltype(sol.u[1].u[1][1])}(undef, nsteps, 2 * npaths)

    @inbounds for j in 1:npaths
        @views spot_grid[:, j] = getindex.(sol.u[j].u, 1)
    end

    @inbounds for k in (npaths + 1):(2 * npaths)
        @views spot_grid[:, k] = getindex.(antithetic.u[k - npaths].u, 1)
    end

    return spot_grid
end

""" 
    Get the spot grid from a simulation by first making an SDE problem and then simulating paths. Returns the extracted spot grid and a boolean indicating if the spot grid matrix has been transposed. 
"""
function get_spot_grid(prob, method)
    sde_prob = sde_problem(prob, method.mc_method.dynamics, method.mc_method.strategy)
    sol = Hedgehog.simulate_paths(sde_prob, method.mc_method, method.mc_method.config.variance_reduction)
    return Hedgehog.extract_spot_grid(sol)
end

"""
    solve(prob::PricingProblem, method::LSM)

Prices an American option using the Least Squares Monte Carlo method.

# Arguments
- `prob`: A `PricingProblem` containing an American `VanillaOption`.
- `method`: An `LSM` pricing method.

# Returns
- An `LSMSolution` containing price and stopping strategy.
"""
function solve(
    prob::PricingProblem{VanillaOption{TS,TE,American,C,S},I},
    method::L,
) where {TS,TE,I<:AbstractMarketInputs,C,S, L<:LSM}

    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    sde_prob = sde_problem(prob, method.mc_method.dynamics, method.mc_method.strategy)
    sol = Hedgehog.simulate_paths(sde_prob, method.mc_method, method.mc_method.config.variance_reduction)
    spot_grid = Hedgehog.extract_spot_grid(sol)
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return LSMSolution(prob, method, price, stopping_info, spot_grid)
end

""" This works the same as solve(), but allows for faster implementations for special cases and uses a spot-grid that is performant for column-major memory access. """
function solve_quick(
    prob::PricingProblem{VanillaOption{TS,TE,American,C,S},I},
    method::L,
) where {TS,TE,I<:AbstractMarketInputs,C,S, L<:LSM}

    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    spot_grid = get_spot_grid(prob, method)

    npaths, ntimes = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[p, nsteps+1])) for p = 1:npaths]

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[:, i])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[in_the_money, i]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return LSMSolution(prob, method, price, stopping_info, spot_grid)
end

"""
    update_stopping_info!(
        stopping_info::Vector{Tuple{Int, U}},
        paths::Vector{Int},
        cont_value::Vector{T},
        payoff_t::Vector{S},
        t::Int
    )

Updates the stopping times and values in-place based on immediate vs. continuation value comparison.

# Arguments
- `stopping_info`: Current best (time, value) for each path.
- `paths`: Indices of in-the-money paths.
- `cont_value`: Estimated continuation values.
- `payoff_t`: Immediate exercise values.
- `t`: Current time index.
"""
function update_stopping_info!(
    stopping_info::Vector{Tuple{Int,U}},
    paths::Vector{Int},
    cont_value::Vector{T},
    payoff_t::Vector{S},
    t::Int,
) where {T,S,U}
    exercise = payoff_t[paths] .> cont_value
    stopping_info[paths[exercise]] .= [(t, payoff_t[p]) for p in paths[exercise]]
end

function get_spot_grid(
    prob::PricingProblem{V, <:BlackScholesInputs},
    method::L,
) where {V, L<:LSM}
    market = prob.market_inputs
    K = method.mc_method.config.steps
    var_red = method.mc_method.config.variance_reduction
    T = yearfrac(market.referenceDate, prob.payoff.expiry)

    r = zero_rate(market.rate, 0.0)
    σ = get_vol(market.sigma, nothing, nothing)
    S₀ = market.spot
    Δt = T/K
    N = method.mc_method.config.trajectories

    return simulate_paths_black_scholes(r, σ, S₀, Δt, N, K, var_red, method.mc_method.config.seeds)
end

function simulate_paths_black_scholes(r, σ, S₀, Δt, N, K, ::NoVarianceReduction, seed)
    rng = Xoshiro(seed[1])
    drift_part = ((r - 0.5 * σ^2) * Δt)
    diffusion_part = σ * sqrt(Δt)
    paths = Matrix{typeof(drift_part*diffusion_part*S₀)}(undef, N, K)
    randn!(rng, paths)
    @views for k in 1:K
        if k == 1
            paths[:, 1] = S₀ .* exp.(drift_part .+ diffusion_part .* paths[:,1])
            continue
        end
        paths[:, k] = paths[:, k - 1] .* exp.(drift_part .+ diffusion_part .* view(paths, :, k))
    end
    return paths
end
function simulate_paths_black_scholes(r, σ, S₀, Δt, N, K, ::Antithetic, seed)
    rng = Xoshiro(seed[1])
    drift_part = ((r - 0.5 * σ^2) * Δt)
    diffusion_part = σ * sqrt(Δt)
    paths = Matrix{typeof(drift_part*diffusion_part*S₀)}(undef, 2*N, K)
    randn!(rng, view(paths, 1:N, :))
    paths[N+1:2*N, :] .= -paths[1:N, :]
    @views for k in 1:K
        if k == 1
            paths[:, 1] = S₀ .* exp.(drift_part .+ diffusion_part .* paths[:,1])
            continue
        end
        paths[:, k] = paths[:, k - 1] .* exp.(drift_part .+ diffusion_part .* view(paths, :, k))
    end
    return paths
end