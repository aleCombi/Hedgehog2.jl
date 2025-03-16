using DifferentialEquations

export AbstractStochasticProcess, GBMProcess, HestonProcess, get_sde_function, combine_sde_functions, GBMTimeDependent, get_sde_problem

function combine_sde_functions(sde_functions::Tuple{Vararg{SDEFunction}})
    n = length(sde_functions)  # Number of 1D SDEs

    # Construct multi-dimensional drift function
    f = (u, p, t) -> begin
        return map(i -> sde_functions[i].f([u[i]], p, t)[1], 1:n)
    end

    # Construct multi-dimensional diffusion function
    g = (u, p, t) -> begin
        return map(i -> sde_functions[i].g([u[i]], p, t)[1], 1:n)
    end

    # Check if all SDEFunctions have an analytic solution
    if all(sf -> sf.analytic !== nothing, sde_functions)
        # Construct a multi-dimensional analytical solution
        analytic = (u0, p, t, W) -> begin
            return map(i -> sde_functions[i].analytic([u0[i]], p, t, W[i])[1], 1:n)
        end
    else
        analytic = nothing  # No combined analytic solution if any function lacks it
    end

    return SDEFunction(f, g, analytic=analytic)
end

function get_sde_problem(processes, Γ, u0, tspan, p; kwargs...)
    sde_functions = get_sde_function.(processes)
    combined_sde_function = combine_sde_functions(sde_functions)
    noise = CorrelatedWienerProcess(Γ, tspan[1], zeros(length(processes)))

    return SDEProblem(combined_sde_function, u0, tspan, p; noise=noise, kwargs...)
end


# Abstract type for stochastic processes
"""
    AbstractStochasticProcess

An abstract type representing a generic stochastic process.
"""
abstract type AbstractStochasticProcess end

# Define the GBM process
"""
    GBMProcess <: AbstractStochasticProcess

Represents a Geometric Brownian Motion (GBM) process with drift `μ` and volatility `σ`.
"""
struct GBMProcess <: AbstractStochasticProcess
    μ
    σ
end

function get_sde_function(process::GBMProcess)
    drift(u, p, t) = process.μ .* u 

    diffusion(u, p, t) = process.σ .* u

    analytic_solution(u₀, p, t, W) = u₀ .* exp.((process.μ - (process.σ^2) / 2) .* t .+ process.σ * W)

    return SDEFunction(drift, diffusion, analytic = analytic_solution)
end

# Define the Heston process
"""
    HestonProcess <: AbstractStochasticProcess

Represents the Heston stochastic volatility model with parameters:
- `μ`: Drift of the asset price
- `κ`: Mean reversion speed of variance
- `θ`: Long-run variance
- `σ`: Volatility of variance
- `ρ`: Correlation between asset and variance processes
"""
struct HestonProcess <: AbstractStochasticProcess
    μ
    κ
    θ
    σ
    ρ
end

function get_sde_function(process::HestonProcess)
    function drift(u, p, t)
        S, V = u
        return [process.μ * S, process.κ * (process.θ - V)] 
    end
    
    function diffusion(u, p, t)
        S, V = u
        return [process.σ * S * sqrt(V), process.σ * sqrt(V)]  
    end

    return SDEFunction(drift, diffusion)
end

struct GBMTimeDependent
    μ
    σ
    mean_σ²
end

function get_sde_function(process::GBMTimeDependent)
    drift(u, p, t) = process.μ .* u

    diffusion(u, p, t) = sqrt(process.mean_σ²(t)) .* u

    # observe that the noise here is integrated.
    # That is, W is not such that dX = mu dt + sigma_t dW_t but it is W' such that W'_t = t (int_0^t sigma_s dW_t) / (int_0^t sigma_s^2)
    function analytic_solution(u₀, p, t, W)
        I_t = sqrt(process.mean_σ²(t)) ./ t ./ t
        return u₀ .* exp.((process.μ .- 0.5 * I_t^2) .* t .+ I_t .* W)
    end

    return SDEFunction(drift, diffusion, analytic = analytic_solution)
end

# dX_t = (θ_t - a t) dt + σ_t dW_t
struct OUTimeDependent
    θ
    a
    σ
end
