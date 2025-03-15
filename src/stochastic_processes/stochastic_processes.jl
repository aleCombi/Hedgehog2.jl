using DifferentialEquations

export AbstractStochasticProcess, GBMProcess, HestonProcess, get_drift, get_diffusion, get_analytic_solution, get_sde_function

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

# Drift function for GBM
"""
    drift(process::GBMProcess, u, t)

Computes the drift term of the GBM process at time `t` for state `u`.
Drift equation: `du/dt = μ * u`
"""
function drift(process::GBMProcess, u, t)
    return process.μ .* u  # Element-wise broadcasting for array compatibility
end

# Drift function for Heston
"""
    drift(process::HestonProcess, u, t)

Computes the drift term of the Heston process at time `t` for state `u = (S, V)`.
Drift equations:
- `dS/dt = μ * S`
- `dV/dt = κ * (θ - V)`
"""
function drift(process::HestonProcess, u, t)
    S, V = u
    return [process.μ * S, process.κ * (process.θ - V)]  # Drift for (Stock price, Variance)
end

# Diffusion function for GBM
"""
    diffusion(process::GBMProcess, u, t)

Computes the diffusion term of the GBM process at time `t` for state `u`.
Diffusion equation: `dW_t = σ * u * dB_t`
"""
function diffusion(process::GBMProcess, u, t)
    return process.σ .* u  # Element-wise broadcasting for array compatibility
end

# Diffusion function for Heston
"""
    diffusion(process::HestonProcess, u, t)

Computes the diffusion term of the Heston process at time `t` for state `u = (S, V)`.
Diffusion equations:
- `dS_t = σ * sqrt(V) * dB_t`
- `dV_t = ξ * sqrt(V) * dW_t`, with correlation `ρ`.
"""
function diffusion(process::HestonProcess, u, t)
    S, V = u
    return [process.σ * S * sqrt(V), process.σ * sqrt(V)]  # Diffusion for (Stock price, Variance)
end

# Higher-order functions to return drift! and diffusion!
"""
    get_drift!(process::AbstractStochasticProcess)

Returns the in-place drift function `drift!` for the given process.
"""
function get_drift(process::P) where P<:AbstractStochasticProcess
    return (u, p, t) -> drift(process, u, t)
end

"""
    get_diffusion!(process::AbstractStochasticProcess)

Returns the in-place diffusion function `diffusion!` for the given process.
"""
function get_diffusion(process::P) where P<:AbstractStochasticProcess
    return (u, p, t) -> diffusion(process, u, t)
end

function get_analytic_solution(process::P) where P <: AbstractStochasticProcess
    return Nothing()
end

function get_analytic_solution(::GBMProcess)
    return (u₀, p, t, W) -> u₀ * exp((0.05 - (0.2^2) / 2) * t + 0.2 * W)
end

function get_sde_function(process::P) where P<:AbstractStochasticProcess
    return SDEFunction(get_drift(process), get_diffusion(process), analytic=get_analytic_solution(process))
end

