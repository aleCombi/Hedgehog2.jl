using Distributions, DifferentialEquations, Random, StaticArrays

"""
    HestonProcess(t0, heston_dist)

Constructs a `NoiseProcess` for exact Heston model sampling.

- `t0`: Initial time
- `heston_dist`: A function from time to `HestonDistribution` object containing model parameters.
"""
#TODO: not entirely correct, as it works only in the one-step simulation scenario. Otherwise it would need to be 2D
function HestonNoise(t0, heston_dist, Z0=nothing; kwargs...)
    log_S0 = log(heston_dist(t0).S0)  # Work in log-space
    @inline function Heston_dist(DW, W, dt, u, p, t, rng) #dist
        heston_dist_at_t = heston_dist(dt)
        return @fastmath rand(rng, heston_dist_at_t; kwargs...)  # Calls exact Heston sampler
    end

    return NoiseProcess{false}(t0, log_S0, Z0, Heston_dist, (dW, W, W0, Wh, q, h, u, p, t, rng) -> 1)
end