using Distributions, DifferentialEquations, Random, StaticArrays

"""
    HestonProcess(t0, heston_dist)

Constructs a `NoiseProcess` for exact Heston model sampling.

- `t0`: Initial time
- `heston_dist`: A `HestonDistribution` object containing model parameters.
"""
function HestonNoise(t0, heston_dist::HestonDistribution, Z0=nothing)
    log_S0 = log(heston_dist.S0)  # Work in log-space
    u0 = SVector(log_S0, heston_dist.V0)  # Use SVector for efficiency

    # Correct function signature for WHITE_NOISE_DIST
    @inline function Heston_dist(DW, W, dt, u, p, t, rng) #dist
        return rand(rng, heston_dist)  # Calls exact Heston sampler
    end

    return NoiseProcess{false}(t0, log_S0, Z0, Heston_dist, (dW, W, W0, Wh, q, h, u, p, t, rng) -> 1)
end
