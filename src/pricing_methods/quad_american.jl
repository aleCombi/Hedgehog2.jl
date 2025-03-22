export QUAD

struct QUAD{D} <: AbstractPricingMethod
    boundary::Float64        # half-width of the log-price domain
    n_grid::Int              # number of intervals → n_grid + 1 points
    n_exercise::Int          # number of exercise dates
    dynamics::D
end

QUAD(boundary::Real, n_grid::Int, n_exercise::Int, dynamics::D) where {D} =
    QUAD{D}(boundary, n_grid, n_exercise, dynamics)

function transition_density(::LognormalDynamics, mkt::BlackScholesInputs, log_diff::Matrix, Δt::Float64)
    μ = (mkt.rate - 0.5 * mkt.sigma^2) * Δt
    σ = mkt.sigma * √Δt
    return pdf.(Normal(μ, σ), log_diff)
end

function compute_price(
    payoff::VanillaOption{American, Put, Spot},
    market_inputs::BlackScholesInputs,
    method::QUAD{LognormalDynamics}
)
    T = Dates.value(payoff.expiry - market_inputs.referenceDate) / 365
    dt = T / method.n_exercise
    r = market_inputs.rate

    # Compute log-price grid using n_grid
    step = 2 * method.boundary / method.n_grid
    log_asset = range(-method.boundary, method.boundary; length=method.n_grid + 1)
    n = length(log_asset)

    log_diff = @. log_asset - log_asset'
    F = transition_density(method.dynamics, market_inputs, log_diff, dt)

    # Spot grid and payoff at maturity
    spot_grid = market_inputs.spot .* exp.(log_asset)
    V = payoff.(spot_grid)

    # Precompute edge rows for trapezoidal correction
    f1 = F[:, 1]
    fN = F[:, end]

    for _ in 1:method.n_exercise
        v1, vN = V[1], V[end]
        continuation = exp(-r * dt) * step * (F * V .- 0.5 * (v1 .* f1 + vN .* fN))
        exercise = payoff.(spot_grid)
        V = max.(continuation, exercise)
    end

    idx = findfirst(x -> isapprox(x, 0.0; atol=1e-10), log_asset)
    return V[idx]
end
