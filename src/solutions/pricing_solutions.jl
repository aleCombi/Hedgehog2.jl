
export solve_custom_ensemble, CustomEnsembleProblem
abstract type AbstractPricingSolution end


struct CustomEnsembleProblem{P, F}
    base_problem::P
    seeds::Vector{Int64}
    modify::F  # (base_problem, seed, index) -> new_problem
end

struct CustomEnsembleSolution{S}
    solutions::Vector{S}
    seeds::Vector{Int64}
end

function solve_custom_ensemble(prob::CustomEnsembleProblem; dt, solver=EM(), trajectories=nothing)
    N = trajectories === nothing ? length(prob.seeds) : trajectories
    seeds = length(prob.seeds) == N ? prob.seeds : collect(1:N)

    sols = Vector{Any}(undef, N)

    Threads.@threads for i in 1:N
        seed = seeds[i]
        pmod = remake(prob.base_problem; seed=seed)
        pmod = prob.modify(prob.base_problem, seed, i)
        sols[i] = DifferentialEquations.solve(pmod, solver; dt=dt, save_noise=true)
    end

    return CustomEnsembleSolution(sols, seeds)
end

struct MonteCarloSolution{S} <: AbstractPricingSolution
    price
    ensemble::CustomEnsembleSolution{S}
end

struct AnalyticSolution <: AbstractPricingSolution
    price
end

struct CarrMadanSolution <: AbstractPricingSolution
    price
    integral_solution::SciMLBase.IntegralSolution
end

struct LSMSolution{T, S, M} <: AbstractPricingSolution
    price::T
    stopping_info::Vector{Tuple{Int, S}}
    spot_paths::Matrix{M}
end

struct CRRSolution{T} <: AbstractPricingSolution
    price::T
end
