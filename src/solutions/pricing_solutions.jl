
abstract type AbstractPricingSolution end

struct MonteCarloSolution <: AbstractPricingSolution
    price
    ensemble::EnsembleSolution
end

struct AnalyticSolution <: AbstractPricingSolution
    price
end

struct CarrMadanSolution <: AbstractPricingSolution
    price::Float64
    integral_solution::SciMLBase.IntegralSolution
end

struct LSMSolution{T, S, M} <: AbstractPricingSolution
    price::T
    stopping_info::Vector{Tuple{Int, S}}
    spot_paths::Matrix{M}
end

