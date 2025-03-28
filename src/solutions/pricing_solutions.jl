
abstract type AbstractPricingSolution end

struct MonteCarloSolution <: AbstractPricingSolution
    price
    ensemble::EnsembleSolution
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
