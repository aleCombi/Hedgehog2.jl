
abstract type AbstractPricingSolution end

struct MonteCarloSolution{S, T<:Number} <: AbstractPricingSolution
    price::T
    ensemble::S
end

struct AnalyticSolution{T <: Number} <: AbstractPricingSolution
    price::T
end

struct CarrMadanSolution{T <: Number} <: AbstractPricingSolution
    price::T
    integral_solution::SciMLBase.IntegralSolution
end

struct LSMSolution{T,S,M} <: AbstractPricingSolution
    price::T
    stopping_info::Vector{Tuple{Int,S}}
    spot_paths::Matrix{M}
end

struct CRRSolution{T} <: AbstractPricingSolution
    price::T
end