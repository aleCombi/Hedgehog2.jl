abstract type AbstractPricingSolution end

struct MonteCarloSolution <: AbstractPricingSolution
    price
    ensemble::EnsembleSolution
end

struct AnalyticSolution <: AbstractPricingSolution
    price
end
