using BenchmarkTools

abstract type AbstractPayoff end

# Abstract version
struct VanillaAbstract <: AbstractPayoff
    strike::Real
    expiry::Real
end

price(opt::VanillaAbstract) = opt.strike + 1.0


# Concrete parametric version
struct VanillaConcrete{T<:Real} <: AbstractPayoff
    strike::T
    expiry::T
end

price(opt::VanillaConcrete) = opt.strike + 1.0


# Instances
a = VanillaAbstract(100.0, 1.0)
b = VanillaConcrete(100.0, 1.0)

# Benchmarks
@btime price($b)   # Concrete field
@btime price($a)   # Abstract field
