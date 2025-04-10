using Accessors, ForwardDiff

struct Input{T,S}
    first::T
    second::S
end

@generated function get_lenses(lenses::T, input) where {T<:Tuple}
    N = length(T.parameters)
    body = [:(lenses[$i](input)) for i in 1:N]
    return quote
        collect(($(body...),))
    end
end

@generated function set_lenses(lenses::T, input, values::AbstractVector) where {T<:Tuple}
    N = length(T.parameters)
    ex = :(input)
    for i in 1:N
        ex = :(set($ex, lenses[$i], values[$i]))
    end
    return ex
end

function sum_inputs(inputs::I) where I <: Input
    return inputs.first * inputs.second
end

first_lens = @optic _.first
second_lens =  @optic _.second
lenses = (first_lens, second_lens)
my_input = Input(1.2, 4.5)
get_lenses(lenses, my_input)
set_lenses(lenses, my_input, [2.0,2.4])

make_func(lenses, input, sum_inputs) = x -> sum_inputs(set_lenses(lenses, input, x))
my_func_new = make_func(lenses, my_input, sum_inputs)

x0 = get_lenses(lenses, my_input)

@code_warntype my_func_new(x0)
@code_warntype ForwardDiff.gradient(my_func_new, x0)

ForwardDiff.gradient((x)->x[1]*x[2], x0)

@btime ForwardDiff.derivative((x)->x*x0[2], x0[1])