import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^

export tuplef2ftuple
export Discontinuous

"""
     Discontinuous

 A type that contains a discontinuous function and an index of any of its discontinuities. This allows the construction of step parameter profiles that retain discontinuity information. You can also call the `Discontinuous`` type like a normal function, if you want to.
"""
struct Discontinuous <: Function
    f::Function
    d::Set
end

arithmetics = (:+, :-, :*, :/, :^)
for a ∈ arithmetics
    eval(quote
        ($a)(c::Real, D1::Discontinuous) = Discontinuous(x -> ($a)(c, D1.f(x)), D1.d)
        ($a)(D1::Discontinuous, c::Real, ) = Discontinuous(x -> ($a)(D1.f(x), c), D1.d)
        ($a)(D1::Discontinuous, D2::Discontinuous) = Discontinuous((x...) -> ($a)(D1.f(x...), D2.f(x...)), union(D1.d, D2.d))
        ($a)(f::Function, D::Discontinuous) = Discontinuous((x...) -> ($a)(f(x...), D.f(x...)), D.d)
        ($a)(D::Discontinuous, f::Function) = Discontinuous((x...) -> ($a)(D.f(x...), f(x...)), D.d)
    end)
end


# Overload call so that you can use a Discontinuous like a normal (vectorised) function, if you want
(D::Discontinuous)(x::Real) = D.f(x)
(D::Discontinuous)(x::Union{Array, StepRange, StepRangeLen, UnitRange}) = D.f.(x)


# ! To use stiff solvers that error about instability with discontinuous profiles, try e.g. Rosenbrock23(autodiff = false) and turning off adaptive timestepping

"""
    tuplef2ftuple(f::Tuple{Function}, params::Tuple)
Convert a vector of curried functions `f` that each accept an element of the parameter tuple `params` into a function of the form `f(t) -> x::Vector`, where `xᵢ = fᵢ(paramsᵢ)(t)` for `i = 1:length(f)`
"""
function tuplef2ftuple(f, params)
    # turn a tuple of functions into a function of tuples
    if all(isempty.(params)) # The f's are just functions on their own, no need to add parameters
        # Be warned that you can't mix these; either use all parameter functions, or all standard functions. Don't be greedy.
        if f isa Tuple{<:Function} || f isa Vector{<:Function}
            ds = [fi isa Discontinuous ? fi.d : [] for fi in f]
            ds = reduce(∪, ds) |> Set
            pp = Discontinuous(t->[x(t) for x in f], ds)
        else
            pp = f
        end
        return pp
    elseif eltype(f) <: Function
        ps = Vector{Function}(undef, length(f))
        for i = 1:length(f)
            ps[i] = f[i](params[i]...)
        end
        ds = [fi isa Discontinuous ? fi.d : [] for fi in ps]
        ds = reduce(∪, ds) |> Set
        p = Discontinuous(t->map((x, g) -> g(x), fill(t, length(ps)), ps), ds) # Something like that
    else
        p = f(params...)
    end
    return p
end
