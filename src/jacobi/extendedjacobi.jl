"""
ExtendedJacobi{kind,T}()

is a quasi-matrix representing extended Jacobi functions on ℝ. 

For an untransformed ExtendedJacobi, for x∈[-1,1]  ExtendedJacobi()[x, n] = Jacobi()[x, n]
and outside the interval x∉[-1,1], there is an explicit formula. 
"""

#### 
# ExtendedJacobi
####

struct ExtendedJacobi{T} <: Basis{T}
    a::T
    b::T
    ExtendedJacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

ExtendedJacobi(a::T, b::T) where T = ExtendedJacobi{Float64}(a::T, b::T)

axes(P::ExtendedJacobi) = (Inclusion(ℝ), OneToInf())

==(P::ExtendedJacobi, Q::ExtendedJacobi) = P.a == Q.a && P.b == Q.b

function getindex(P::ExtendedJacobi{T}, x::Real, j::Int)::T where T
    -1 <= x <= 1 && return Jacobi{T}(P.a, P.b)[x,j]
    return error("Not implemented for |x|>1 yet.")
end

#### 
# ExtendedWeightedJacobi
####

struct ExtendedWeightedJacobi{T} <: Basis{T}
    a::T
    b::T
    ExtendedWeightedJacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end


ExtendedWeightedJacobi(a::T, b::T) where T = ExtendedWeightedJacobi{Float64}(a::T, b::T)

axes(H::ExtendedWeightedJacobi) = (Inclusion(ℝ), OneToInf())

==(P::ExtendedWeightedJacobi, Q::ExtendedWeightedJacobi) = P.a == Q.a && P.b == Q.b

function getindex(P::ExtendedWeightedJacobi{T}, x::Real, j::Int)::T where T
    -1 <= x <= 1 && return Weighted(Jacobi{T}(P.a, P.b))[x,j]
    return 0.
end