"""
SumSpaceJacobi{kind,T}()

is a quasi-matrix representing the Jacobi sum space of the specified kind (1 or 2)
on (-∞,∞).
"""
###
# Primal and dual Jacobi sum space
###

struct SumSpaceJacobi{kind,T,E} <: Basis{T} 
    a::T
    b::T
    I::E
    SumSpaceJacobi{kind, T, E}(a, b, I) where {kind, T, E} = new{kind, T, E}(convert(T,a), convert(T,b), convert(E,I))
end
SumSpaceJacobi{kind, T}(a::T, b::T, I::AbstractVector=[-1.,1.]) where {kind, T} = SumSpaceJacobi{kind, T, typeof(I)}(a::T, b::T, I)
SumSpaceJacobi{kind}(a::Float64, b::Float64, I::AbstractVector=[-1.,1.]) where kind = SumSpaceJacobi{kind, Float64}(a, b, I)

const SumSpaceJacobiP = SumSpaceJacobi{1}
const SumSpaceJacobiD = SumSpaceJacobi{2}

axes(S::SumSpaceJacobi) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

==(P::SumSpaceJacobi{kind}, Q::SumSpaceJacobi{kind}) where kind = P.I == Q.I && P.a == Q.a && P.b == Q.b
==(P::SumSpaceJacobi, Q::SumSpaceJacobi) = false

function getindex(S::SumSpaceJacobiP{T}, x::Real, j::Int)::T where T
    if S.a != S.b || S.a < 0
        error("Currently can only implement s = P.a = P.b, s≥0")
    end
    y = affinetransform(S.I[1], S.I[2], x)
    isodd(j) && return ExtendedJacobi{T}(-S.a, -S.b)[y, (j ÷ 2)+1]
    ExtendedWeightedJacobi{T}(S.a, S.b)[y, j ÷ 2]
end

function getindex(S::SumSpaceJacobiD{T}, x::Real, j::Int)::T where T
    if S.a != S.b || S.a < 0
        error("Currently can only implement s = P.a = P.b, s≥0")
    end
    y = affinetransform(S.I[1],S.I[2], x)
    isodd(j) && return ExtendedJacobi{T}(S.a, S.b)[y, (j ÷ 2)+1]
    ExtendedWeightedJacobi{T}(-S.a, -S.b)[y, j ÷ 2]
end