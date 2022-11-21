"""
SumSpace{T}()

is a quasi-matrix representing a sum space on ℝ. 
"""
struct SumSpace{T, E} <: Basis{T}
    P
    Q
    I::E
end

SumSpace{T}(P, Q, I::AbstractVector=[-1.,1.]) where T = SumSpace{T, typeof(I)}(P, Q, I)
SumSpace(P, Q) = SumSpace{Float64}(P, Q)


axes(S::SumSpace) = (Inclusion(ℝ), _BlockedUnitRange(2:2:∞))
==(S::SumSpace, Z::SumSpace) = S.P == Z.P && S.Q == Z.Q && S.I == Z.I

function getindex(S::SumSpace{T}, x::Real, j::Int)::T where T
    el_no = length(S.I)-1
    i = ((j-1) ÷ (2*el_no)) + 1     # Poly/function order
    el = ((j-1) ÷ 2) % el_no + 1    # Element number

    y = affinetransform(S.I[el],S.I[el+1], x)
    isodd(j) && return S.P[y, i]
    S.Q[y, i]
end
