"""
SumSpace{T}()

is a quasi-matrix representing a sum space on ℝ. 
"""
struct SumSpace{T, F, E} <: Basis{T}
    P::F
    I::E
end

# SumSpace(P, I) = SumSpace{Float64, Tuple{Vararg{Basis{Float64}}}, typeof(I)}(P, I::AbstractVector)
SumSpace{T, F}(P, I::AbstractVector=[-1.,1.]) where {T, F} = SumSpace{T, F, typeof(I)}(P, I::AbstractVector)
SumSpace{T}(P, I::AbstractVector=[-1.,1.]) where T = SumSpace{T, Tuple{Vararg{Basis{T}}}, typeof(I)}(P, I::AbstractVector)
SumSpace(P) = SumSpace{Float64}(P)


# axes(S::SumSpace) = (Inclusion(ℝ), _BlockedUnitRange(length(S.P):length(S.P):∞))
function axes(S::SumSpace)  
    try
        return (Inclusion(axes(S.P[1],1)), OneToInf())
    catch
        return (Inclusion(ℝ), OneToInf())
    end
end
==(S::SumSpace, Z::SumSpace) = S.P == Z.P && S.I == Z.I

"""
Let (m,k,n) denote the triple (function number, interval number, polynomial degree). Let M be
the number of functions in the sum space and K the number of intervals.

We order the functions as:

|(1,1,1), ..., (M,1,1)|, (1,2,1), ..., (M,K,1), (1,1,2), ..., (M, K, ∞).

Note that MK(n-1) + M(k-1) + m = j (index number)
"""

function getindex(S::SumSpace{T}, x::Real, j::Int)::T where T
    M = length(S.P)
    K = length(S.I)-1
    n = (j-1) ÷ (M*K) + 1           # polynomial degree
    k = ((j- M*K*(n-1))-1) ÷ M + 1  # interval number
    m = j - M*K*(n-1) - M*(k-1)     # function number

    y = affinetransform(S.I[k],S.I[k+1], x)
    if S.P[m] isa Function
        S.P[m](y)[n]
    else
        S.P[m][y, n]
    end
end

function assemble(S::SumSpace{T}, x::AbstractArray, j::Int)::T where T
    M = length(S.P)
    K = length(S.I)-1
    n = (j-1) ÷ (M*K) + 1           # polynomial degree
    k = ((j- M*K*(n-1))-1) ÷ M + 1  # interval number
    m = j - M*K*(n-1) - M*(k-1)     # function number

    y = affinetransform(S.I[k],S.I[k+1], x)
    if S.P[m] isa Function
        S.P[m](y)[n]
    else
        S.P[m][y, n]
    end
end

function getindex(S::SumSpace{T}, xy::StaticVector{2}, j::Int)::T where T
    M = length(S.P)
    K = length(S.I)-1
    n = (j-1) ÷ (M*K) + 1           # polynomial degree
    k = ((j- M*K*(n-1))-1) ÷ M + 1  # interval number
    m = j - M*K*(n-1) - M*(k-1)     # function number

    # @assert S.P[m] isa ExtendedZernike || S.P[m] isa ExtendedWeightedZernike

    # y = affinetransform(S.I[k],S.I[k+1], x)
    # TODO: get multiple "intervals" working.
    # This scaling is exclusively for radial symmetric polys...
    rθ = RadialCoordinate(xy)
    r̃ = affinetransform(S.I[k],S.I[k+1], rθ.r)
    y = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    S.P[m][y, n]
    # if S.P[m] isa Function
    #     S.P[m](y)[n]
    # else
    #     S.P[m][y, n]
    # end
end