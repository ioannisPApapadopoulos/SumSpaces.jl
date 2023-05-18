"""
ExtendedSymmetricLaguerre{T}()

is a quasi-matrix representing extended symmetric Laguerre functions on ℝ. These are defined as

        (-Δˢ)[exp(-x²) Lᵅₙ(x²)]
"""


#### 
# SymmetricLaguerre
####

struct SymmetricLaguerreWeight{T} <: Weight{T}
    α::T
end

SymmetricLaguerreWeight{T}() where T = SymmetricLaguerreWeight{T}(zero(T))
SymmetricLaguerreWeight() = SymmetricLaguerreWeight{Float64}()
axes(::SymmetricLaguerreWeight{T}) where T = (Inclusion(ℝ),)
function getindex(w::SymmetricLaguerreWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    exp(-x^2) # x^(2*w.α) * 
end

struct SymmetricLaguerre{T} <: Basis{T}
    α::T
    SymmetricLaguerre{T}(α) where T = new{T}(convert(T,α))
end

axes(P::SymmetricLaguerre) = (Inclusion(ℝ), OneToInf())
==(P::SymmetricLaguerre, Q::SymmetricLaguerre) = P.s == Q.s
orthogonalityweight(S::SymmetricLaguerre{T}) where T = SymmetricLaguerreWeight{T}(S.α) 

function getindex(L::SymmetricLaguerre{T}, x::Real, j::Int)::T where T
    Laguerre{T}(L.α)[x^2, j]
end

#### 
# ExtendedSymmetricLaguerre
####

struct ExtendedSymmetricLaguerre{T} <: Basis{T}
    α::T
    s::T
    ExtendedSymmetricLaguerre{T}(α,s) where T = new{T}(convert(T,α), convert(T,s))
end

axes(P::ExtendedSymmetricLaguerre) = (Inclusion(ℝ), OneToInf())
==(P::ExtendedSymmetricLaguerre, Q::ExtendedSymmetricLaguerre) = P.s == Q.s && P.α == Q.α

function getindex(H::ExtendedSymmetricLaguerre{T}, x::Real, j::Int)::T where T
    n = j-1; α = convert(T, H.α); s = convert(T,H.s);

    # s ≈ 0.5 && return error("Not implemented for s = 1/2")

    t = 4^s*gamma(s+1/2)*gamma(n+s+α+1)*pFq((s+1/2,n+s+α+1),(1/2,s+α+1),-x^2)
    b = sqrt(π)*gamma(n+1)*gamma(s+α+1)
    t/b
end

###
# Fractional Laplacians
###

function *(L::AbsLaplacianPower, W::Weighted{<:Any,<:SymmetricLaguerre})
    T = eltype(W)
    # Formula works for exp(-x^2)*L^\alpha_n so does NOT include the weight x^2α.
    # The notation we use here means it L*Weighted(SymmetricLaguerre) only does what
    # we expect when α = 0. 
    @assert axes(L,1) == axes(W.P,1) && W.P.α == 0
    s = convert(T, L.α)
    α = convert(T, W.P.α)
    ExtendedSymmetricLaguerre{T}(α,s)
end