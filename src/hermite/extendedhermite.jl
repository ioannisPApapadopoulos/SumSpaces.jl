"""
ExtendedHermite{T}()

is a quasi-matrix representing extended Hermite functions on ℝ. These are defined abs

        (-Δˢ)[exp(-x²) Hₙ(x)]
"""

#### 
# ExtendedHermite
####

struct ExtendedHermite{T} <: Basis{T} 
    s::T
    ExtendedHermite{T}(s) where T = new{T}(convert(T,s))
end

ExtendedHermite(s::T) where T = ExtendedHermite{Float64}(s::T)

axes(P::ExtendedHermite) = (Inclusion(ℝ), OneToInf())

==(P::ExtendedHermite, Q::ExtendedHermite) = P.s == Q.s₁F₁

function getindex(H::ExtendedHermite{T}, x::Real, j::Int)::T where T
    n = j-1; s = convert(T, H.s);
    bπ = convert(T, π)

    if s ≈ 0.5
        return -(1/sqrt(bπ)) * 2^(1+n-2*floor(n/2)) * x^(n-2*floor(n/2)) * gamma(s + 1/2 + floor((n+1)/2)) * _₁F₁(s + 1/2 + floor((n+1)/2), (1 + 2n - 4*floor(n/2))/2, -x^2)
    elseif s ≈ -0.5
        n == 0 && return 0.0
        n > 15 && return 0.0
        x^(n - 2*floor(n/2)) / 2 * ((-1)^(floor(n/2) + 1) * gamma(max(1, floor((n - 2)/2)+1)) * (1 - n + 2*floor(n/2)) / sqrt(bπ) + (-1)^floor(n/2) * gamma(floor((n - 1)/2)+1) * _₁F₁(n - floor(n/2), one(T)/2 + n - 2*floor(n/2), -x^2) / gamma(one(T)/2 + n - 2*floor(n/2)))
    else       
        t = 4^s*bπ*x^(n-2*floor(n/2)) * _₁F₁(n+s-floor(n/2)+one(T)/2,n-2floor(n/2)+one(T)/2,-x^2)
        b = sin(bπ/2*(4floor(n/2)-2n-2s+1))*gamma(n-2floor(n/2)+one(T)/2)*gamma(-n-s+floor(n/2)+one(T)/2)
        return t/b
    end
end

#### 
# ExtendedNormalizedHermite
####

struct ExtendedNormalizedHermite{T} <: Basis{T} 
    s::T
    ExtendedNormalizedHermite{T}(s) where T = new{T}(convert(T,s))
end

ExtendedNormalizedHermite(s::T) where T = ExtendedNormalizedHermite{Float64}(s::T)

axes(P::ExtendedNormalizedHermite) = (Inclusion(ℝ), OneToInf())

==(P::ExtendedNormalizedHermite, Q::ExtendedNormalizedHermite) = P.s == Q.s₁F₁

function getindex(H::ExtendedNormalizedHermite{T}, x::Real, j::Int)::T where T
    n = j-1
    bπ = convert(T,π)
    sqrt(sqrt(1/bπ))/sqrt(2^n*gamma(big(n+1))) * ExtendedHermite(H.s)[x,j]
end

###
# Fractional Laplacian
###
function *(L::AbsLaplacianPower, W::Weighted{<:Any,<:Hermite})
    T = eltype(W)
    @assert axes(L,1) == axes(W.P,1)
    s = convert(T, L.α)
    ExtendedHermite{T}(s)
end

function *(L::AbsLaplacianPower, W::Weighted{<:Any,Normalized{<:Any, <:Hermite}})
    T = eltype(W)
    @assert axes(L,1) == axes(W.P.P,1)
    s = convert(T, L.α)
    ExtendedNormalizedHermite{T}(s)
end