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

==(P::ExtendedHermite, Q::ExtendedHermite) = P.s == Q.s

orthogonalityweight(::ExtendedHermite{T}) where T = HermiteWeight{T}()
jacobimatrix(H::ExtendedHermite{T}) where T = jacobimatrix(Hermite{T}())

function getindex(H::ExtendedHermite{T}, x::Real, j::Int)::T where T
    n = j-1; s = convert(T,H.s);

    if s ≈ 0.5
        return -(1/sqrt(π)) * 2^(1+n-2*floor(n/2)) * x^(n-2*floor(n/2)) * gamma(s + 1/2 + floor((n+1)/2)) * _₁F₁(s + 1/2 + floor((n+1)/2), (1 + 2n - 4*floor(n/2))/2, -x^2)
    else       
        t = 4^s*convert(T,π)*x^(n-2*floor(n/2)) * _₁F₁(n+s-floor(n/2)+one(T)/2,n-2floor(n/2)+one(T)/2,-x^2)
        b = sin(convert(T,π)/2*(4floor(n/2)-2n-2s+1))*gamma(n-2floor(n/2)+one(T)/2)*gamma(-n-s+floor(n/2)+one(T)/2)
        return t/b
    end
end

function *(L::AbsLaplacianPower, W::Weighted{<:Any,<:Hermite})
    T = eltype(W)
    @assert axes(L,1) == axes(W.P,1)
    s = convert(T, L.α)
    ExtendedHermite{T}(s)
end