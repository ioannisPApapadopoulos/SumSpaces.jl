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

function getindex(H::ExtendedHermite{T}, x::Real, j::Int)::T where T
    n = j-1; s = convert(T,H.s);

    s ≈ 0.5 && return error("Not implemented for s = 1/2")

    t = 4^s*convert(T,π)*x^(n-2*floor(n/2)) * _₁F₁(n+s-floor(n/2)+one(T)/2,n-2floor(n/2)+one(T)/2,-x^2)
    b = sin(convert(T,π)/2*(4floor(n/2)-2n-2s+1))*gamma(n-2floor(n/2)+one(T)/2)*gamma(-n-s+floor(n/2)+one(T)/2)
    t/b
end