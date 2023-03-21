"""
ExtendedZernike{T}(a, b)

is a quasi-matrix representing extended Zernike functions on ℝ. 

"""

#### 
# ExtendedZernike
####

struct ExtendedZernike{T} <: Basis{T}
    a::T
    b::T
    ExtendedZernike{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

ExtendedZernike(a::T, b::T) where T = ExtendedZernike{Float64}(a::T, b::T)

axes(P::ExtendedZernike{T}) where T = (Inclusion(ℝ^2),blockedrange(oneto(∞)))

==(P::ExtendedZernike, Q::ExtendedZernike) = P.a == Q.a && P.b == Q.b

function getindex(P::ExtendedZernike{T},  xy::StaticVector{2}, j::Int)::T where T
    a, b = convert(T,P.a), convert(T, P.b)
    @assert a ≈ 0
    s = b    # Fractional power
    d = 2    # Dimension of space

    bl = findblockindex(axes(P,2), j)
    ℓ = bl.I[1]-1 # degree
    k = bl.α[1] # index of degree
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k) # Fourier mode
    n = (ℓ - m)

    c1 = T(4)^s*gamma(1+s+n)
    c2 = gamma((d+2*(m+s+n))/2) / (factorial(n)*gamma((d+s*(m+n))/2))

    xy in UnitDisk{T}() && return c1*c2*Zernike{T}(a, b)[xy, j]

    c3 = (-1)^n * gamma(d/2+m+n+s)/(gamma(-n-s)*gamma(d/2+m+2n+s+1))

    r2 = first(xy)^2+last(xy)^2
    return c1*c3*_₂F₁(n+s+1,d/2+m+n+s,d/2+m+2n+s+1,one(T)/r2) / (r2^(d/2+m+n+s))
end

#### 
# ExtendedWeightedZernike
####

struct ExtendedWeightedZernike{T} <: Basis{T}
    a::T
    b::T
    ExtendedWeightedZernike{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end


ExtendedWeightedZernike(a::T, b::T) where T = ExtendedWeightedZernike{Float64}(a::T, b::T)

axes(P::ExtendedWeightedZernike{T}) where T = (Inclusion(ℝ^2),blockedrange(oneto(∞)))
==(P::ExtendedWeightedZernike, Q::ExtendedWeightedZernike) = P.a == Q.a && P.b == Q.b

function getindex(P::ExtendedWeightedZernike{T}, xy::StaticVector{2}, j::Int)::T where T
    xy in UnitDisk{T}() && return Weighted(Zernike{T}(P.a, P.b))[xy,j]
    return zero(T)
end

###
# Fractional Laplacians
###

# function _coeff_zernike(s::T) where T
#     d = 2
#     k = mortar(Base.OneTo.(oneto(∞))) 
#     ℓ = mortar(Fill.(oneto(∞),oneto(∞))) .- 1
#     m = k .- isodd.(k) .* iseven.(ℓ) .- iseven.(k) .* isodd.(ℓ) 
#     n = ℓ .- m
#     T(4)^s*gamma.(1+s .+ n) .* gamma.((d.+2*(s.+ℓ)) ./ 2) ./ (factorial.(n).*gamma.((d.+s*ℓ) ./ 2))
# end

function *(L::AbsLaplacianPower, P::ExtendedZernike{T}) where T
    @assert axes(L,1) == axes(P,1) && P.a ≈ 0 && P.b == -L.α
    ExtendedWeightedZernike{T}(P.a,P.b)
end

function *(L::AbsLaplacianPower, Q::ExtendedWeightedZernike{T}) where T
    @assert axes(L,1) == axes(Q,1) && Q.a ≈ 0 && Q.b == L.α
    s = Q.b
    ExtendedZernike{T}(Q.a, Q.b)
end