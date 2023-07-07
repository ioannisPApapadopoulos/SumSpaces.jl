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

function getindex(Z::ExtendedZernike{T},  xy::StaticVector{2}, j::Int)::T where T
    a, b = convert(T,Z.a), convert(T,Z.b)
    @assert a ≈ 0
    s = b    # Fractional power
    d = 2    # Dimension of space

    bl = findblockindex(axes(Z,2), j)
    ℓ = bl.I[1]-1 # degree
    k = bl.α[1] # index of degree
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k) # Fourier mode
    n = (ℓ - m) ÷ 2
    𝐣 = isodd.(ℓ .- k)

    c1 = (4*one(T))^s*gamma(1+s+n) / gamma(n+one(T))
    c2 = gamma(d/2+m+s+n) / gamma(d/2+m+n)


    xy in UnitDisk{T}() && return c1*c2*Zernike{T}(a, b)[xy, j]

    nrm = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) / sqrt((massmatrix(Jacobi{T}(b,a+m)).diag)[n+1])
    c3 = (-1)^n * gamma(d/2+m+n+s)/(gamma(-n-s)*gamma(d/2+m+2n+s+1))
    
    rθ = RadialCoordinate(xy)
    r, θ = rθ.r, rθ.θ

    V = 𝐣 == 1 ? r^m*cos(m*θ) : r^m*sin(m*θ)
    return c1*c3 * nrm * V * _₂F₁(n+s+1,d/2+m+n+s,d/2+m+2n+s+1,one(T)/r^2) / (r^(d+2*(m+n+s)))
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
    ExtendedZernike{T}(Q.a, Q.b)
end