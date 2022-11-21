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
    x in ChebyshevInterval() && return Jacobi{T}(P.a, P.b)[x,j]

    if P.a != P.b
        error("a ≂̸ b, not implemented for different Jacobi parameters.")
    end
    s = P.a
    if isodd(j)
        n = Int((j-1)/2)
        c = -Jacobi{T}(s, s)[1,j] * sin(π*(n + s)) * gamma(2n+2s+1) * gamma(n + 1/2)
        k = 4^s * gamma(s+n+1/2) * Jacobi{T}(s, -1/2)[1,n+1] * sqrt(π) * (-4)^n * gamma(2n+s+3/2)
        c = c/k
        return c * _₂F₁(n+s+1/2, n+s+1, 2n+s+3/2, 1/x^2) / abs(x)^(2n+2s+1)
    else
        n = Int((j-2)/2)
        c = Jacobi{T}(s, s)[1,j] * (-1)^(n+1) * sin(π*(n + s)) * gamma(2n+2s+2)  * gamma(n + 3/2)
        k = 4^s * gamma(s+n+3/2) * Jacobi{T}(s, 1/2)[1,n+1] * sqrt(π) * 2^(2n+1) * gamma(2n+s+5/2)
        c = c/k
        return c * x * _₂F₁(n+s+1, n+s+3/2, 2n+s+5/2, 1/x^2) / abs(x)^(2n+2s+3)
    end
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
    x in ChebyshevInterval() && return Weighted(Jacobi{T}(P.a, P.b))[x,j]
    return 0.
end