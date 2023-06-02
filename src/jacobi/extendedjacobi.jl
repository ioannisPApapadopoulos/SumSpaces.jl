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
    a, b = convert(T,P.a), convert(T, P.b)
    x in ChebyshevInterval() && return Jacobi{T}(a, b)[x,j]

    if a != b
        error("a ≂̸ b, not implemented for different Jacobi parameters.")
    end
    s = a
    if isodd(j)
        if j == 1 && s ≈ -1/2
            # @warn "degree 0 when s = -1/2 is undefined."
            return one(T)
        end
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

recurrencecoefficients(P::ExtendedJacobi{T}) where T = recurrencecoefficients(Jacobi{T}(P.a, P.b))

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

recurrencecoefficients(P::ExtendedWeightedJacobi{T}) where T = recurrencecoefficients(Jacobi{T}(P.a, P.b))

###
# Derivative of ExtendedJacobi
###

struct DerivativeExtendedJacobi{T} <: Basis{T}
    a::T
    b::T
    DerivativeExtendedJacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

DerivativeExtendedJacobi(a::T, b::T) where T = DerivativeExtendedJacobi{Float64}(a::T, b::T)

axes(P::DerivativeExtendedJacobi) = (Inclusion(ℝ), OneToInf())

==(P::DerivativeExtendedJacobi, Q::DerivativeExtendedJacobi) = P.a == Q.a && P.b == Q.b

function getindex(P::DerivativeExtendedJacobi{T}, x::Real, j::Int)::T where T
    a, b = convert(T,P.a), convert(T, P.b)
    if x in ChebyshevInterval()
        if j==1
            return zero(T)
        else 
            return (a+b+j)/T(2)*Jacobi{T}(a+one(T), b+one(T))[x,j-1]
        end
    end

    if a != b
        error("a ≂̸ b, not implemented for different Jacobi parameters.")
    end
    s = a
    if j==1 && s ≈ -1/2
        return zero(T)
    elseif isodd(j)
        n = Int((j-1)/2)
        c = -Jacobi{T}(s, s)[1,j] * sin(π*(n + s)) * gamma(2n+2s+1) * gamma(n + 1/2)
        k = 4^s * gamma(s+n+1/2) * Jacobi{T}(s, -1/2)[1,n+1] * sqrt(π) * (-4)^n * gamma(2n+s+3/2)
        c = c/k
        return c * (-(2/((3/2 + 2n + s)*x^3)) * (1/2 + n + s) * (1 + n + s) * abs(x)^(-1 - 2n - 2*s) * _₂F₁(3/2+n+s, 2+n+s, 5/2+2n+s, 1/x^2)
        + (-1-2n-2*s)*abs(x)^(-2-2n-2*s) * _₂F₁(1/2+n+s, 1+n+s, 3/2+2n+s, 1/x^2)*sign(x)
        )
        # return c * _₂F₁(n+s+1/2, n+s+1, 2n+s+3/2, 1/x^2) / abs(x)^(2n+2s+1)
    else
        n = Int((j-2)/2)
        c = Jacobi{T}(s, s)[1,j] * (-1)^(n+1) * sin(π*(n + s)) * gamma(2n+2s+2)  * gamma(n + 3/2)
        k = 4^s * gamma(s+n+3/2) * Jacobi{T}(s, 1/2)[1,n+1] * sqrt(π) * 2^(2n+1) * gamma(2n+s+5/2)
        c = c/k
        F = (
               abs(x)^(-3-2n-2*s) * _₂F₁(1+n+s, 3/2+n+s, 5/2+2n+s, 1/x^2) 
            - (2*(1+n+s)*(3/2+n+s)*abs(x)^(-3-2n-2*s)*_₂F₁(2+n+s, 5/2+n+s, 7/2+2n+s, 1/x^2))/((5/2+2n+s)*x^2) 
            + (-3-2n-2*s)*x*abs(x)^(-4-2n-2*s)*_₂F₁(1+n+s, 3/2+n+s, 5/2+2n+s, 1/x^2)*sign(x)
        )
        return c * F
    end
end

@simplify function *(D::Derivative, P::ExtendedJacobi)
    DerivativeExtendedJacobi(P.a, P.b)
end

#### 
# Derivative of ExtendedWeightedJacobi
####

struct DerivativeExtendedWeightedJacobi{T} <: Basis{T}
    a::T
    b::T
    DerivativeExtendedWeightedJacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end


DerivativeExtendedWeightedJacobi(a::T, b::T) where T = DerivativeExtendedWeightedJacobi{Float64}(a::T, b::T)

axes(H::DerivativeExtendedWeightedJacobi) = (Inclusion(ℝ), OneToInf())

==(P::DerivativeExtendedWeightedJacobi, Q::DerivativeExtendedWeightedJacobi) = P.a == Q.a && P.b == Q.b

function getindex(P::DerivativeExtendedWeightedJacobi{T}, x::Real, j::Int)::T where T
    x in ChebyshevInterval() && return -2j*Weighted(Jacobi{T}(P.a-one(T), P.b-one(T)))[x,j+1]
    return 0.
end

@simplify function *(D::Derivative, P::ExtendedWeightedJacobi)
    DerivativeExtendedWeightedJacobi(P.a, P.b)
end

###
# Generalised Extended Jacobi
###
struct GeneralExtendedJacobi{T} <: Basis{T}
    a::T
    s::T
    GeneralExtendedJacobi{T}(a, s) where T = new{T}(convert(T,a), convert(T,s))
end

GeneralExtendedJacobi(a::T, s::T) where T = GeneralExtendedJacobi{Float64}(a::T, s::T)

axes(P::GeneralExtendedJacobi) = (Inclusion(ℝ), OneToInf())

==(P::GeneralExtendedJacobi, Q::GeneralExtendedJacobi) = P.a == Q.a && P.s == Q.s

function _coeffn(s::T, n::Int) where T
    m = n ÷ 2
    if iseven(n)
        return 4^s * gamma(T(1) + s + m)*gamma(s + T(1)/2 + m) / (gamma(T(1) + m) * gamma(m + T(1)/2))
    else
        return 4^s * gamma(T(1) + s + m)*gamma(s + T(3)/2 + m) / (gamma(T(1) + m) * gamma(m + T(3)/2))
    end
end

function getindex(P::GeneralExtendedJacobi{T}, x::Real, j::Int)::T where T
    a, s = convert(T, P.a), convert(T, P.s)
    n = j-1
    a == s && return _coeffn(s, n) * ExtendedJacobi{T}(s, s)[x, j]
    # a,s,n = map(big, (a,s,n))
    bn = n > 20 ? big(n) : n
    c = 4^s * gamma(a + n + 1) / factorial(bn) * x^(n - 2 * floor(n/2))
    bπ = convert(T,π)

    if s ≈ 0.5
        if x in ChebyshevInterval()
            return c*((-T(1))^(floor(n/2)) * factorial(floor((n + 1)/2)) * _₂F₁(-a + T(1)/2 - floor(n/2), n - floor(n/2) + T(1), n - 2*floor(n/2) + T(1)/2, x^2) / (gamma(n - 2*floor(n/2) + T(1)/2) * gamma(a + floor(n/2) + T(1)/2)))
        else
            return c*((-T(2)^(-n - 1) * abs(x)^(-2*floor((n - 1)/2) - 4) * gamma(n + 2) * _₂F₁(T(3)/2 + floor(n/2), (2 * floor((n - 1)/2) + T(3))/2 + T(1)/2, (2n + 3)/2 + a, T(1)/x^2) * sin(bπ/2)) / (sqrt(bπ) * gamma((2n + T(3))/2 + a)))
        end
    else
        if x in ChebyshevInterval()
            num = bπ * _₂F₁(-a + s - floor(n/2), n + s - floor(n/2) + T(1)/2, n - 2 * floor(n/2) + T(1)/2, x^2)
            den = sin(bπ/2 * (T(1) - 2 * n - 2 * s + 4 * floor(n/2))) *
                gamma(n - 2 * floor(n/2) + T(1)/2) *
                gamma(-n - s + floor(n/2) + T(1)/2) *
                gamma(a - s + floor(n/2) + T(1))
            return c*num/den
        else
            num = (2^(-n - 2 * s) * abs(x)^(-2 * floor((n - 1)/2) - 3 - 2 * s) *
                gamma(n + T(1) + 2 * s) *
                _₂F₁(T(1) + floor(n/2) + s, (2 * floor((n - 1)/2) + 3)/2 + s, (2 * n + T(3))/2 + a, T(1) / x^2) *
                sin(bπ * s))
    
            den = sqrt(bπ) * gamma((2 * n + T(3))/2 + a)
            return -c*num/den
        end
    end
end

###
# Fractional Laplacians
###

function _coeff(s, nstart)
    n = 0:∞
    cₒ = 4^s .* gamma.(1 .+ s .+ n).*gamma.(s .+ 3/2 .+ n) ./ (gamma.(1 .+ n) .* gamma.(n .+ 3/2))
    cₑ = 4^s .* gamma.(1 .+ s .+ n).*gamma.(s .+ 1/2 .+ n) ./ (gamma.(1 .+ n) .* gamma.(n .+ 1/2))
    Interlace(cₑ, cₒ)[nstart:end]
end

function *(L::AbsLaplacianPower, P::ExtendedJacobi{T}) where T
    @assert axes(L,1) == axes(P,1) && P.a == P.b == -L.α
    s = P.a
    ExtendedWeightedJacobi{T}(s,s) .* (one(T) ./ _coeff(s, 1))'
end

function *(L::AbsLaplacianPower, Q::ExtendedWeightedJacobi{T}) where T
    @assert axes(L,1) == axes(Q,1)

    Q.a == Q.b == L.α && return ExtendedJacobi{T}(Q.a, Q.a) .* _coeff(Q.a, 1)'
    Q.a == Q.b && return GeneralExtendedJacobi{T}(Q.a, L.α)
end

function *(L::AbsLaplacianPower, G::GeneralExtendedJacobi{T}) where T
    @assert axes(L,1) == axes(G,1)

    if L.α == -G.s
        return ExtendedWeightedJacobi{T}(G.a, G.a)
    else
        return GeneralExtendedJacobi{T}(G.a, L.α+G.s)
    end
end