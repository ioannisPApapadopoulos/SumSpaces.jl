"""
ExtendedChebyshev{kind,T}()

is a quasi-matrix representing the Hilbert transform of ChebyshevU polynomials of the specified kind 2
on ℝ. 

For an untransformed ExtendedChebyshevT, for x∈[-1,1]  ExtendedChebyshevT[x, n] = ChebyshevT()[x, n+1]
and outside the interval x∉[-1,1], there is an explicit formula for ExtendedChebyshevT[x, n]. 
"""

struct ExtendedChebyshev{kind,T} <: Basis{T} end
ExtendedChebyshev{kind}() where kind = ExtendedChebyshev{kind,Float64}()

axes(H::ExtendedChebyshev) = (Inclusion(ℝ), OneToInf())

==(a::ExtendedChebyshev{kind}, b::ExtendedChebyshev{kind}) where kind = true
==(a::ExtendedChebyshev, b::ExtendedChebyshev) = false

#### 
# Extended Chebyshev T
####

const ExtendedChebyshevT = ExtendedChebyshev{1}

function getindex(H::ExtendedChebyshevT{T}, x::Real, j::Int)::T where T
    x in ChebyshevInterval() && return ChebyshevT{T}()[x,j]
    
    j < 1 && throw_boundserror(H, [x,j])
    
    # ξ = inv(x + sqrtx2(x))
    # return transpose(ξ.^(j-1))
    if j == 1
        # return zero(T) # Experimental decaying support
        return one(T)
    else
        return (x + sqrtx2(x))^(1-j)
    end
end

"""
    extendedchebyshevt(n, z)

computes the `n`-th extended chebyshev T at `z`.
"""

#### 
# Extended Chebyshev U
####
const ExtendedChebyshevU = ExtendedChebyshev{2}

function getindex(H::ExtendedChebyshevU{T}, x::Real, j::Int)::T where T
    
    j < 1 && throw_boundserror(H, [x,j])

    # eU_-1 = HilbertTransform[wT_1] - 1
    if j == 1
        abs(x) ≤ 1 && return zero(T)
        ξ = - sign(x) * x / sqrt(x^2 - one(T))
    # eU_0 = HilbertTransform[wT_0]
    elseif j == 2
        abs(x) ≤ 1 && return zero(T)
        ξ = - sign(x) / sqrt(x^2 - one(T))
    elseif isodd(j)
        abs(x) ≤ 1 && return ChebyshevU{T}()[x,j-2]
        η = inv(x + sqrtx2(x))
        ξ = 2 * sum(η .^(2:2:j-3)) + one(T) - sign(x) * x / sqrt(x^2 - one(T))
        # ξ = 2 * sum(η .^(2:2:j-3)) - abs(x) / sqrt(x^2 - one(T)) # decaying support
    else
        abs(x) ≤ 1 && return ChebyshevU{T}()[x,j-2]
        η = inv(x + sqrtx2(x))
        ξ = 2 * sum(η .^(1:2:j-3)) - sign(x) / sqrt(x^2 - one(T))
    end
        
    return ξ
end

struct ExtendedWeightedChebyshev{kind,T} <: Basis{T} end
ExtendedWeightedChebyshev{kind}() where kind = ExtendedWeightedChebyshev{kind,Float64}()

axes(H::ExtendedWeightedChebyshev) = (Inclusion(ℝ), OneToInf())

==(a::ExtendedWeightedChebyshev{kind}, b::ExtendedWeightedChebyshev{kind}) where kind = true
==(a::ExtendedWeightedChebyshev, b::ExtendedWeightedChebyshev) = false

###
# Extended Weighted Chebyshev T
###

const ExtendedWeightedChebyshevT = ExtendedWeightedChebyshev{1}

function getindex(H::ExtendedWeightedChebyshevT{T}, x::Real, j::Int)::T where T
    -1 <= x <= 1 && return Weighted(ChebyshevT{T}())[x,j]
    return 0.
end
summary(io::IO, w::ExtendedWeightedChebyshevT{Float64}) = print(io, "ExtendedWeightedChebyshevT()")

###
# Extended Weighted Chebyshev U
###

const ExtendedWeightedChebyshevU = ExtendedWeightedChebyshev{2}

function getindex(H::ExtendedWeightedChebyshevU{T}, x::Real, j::Int)::T where T
    x in ChebyshevInterval() && return Weighted(ChebyshevU{T}())[x,j]
    return zero(T)
end
summary(io::IO, w::ExtendedWeightedChebyshevU{Float64}) = print(io, "ExtendedWeightedChebyshevU()")

==(a::ExtendedWeightedChebyshevU, b::ExtendedWeightedChebyshevU) = true
==(a::ExtendedWeightedChebyshevU, b::ExtendedWeightedChebyshevT) = false


###
# inner products
###

@simplify *(Tc::QuasiAdjoint{<:Any,<:ExtendedChebyshevT}, V::ExtendedWeightedChebyshevT) =
    ChebyshevT{eltype(Tc)}()'Weighted(ChebyshevT{eltype(V)}())
@simplify *(Tc::QuasiAdjoint{<:Any,<:ChebyshevT}, V::ExtendedWeightedChebyshevT) =
    Tc*Weighted(ChebyshevT{eltype(V)}())    
@simplify *(Tc::QuasiAdjoint{<:Any,<:ExtendedChebyshevT}, V::Weighted{<:Any,<:ChebyshevT}) =
    ChebyshevT{eltype(Tc)}()'V


@simplify function *(Tc::QuasiAdjoint{<:Any,<:ExtendedChebyshevT}, Ũ::ExtendedChebyshevU)
    M = ChebyshevU{eltype(Tc)}()'ChebyshevT{eltype(Ũ)}() # TODO:Check
    TT = eltype(M)
    ApplyArray(hvcat, 2, convert(TT,Inf), Zeros{TT}(1,∞), Zeros{TT}(∞), M)
end