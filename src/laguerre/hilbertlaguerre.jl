"""
HilbertWeightedLaguerre{kind,T}()

is a quasi-matrix representing extended Laguerre functions on ℝ. 
"""

struct HilbertWeightedLaguerre{T} <: Basis{T}
    α::T
    HilbertWeightedLaguerre{T}(α) where T = new{T}(convert(T,α))
end

# HilbertWeightedLaguerre(α::T) where T = HilbertWeightedLaguerre{Float64}(α::T)
HilbertWeightedLaguerre{T}() where T = HilbertWeightedLaguerre{T}(zero(T))
HilbertWeightedLaguerre() = HilbertWeightedLaguerre{Float64}()
HilbertWeightedLaguerre(α::T) where T = HilbertWeightedLaguerre{float(T)}(α)

axes(P::HilbertWeightedLaguerre) = (Inclusion(ℝ), OneToInf())
==(P::HilbertWeightedLaguerre, Q::HilbertWeightedLaguerre) = P.α == Q.α

function getindex(H::HilbertWeightedLaguerre{T}, x::Real, j::Int)::T where T
    wL = Weighted(Laguerre(H.α))
    xc = axes(wL,1)
    (inv.(x .- xc') * wL)[j] / π
end

#### 
# ExtendedWeightedLaguerre
####

struct ExtendedWeightedLaguerre{T} <: Basis{T}
    α::T
    ExtendedWeightedLaguerre{T}(α) where T = new{T}(convert(T,α))
end


ExtendedWeightedLaguerre(α::T) where T = ExtendedWeightedLaguerre{Float64}(α::T)

axes(H::ExtendedWeightedLaguerre) = (Inclusion(ℝ), OneToInf())

==(P::ExtendedWeightedLaguerre, Q::ExtendedWeightedLaguerre) = P.α == Q.α

function getindex(P::ExtendedWeightedLaguerre{T}, x::Real, j::Int)::T where T
    x in HalfLine{T}() && return Weighted(Laguerre{T}(P.α))[x,j]
    return 0.
end

#####
# Derivatives
#####

@simplify function *(D::Derivative, w_A::ExtendedWeightedLaguerre)
    T = promote_type(eltype(D),eltype(w_A))
    D = BandedMatrix(-1=>one(T):∞)
    ExtendedWeightedLaguerre{T}(w_A.α-1)*D
end

@simplify function *(D::Derivative, w_A::HilbertWeightedLaguerre)
    T = promote_type(eltype(D),eltype(w_A))
    D = BandedMatrix(-1=>one(T):∞)
    HilbertWeightedLaguerre{T}(w_A.α-1)*D
end

#####
# Conversion
#####

function \(w_A::ExtendedWeightedLaguerre, w_B::ExtendedWeightedLaguerre)
    T = promote_type(eltype(w_A), eltype(w_B))
    if w_A.α ≈ w_B.α
        Eye{T}(∞)
    elseif w_A.α + 1 ≈ w_B.α
        BandedMatrix(0=>w_B.α:∞, -1=>-one(T):-one(T):-∞)
    else
        error("Not implemented for this choice of w_A.α and w_B.α.")
    end
end

function \(w_A::HilbertWeightedLaguerre, w_B::HilbertWeightedLaguerre)
    T = promote_type(eltype(w_A), eltype(w_B))
    if w_A.α ≈ w_B.α
        Eye{T}(∞)
    elseif w_A.α + 1 ≈ w_B.α
        BandedMatrix(0=>w_B.α:∞, -1=>-one(T):-one(T):-∞)
    else
        error("Not implemented for this choice of w_A.α and w_B.α.")
    end
end