"""
SumSpaceL{T}() and SumSpaceLD{T}()

are quasi-matrices representing the Laguerre primal and dual sum spaces
on (-∞,∞).
"""
###
# Primal and dual sum space
###

struct SumSpaceL{T,E} <: Basis{T} 
    I::E
end

struct SumSpaceLD{T,E} <: Basis{T} 
    I::E
end

SumSpaceL{T}(I::AbstractVector=[-1.,1.]) where T = SumSpaceL{T, typeof(I)}(I)
SumSpaceL(I::AbstractVector=[-1.,1.]) = SumSpaceL{Float64}(I)

SumSpaceLD{T}(I::AbstractVector=[-1.,1.]) where T = SumSpaceLD{T, typeof(I)}(I)
SumSpaceLD(I::AbstractVector=[-1.,1.]) = SumSpaceLD{Float64}(I)

axes(S::SumSpaceL) = (Inclusion(ℝ), _BlockedUnitRange(2:2:∞))
axes(S::SumSpaceLD) = (Inclusion(ℝ), _BlockedUnitRange(2:2:∞))

==(a::SumSpaceL, b::SumSpaceL) = a.I == b.I
==(a::SumSpaceLD, b::SumSpaceLD) = a.I == b.I

==(a::SumSpaceL, b::SumSpaceLD) = false

function getindex(S::SumSpaceL{T}, x::Real, j::Int)::T where T
    y = affinetransform(S.I[1],S.I[2], x)
    isodd(j) && return HilbertWeightedLaguerre{T}(one(T)/2)[y, (j ÷ 2)+1]
    ExtendedWeightedLaguerre{T}(one(T)/2)[y, j ÷ 2]
end

function getindex(S::SumSpaceLD{T}, x::Real, j::Int)::T where T
    y = affinetransform(S.I[1],S.I[2], x)
    isodd(j) && return HilbertWeightedLaguerre{T}(-one(T)/2)[y, (j ÷ 2)+1]
    ExtendedWeightedLaguerre{T}(-one(T)/2)[y, j ÷ 2]
end


###
# Operators
###

# Identity Sp -> Sds
function \(Sd::SumSpaceLD, Sp::SumSpaceL)
    Sd.I != Sp.I && error("Sum spaces bases not centred on same interval")
    T = promote_type(eltype(Sp), eltype(Sd))
    onevec = mortar(Fill.(one(T),Fill(2,∞)))
    d = Interlace(one(T)/2:∞, one(T)/2:∞) .* onevec
    ld = Interlace(-one(T):-one(T):-∞, -one(T):-one(T):-∞) .* onevec
    dat = BlockBroadcastArray(hcat,d,ld)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,0), (0,0))
    return A
end

# Derivative Sp -> Sd
@simplify function *(D::Derivative, Sp::SumSpaceL)
    T = eltype(Sp)
    (a,b) = Sp.I

    fracvec = mortar(Fill.(2* one(T) / (b-a), Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    ld = Interlace(one(T):∞, one(T):∞) .* fracvec 
    dat = BlockBroadcastArray(hcat,zs,ld)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,0), (0,0))
    return ApplyQuasiMatrix(*, SumSpaceLD{T}(Sp.I), A)
end

# Hilbert: Sp -> Sp 
@simplify function *(H::Hilbert, Sp::SumSpaceL)
    T = eltype(Sp)
    onevec = mortar(Fill.(convert(T, π), Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    dat = BlockBroadcastArray(hcat,onevec,zs,-onevec)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (0,0), (1,1))
    return ApplyQuasiMatrix(*, SumSpaceL{T}(Sp.I), A)
end

# x Sp -> Sp
# function jacobimatrix(Sp::SumSpaceL)
#     T = eltype(Sp)
#     halfvec = mortar(Fill.(1/2,Fill(2,∞)))
#     zs = mortar(Zeros.(Fill(2,∞)))
#     dat = BlockBroadcastArray(hcat,halfvec,zs,zs,zs,halfvec,zs)
#     dat = BlockVcat([0,0,0,0,0,1/2]', dat)
#     A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,1), (1,0))'
#     return A
# end