"""
SumSpaceP{T}() and SumSpaceD{T}()

are quasi-matrices representing the primal and dual sum spaces
on (-∞,∞).
"""
###
# Primal and dual sum space
###

struct SumSpaceP{T,E} <: Basis{T} 
    I::E
end

struct SumSpaceD{T,E} <: Basis{T} 
    I::E
end

SumSpaceP{T}(I::AbstractVector=[-1.,1.]) where T = SumSpaceP{T, typeof(I)}(I)
SumSpaceP(I::AbstractVector=[-1.,1.]) = SumSpaceP{Float64}(I)

SumSpaceD{T}(I::AbstractVector=[-1.,1.]) where T = SumSpaceD{T, typeof(I)}(I)
SumSpaceD(I::AbstractVector=[-1.,1.]) = SumSpaceD{Float64}(I)

# const SumSpaceP = SumSpace{1}
# const SumSpaceD = SumSpace{2}

axes(S::SumSpaceP) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))
axes(S::SumSpaceD) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

==(a::SumSpaceP, b::SumSpaceP) = a.I == b.I
==(a::SumSpaceD, b::SumSpaceD) = a.I == b.I

==(a::SumSpaceP, b::SumSpaceD) = false

function getindex(S::SumSpaceP{T}, x::Real, j::Int)::T where T
    y = affinetransform(S.I[1],S.I[2], x)
    isodd(j) && return ExtendedChebyshevT{T}()[y, (j ÷ 2)+1]
    ExtendedWeightedChebyshevU{T}()[y, j ÷ 2]
end

function getindex(S::SumSpaceD{T}, x::Real, j::Int)::T where T
    y = affinetransform(S.I[1],S.I[2], x)
    isodd(j) && return ExtendedChebyshevU{T}()[y, (j ÷ 2)+1]
    ExtendedWeightedChebyshevT{T}()[y, j ÷ 2]
end

###
# Appended sum space
###

struct AppendedSumSpace{T, E} <: Basis{T} 
    A
    C
    I::E
end
AppendedSumSpace{T}(A, C, I::AbstractVector) where T = AppendedSumSpace{T,typeof(I)}(A, C, I)
AppendedSumSpace(A, C, I::AbstractVector) = AppendedSumSpace{Float64}(A, C, I)
AppendedSumSpace(A, C) = AppendedSumSpace(A, C, [-1.,1.])


axes(ASp::AppendedSumSpace) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))


function getindex(ASp::AppendedSumSpace{T}, x::Real, j::Int)::T where T

    if j == 1
        return SumSpaceP{T}(ASp.I)[x,1]
    elseif 2<=j<=5
        return ASp.A[j-1](x)
    else
        return SumSpaceP{T}(ASp.I)[x,j-4]
    end
end

###
# Operators
###

# Credit to Timon Gutleb for help for the below implementation of the identity mapping
# Identity Sp -> Sd
function \(Sd::SumSpaceD, Sp::SumSpaceP)
    Sd.I != Sp.I && error("Sum spaces bases not centred on same element")
    T = promote_type(eltype(Sp), eltype(Sd))
    halfvec = mortar(Fill.(1/2,Fill(2,∞)))
    d = Diagonal((-1).^(2:∞))*halfvec
    zs = mortar(Zeros.(Fill(2,∞)))
    ld = Diagonal((-1).^(1:∞))*halfvec
    dat = BlockBroadcastArray(hcat,d,zs,zs,zs,ld,zs)
    dat = BlockVcat([-1.,0.,0.,0.,0.,1.]', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (2,0), (1,0))
    return A
end

# Identity ASp -> Sd
function \(Sd::SumSpaceD, ASp::AppendedSumSpace)
    Sd.I != ASp.I && error("Sum spaces bases not centred on same element")
    T = promote_type(eltype(ASp), eltype(Sd))
    
    # This should work but it hangs when attempting BlockHcat
    # Bm = Sd \ SumSpace{1,Vector{T},T}(ASp.I)   
    # Id = Eye((axes(Bm,1),))
    # B = BlockVcat(ASp.C[4][1], mortar(Zeros.(Fill(2,∞))))
    # Id = BlockHcat(B, Id)
    # B = BlockVcat(ASp.C[3][1], mortar(Zeros.(Fill(2,∞))))
    # Id = BlockHcat(B, Id)
    # B = BlockVcat(ASp.C[2][1], mortar(Zeros.(Fill(2,∞))))
    # Id = BlockHcat(B, Id)
    # B = BlockVcat(ASp.C[1][1], mortar(Zeros.(Fill(2,∞))))
    # Id = BlockHcat(B, Id)
    # return Bm * BId


    # FIXME: Temporary hack in finite-dimensional indexing
    N = Int64(5e2)
    Bm = (Sd \ SumSpaceP{T}(ASp.I))[1:2N+7,1:2N+3]    
    B = BlockBroadcastArray(hcat, ASp.C[1][1],ASp.C[2][1],ASp.C[3][1],ASp.C[4][1])[1:end,1:end]
    zs = Zeros(∞,4)
    B = vcat(B, zs)
    Id = hcat(B[1:2N+3,:], I[1:2N+3,1:2N+3])

    Id = [Id[:,5] Id[:,1:4] Id[:,6:end]] # permute T0 column to start
    A  = Bm * Id
    return A
end

# Derivative Sp -> Sd
@simplify function *(D::Derivative, Sp::SumSpaceP)
    T = eltype(Sp)
    (a,b) = Sp.I

    fracvec = mortar(Fill.(1. /(b-a),Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    ld = Diagonal(((-1).^(1:∞)) .* (2 .* ((1:∞) .÷ 2))[2:∞] )*fracvec

    dat = BlockBroadcastArray(hcat,zs,ld)
    dat = BlockVcat(Fill(0,2)', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,0), (0,0))
    return ApplyQuasiMatrix(*, SumSpaceD{T}(Sp.I), A)
end

# Hilbert: Sp -> Sp 
@simplify function *(H::Hilbert, Sp::SumSpaceP)
    T = eltype(Sp)
    onevec = mortar(Fill.(convert(T, π), Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    dat = BlockBroadcastArray(hcat,-onevec,zs,onevec)
    dat = BlockVcat(Fill(0,3)', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (0,0), (1,1))
    return ApplyQuasiMatrix(*, SumSpaceP{T}(Sp.I), A)
end

# x Sp -> Sp
function jacobimatrix(Sp::SumSpaceP)
    T = eltype(Sp)
    halfvec = mortar(Fill.(1/2,Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    dat = BlockBroadcastArray(hcat,halfvec,zs,zs,zs,halfvec,zs)
    dat = BlockVcat([0,0,0,0,0,1/2]', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,1), (1,0))'
    return A
end