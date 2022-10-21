
### 
# Element sum space
###

struct ElementSumSpace{kind,T,E} <: Basis{T} 
    I::E
end
ElementSumSpace{kind, T}(I::AbstractVector{T}) where {kind, T} = ElementSumSpace{kind,T,typeof(I)}(I)
ElementSumSpace{kind}(I::AbstractVector) where kind = ElementSumSpace{kind,Float64}(I)
ElementSumSpace{kind}() where kind = ElementSumSpace{kind}([-1.,1.])


const ElementSumSpaceP = ElementSumSpace{1}
const ElementSumSpaceD = ElementSumSpace{2}

axes(ES::ElementSumSpace) = (Inclusion(ℝ), _BlockedUnitRange(1:(length(ES.I)-1):∞))

==(a::ElementSumSpace{kind}, b::ElementSumSpace{kind}) where kind = a.I == b.I
==(a::ElementSumSpace, b::ElementSumSpace) = false

function getindex(ES::ElementSumSpaceP{T}, x::Real, j::Int) where T

    el_no = length(ES.I)-1
    if j == 1
        return SumSpaceP{T}()[x, 1]
    else
        ind = (j-2) ÷ el_no + 1                # Block number - 1
        i = isodd(ind) ? (ind ÷ 2)+1 : ind ÷ 2 # Poly/function order
        el = (j-1) - ((j-2) ÷ el_no)*el_no     # Element number

        y = affinetransform(ES.I[el],ES.I[el+1], x)
        if isodd(ind)
            return ExtendedWeightedChebyshevU{T}()[y, i]
        else
            return ExtendedChebyshevT{T}()[y, i+1]
        end

    end
end

function getindex(ES::ElementSumSpaceD{T}, x::Real, j::Int) where T

    el_no = length(ES.I)-1
    if j == 1
        y = affinetransform(ES.I[1],ES.I[2], x)
        return ExtendedChebyshevU{T}()[y, 1]
    else
        ind = (j-2) ÷ el_no + 1                # Block number - 1
        i = isodd(ind) ? (ind ÷ 2)+1 : ind ÷ 2 # Poly/function order
        el = (j-1) - ((j-2) ÷ el_no)*el_no     # Element number

        y = affinetransform(ES.I[el],ES.I[el+1], x)
        if isodd(ind)
            return ExtendedWeightedChebyshevT{T}()[y, i]
        else
            return ExtendedChebyshevU{T}()[y, i+1]
        end
    end
end


# function getindex(S::SumSpace{2, E, T}, x::Real, j::Int)::T where {E, T}
#     y = affinetransform(S.I[1],S.I[2], x)
#     isodd(j) && return ExtendedChebyshevU{T}()[y, (j ÷ 2)+1]
#     ExtendedWeightedChebyshevT{T}()[y, j ÷ 2]
# end

### 
# Element appended sum space
###

struct ElementAppendedSumSpace{T, E} <: Basis{T} 
    A
    C
    I::E
end
ElementAppendedSumSpace{T}(A, C, I::AbstractVector) where T = ElementAppendedSumSpace{T,typeof(I)}(A, C, I)
ElementAppendedSumSpace(A, C, I::AbstractVector) = ElementAppendedSumSpace{Float64}(A, C, I)
ElementAppendedSumSpace(A, C) = ElementAppendedSumSpace(A, C, [-1.,1.])

axes(ASp::ElementAppendedSumSpace) = (Inclusion(ℝ), _BlockedUnitRange(1:(length(ASp.I)-1):∞))


function getindex(ASp::ElementAppendedSumSpace{T}, x::Real, j::Int) where T    
    el_no = length(ASp.I)-1
    ind = (j-2) ÷ el_no + 1                    
    i = isodd(ind) ? (ind ÷ 2)-1 : (ind ÷ 2)-2   # Poly/function order
    el = (j-1) - ((j-2) ÷ el_no)*el_no         # Element number
    ind += 1                                   # Block number
    
    if j == 1
        return SumSpaceP{T}(ASp.I)[x,1]
    elseif 2<=ind<=5
        return convert(T, ASp.A[ind-1][el](x)[1])
    else
        y = affinetransform(ASp.I[el],ASp.I[el+1], x)
        if iseven(ind)
            return ExtendedWeightedChebyshevU{T}()[y, i]
        else
            return ExtendedChebyshevT{T}()[y, i+1]
        end
    end

end

###
# Helper functions
###

function coefficient_interlace(c, N, el_no)
    cskip = 2N+6
    v = zeros(length(c))
    v = BlockArray(v, vcat(1,Fill(el_no,(length(v)-1)÷el_no)))
    v[1] = c[1]
    for j in 2:cskip+1
        v[Block.(j)] = c[j:cskip:end] 
    end
    return v
end

function coefficient_stack(c, N, el_no)
    cskip = 2N+6
    v = zeros(length(c))
    v = BlockArray(v, vcat(cskip+1,Fill(cskip,el_no-1)))
    v[1] = c[1]
    for j in 2:cskip+1
        v[j:cskip:end] = c[Block.(j)]
    end
    return v
end

###
# Operators
###
"""
Since the linear system is block-diagonal, we solve element-wise. Hence all these
operators are constructed element-wise and thus are local operators except for the
identity map from ASp to Sd which is global. 
"""

# Identity Sp -> Sd
function \(Sd::ElementSumSpaceD, Sp::ElementSumSpaceP)
    Sd.I != Sp.I && error("Element sum spaces bases not centred on same elements")
    el_no = length(Sp.I) - 1

    A = SumSpaceD() \ SumSpaceP()
    return [A for j in 1:el_no]
end

# Hilbert: Sp -> Sp 
function *(H::Hilbert, Sp::ElementSumSpaceP)
    T = eltype(Sp)
    el_no = length(Sp.I) - 1

    onevec = mortar(Fill.(convert(T, π), Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    dat = BlockBroadcastArray(hcat,-onevec,zs,onevec)
    dat = BlockVcat(Fill(0,3)', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (0,0), (1,1))
    return [ApplyQuasiMatrix(*, ElementSumSpaceP{T}(Sp.I), A) for j in 1:el_no]
    # return A
end

# Derivative Sp -> Sd
function *(D::Derivative, Sp::ElementSumSpaceP)
    T = eltype(Sp)
    el_no = length(Sp.I) - 1

    zs = mortar(Zeros.(Fill(2,∞)))
    A = []
    for j = 1:el_no
        fracvec = mortar(Fill.(1. /(Sp.I[j+1]-Sp.I[j]),Fill(2,∞)))
        
        ld = Diagonal(((-1).^(1:∞)) .* (2 .* ((1:∞) .÷ 2))[2:∞] )*fracvec

        dat = BlockBroadcastArray(hcat,zs,ld)
        dat = BlockVcat(Fill(0,2)', dat)
        append!(A, [_BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (1,0), (0,0))])
    end
    return [ApplyQuasiMatrix(*, ElementSumSpaceD{T}(Sp.I), A[j]) for j in 1:el_no]
    # return A
end

# Identity ASp -> Sd
function \(Sd::ElementSumSpaceD, ASp::ElementAppendedSumSpace)
    Sd.I != ASp.I && error("Sum spaces bases not centred on same element")
    T = promote_type(eltype(ASp), eltype(Sd))
    el_no = length(ASp.I)-1

    # FIXME: Temporary hack in finite-dimensional indexing
    N = Int64(5e2)
    # Bm = (Sd \ ElementSumSpaceP{Vector{T},T}(ASp.I))[1:2N+7,1:2N+3] 
    zs = Zeros(∞,4*el_no) 
    A = []
    
   
    B = BlockBroadcastArray(hcat, ASp.C[1]...)[1:end,1:end]
    for j = 2:4
        # FIXME: Should be able to unroll, but it's not playing ball.
        for el = 1:el_no
            B = hcat(B, ASp.C[j][el][1:end])
        end
    end
     
    B = vcat(B[1:end,1:end], zs)
    Id = hcat(B[1:2N+3+(el_no-1)*(2N+2),:], I[1:2N+3+(el_no-1)*(2N+2),1:2N+3+(el_no-1)*(2N+2)])
    Id = [Id[:,4*el_no+1] Id[:,1:4*el_no] Id[:,4*el_no+2:end]] # permute T0 column to start
        # A  = append!(A, [Bm * Id])
    # end
    Bm = _Id_Sp_Sd(ASp)[1:2N+7+(el_no-1)*(2N+6),1:2N+3+(el_no-1)*(2N+2)]
    Bm[3*el_no+2,1] = 1.
    
    rows = [size(Bm,1)]; cols = vcat([1], Fill(el_no, (2*N+6)))
    A = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (sum(rows),sum(cols)))
    
    A[1:end,1:end] = Bm*Id


    return A
end

function _Id_Sp_Sd(ASp)
    T = eltype(ASp)
    el_no = length(ASp.I) - 1

    zs = mortar(Zeros.(Fill(2*el_no,∞)))
    fracvec = mortar(Fill.(one(T)/2,Fill(2*el_no,∞)))
    ld = Diagonal(((-1).^((0:∞).÷el_no)) )*fracvec
    dat = BlockBroadcastArray(hcat,zs,ld)
    
    # dat = BlockBroadcastArray(hcat,ld,zs,zs,zs,zs,zs,zs,zs,-ld,zs,zs,zs)
    dat = BlockBroadcastArray(hcat,ld,zs,-ld)
    # dat = BlockVcat([-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]', dat)
    dat = BlockVcat(T[-1, 0 , 0]', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (2,0), (0,0))   
    return A
end

# x Sp -> Sp
# function jacobimatrix(Sp::ElementSumSpaceP)
#     # FIXME: Temporary hack in finite-dimensional indexing
#     N = Int64(1e2)
#     el_no = length(ASp.I) - 1
#     J = jacobimatrix(SumSpaceP())[1:2N+3,1:2N+3]
#     J[2,1]=0

#     Jm = []

#     a = (Sp.I[1:end-1] + Sp.I[2:end]) ./ 2
#     for j = 1:(length(Sp.I)-1)
#         J[1,1] = a[j]
#         append!(Jm, [J[1:end,1:end]])
#     end

#     return [Jm for j in 1:el_no]
# end
