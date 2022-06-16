
### 
# Element sum space
###

struct ElementSumSpace{kind,E,T} <: Basis{T} 
    I::E
end
ElementSumSpace{kind, T}(I::Vector{T}) where {kind, T} = ElementSumSpace{kind,Vector{T},T}(I::Vector{T})
ElementSumSpace{kind}(I::Vector{Float64}) where kind = ElementSumSpace{kind,Float64}(I::Vector{Float64})
ElementSumSpace{kind}() where kind = ElementSumSpace{kind}([-1.,1.])

function ElementSumSpaceP(a=[-1,1.])
    ElementSumSpace{1}(a)
end

function ElementSumSpaceD(a=[-1,1.])
    ElementSumSpace{2}(a)
end

axes(ES::ElementSumSpace) = (Inclusion(ℝ), _BlockedUnitRange(1:(length(ES.I)-1):∞))

function ==(a::ElementSumSpace{kind}, b::ElementSumSpace{kind}) where kind
    if a.I == b.I 
        return true
    else
        return false
    end
end
==(a::ElementSumSpace, b::ElementSumSpace) = false

function getindex(ES::ElementSumSpace{1, E, T}, x::Real, j::Int)::T where {E, T}

    el_no = length(ES.I)-1
    if j == 1
        return SumSpace{1}()[x, 1]
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

function getindex(ES::ElementSumSpace{2, E, T}, x::Real, j::Int)::T where {E, T}

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

struct ElementAppendedSumSpace{AA, CC,E, T} <: Basis{T} 
    A::AA
    C::CC
    I::E
end
ElementAppendedSumSpace{E,T}(A, C, I::Vector{Float64}) where {E,T} = ElementAppendedSumSpace{Any,Any,E,T}(A, C, I::Vector{Float64})
ElementAppendedSumSpace{E}(A, C, I::Vector{Float64}) where E = ElementAppendedSumSpace{E,Float64}(A, C, I::Vector{Float64})
ElementAppendedSumSpace(A, C, I::Vector{Float64}) = ElementAppendedSumSpace{Vector{Float64}}(A, C, I::Vector{Float64})
ElementAppendedSumSpace(A, C) = ElementAppendedSumSpace(A, C, [-1.,1.])

axes(ASp::ElementAppendedSumSpace) = (Inclusion(ℝ), _BlockedUnitRange(1:(length(ASp.I)-1):∞))


function getindex(ASp::ElementAppendedSumSpace{AA, CC, E, T}, x::Real, j::Int)::T where {AA, CC, E, T}
    
    el_no = length(ASp.I)-1
    ind = (j-2) ÷ el_no + 1                    
    i = isodd(ind) ? (ind ÷ 2)-1 : (ind ÷ 2)-2   # Poly/function order
    el = (j-1) - ((j-2) ÷ el_no)*el_no         # Element number
    ind += 1                                   # Block number
    
    if j == 1
        return SumSpace{1, E, T}(ASp.I)[x,1]
    elseif 2<=ind<=5
        return ASp.A[ind-1][el](x)[1]
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

function coefficient_interlace(c, N, el_no; appended=false)
    cskip = appended==true ? 2N+6 : 2N+2
    v = zeros(length(c))
    v = BlockArray(v, vcat(1,Fill(el_no,(length(v)-1)÷el_no)))
    v[1] = c[1]
    for j in 2:cskip+1
        v[Block.(j)] = c[j:cskip:end] 
    end
    return v
end

function coefficient_stack(c, N, el_no; appended=false)
    cskip = appended==true ? 2N+6 : 2N+2
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
function \(Sd::ElementSumSpace{2}, Sp::ElementSumSpace{1})
    Sd.I != Sp.I && error("Element sum spaces bases not centred on same elements")
    el_no = length(Sp.I) - 1

    A = SumSpace{2}() \ SumSpace{1}()
    return [A for j in 1:el_no]
end

# Hilbert: Sp -> Sp 
function *(H::Hilbert{<:Any,<:Any,<:Any}, Sp::ElementSumSpace{1})
    T = eltype(Sp)
    el_no = length(Sp.I) - 1

    onevec = mortar(Fill.(convert(T, π), Fill(2,∞)))
    zs = mortar(Zeros.(Fill(2,∞)))
    dat = BlockBroadcastArray(hcat,-onevec,zs,onevec)
    dat = BlockVcat(Fill(0,3)', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (0,0), (1,1))
    return [ApplyQuasiMatrix(*, ElementSumSpace{1,T}(Sp.I), A) for j in 1:el_no]
    # return A
end

# Derivative Sp -> Sd
function *(D::Derivative{<:Real}, Sp::ElementSumSpace{1})
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
    return [ApplyQuasiMatrix(*, ElementSumSpace{2,T}(Sp.I), A[j]) for j in 1:el_no]
    # return A
end

# Identity ASp -> Sd
function \(Sd::ElementSumSpace{2}, ASp::ElementAppendedSumSpace)
    Sd.I != ASp.I && error("Sum spaces bases not centred on same element")
    T = promote_type(eltype(ASp), eltype(Sd))
    el_no = length(ASp.I)-1

    # FIXME: Temporary hack in finite-dimensional indexing
    N = Int64(1e2)
    # Bm = (Sd \ ElementSumSpace{1,Vector{T},T}(ASp.I))[1:2N+7,1:2N+3] 
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
    fracvec = mortar(Fill.(0.5,Fill(2*el_no,∞)))
    ld = Diagonal(((-1).^((0:∞).÷el_no)) )*fracvec
    dat = BlockBroadcastArray(hcat,zs,ld)
    
    # dat = BlockBroadcastArray(hcat,ld,zs,zs,zs,zs,zs,zs,zs,-ld,zs,zs,zs)
    dat = BlockBroadcastArray(hcat,ld,zs,-ld)
    # dat = BlockVcat([-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]', dat)
    dat = BlockVcat([-1.,0.,0.]', dat)
    A = _BandedBlockBandedMatrix(dat', (axes(dat,1),axes(dat,1)), (2,0), (0,0))   
    return A
end

# x Sp -> Sp
# function jacobimatrix(Sp::ElementSumSpace{1})
#     # FIXME: Temporary hack in finite-dimensional indexing
#     N = Int64(1e2)
#     el_no = length(ASp.I) - 1
#     J = jacobimatrix(SumSpace{1}())[1:2N+3,1:2N+3]
#     J[2,1]=0

#     Jm = []

#     a = (Sp.I[1:end-1] + Sp.I[2:end]) ./ 2
#     for j = 1:(length(Sp.I)-1)
#         J[1,1] = a[j]
#         append!(Jm, [J[1:end,1:end]])
#     end

#     return [Jm for j in 1:el_no]
# end
