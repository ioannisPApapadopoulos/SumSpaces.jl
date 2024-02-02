module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles, HypergeometricFunctions, BandedMatrices

import ClassicalOrthogonalPolynomials: ∞, Derivative, jacobimatrix, @simplify, HalfLine, Weight, orthogonalityweight, recurrencecoefficients
import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf
import BlockArrays: block, blockindex, Block, _BlockedUnitRange#, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LazyBandedMatrices: Tridiagonal
import LazyArrays: LazyVector

const ConvKernel{T,D1,V,D2} = BroadcastQuasiMatrix{T,typeof(-),Tuple{D1,QuasiAdjoint{V,Inclusion{V,D2}}}}
const Hilbert{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,Inclusion{T,D1},T,D2}}}
sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)

include("chebyshev/extendedchebyshev.jl")
include("chebyshev/sumspace.jl")
include("chebyshev/element-sumspace.jl")
include("jacobi/extendedjacobi.jl")
include("jacobi/sumspace-jacobi.jl")
include("frame.jl")
include("cft/cft.jl")
include("interlace/sumspace.jl")
include("laguerre/hilbertlaguerre.jl")
include("laguerre/sumspace.jl")
include("hermite/extendedhermite.jl")

export  ∞, oneto, Block, Derivative, Hilbert, BlockArray, Fill, Weighted,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace, jacobimatrix,
        ElementSumSpace, ElementSumSpaceP, ElementSumSpaceD, ElementAppendedSumSpace, coefficient_interlace, coefficient_stack,
        ExtendedJacobi, ExtendedWeightedJacobi, SumSpaceJacobi, SumSpaceJacobiP, SumSpaceJacobiD,
        ExtendedWeightedLaguerre, HilbertWeightedLaguerre,
        ExtendedHermite,
        SumSpaceL, SumSpaceLD,
        solvesvd, collocation_points, riemann, evaluate, framematrix, riemannf, riemannT, riemannTf,
        supporter_functions, fft_supporter_functions, interpolate_supporter_functions, coefficient_supporter_functions, inverse_fourier_transform, load_supporter_functions, save_supporter_functions


# Affine transform to scale and shift polys. 
affinetransform(a,b,x) = 2 /(b-a) * (x-(a+b)/2)

struct Interlace{T,AA,BB} <: LazyVector{T}
    a::AA
    b::BB
end

Interlace{T}(a::AbstractVector{T}, b::AbstractVector{T}) where T = Interlace{T,typeof(a),typeof(b)}(a,b)
Interlace(a::AbstractVector{T}, b::AbstractVector{V}) where {T,V} = Interlace{promote_type(T,V)}(a, b)

size(::Interlace) = (ℵ₀,)

getindex(A::Interlace{T}, k::Int) where T = convert(T, isodd(k) ? A.a[(k+1)÷2] : A.b[k÷2])::T

end # module