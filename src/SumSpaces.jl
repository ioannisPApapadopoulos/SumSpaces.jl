module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles

import ClassicalOrthogonalPolynomials: Hilbert, ∞, sqrtx2, Derivative, jacobimatrix
import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf
import BlockArrays: block, blockindex, Block, _BlockedUnitRange#, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix

include("extendedchebyshev.jl")
include("sumspace.jl")
include("element-sumspace.jl")

export  ∞, oneto, Block, Derivative, Hilbert,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace, jacobimatrix,
        ElementSumSpace, ElementSumSpaceP, ElementSumSpaceD, ElementAppendedSumSpace


# Affine transform to scale and shift polys. 
function affinetransform(a,b,x)
    y = 2 ./(b.-a) .* (x.-(a.+b)./2)
end

end # module
