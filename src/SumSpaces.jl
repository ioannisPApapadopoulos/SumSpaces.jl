module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles, HypergeometricFunctions

import ClassicalOrthogonalPolynomials: Hilbert, ∞, sqrtx2, Derivative, jacobimatrix, @simplify
import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf
import BlockArrays: block, blockindex, Block, _BlockedUnitRange#, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix

include("chebyshev/extendedchebyshev.jl")
include("chebyshev/sumspace.jl")
include("chebyshev/element-sumspace.jl")
include("jacobi/extendedjacobi.jl")
include("frame.jl")
include("cft/cft.jl")

export  ∞, oneto, Block, Derivative, Hilbert, BlockArray, Fill,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace, jacobimatrix,
        ElementSumSpace, ElementSumSpaceP, ElementSumSpaceD, ElementAppendedSumSpace, coefficient_interlace, coefficient_stack,
        ExtendedJacobi, ExtendedWeightedJacobi,
        solvesvd, collocation_points, riemann, evaluate, framematrix, riemannf, riemannT, riemannTf,
        supporter_functions, fft_supporter_functions, interpolate_supporter_functions, coefficient_supporter_functions, inverse_fourier_transform, load_supporter_functions, save_supporter_functions


# Affine transform to scale and shift polys. 
affinetransform(a,b,x) = 2 /(b-a) * (x-(a+b)/2)

end # module
