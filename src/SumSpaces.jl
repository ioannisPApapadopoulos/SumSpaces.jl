module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles, HypergeometricFunctions, BandedMatrices

import ClassicalOrthogonalPolynomials: Hilbert, ∞, sqrtx2, Derivative, jacobimatrix, @simplify, HalfLine, Weight, orthogonalityweight, recurrencecoefficients
import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf
import BlockArrays: block, blockindex, Block, _BlockedUnitRange#, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LazyBandedMatrices: Tridiagonal
import SemiclassicalOrthogonalPolynomials: Interlace

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

end # module
