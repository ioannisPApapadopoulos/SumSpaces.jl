module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles

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
# include("cft/mathematica.jl")

export  ∞, oneto, Block, Derivative, Hilbert,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace, jacobimatrix,
        ElementSumSpace, ElementSumSpaceP, ElementSumSpaceD, ElementAppendedSumSpace,
        ExtendedJacobi, ExtendedWeightedJacobi,
        solvesvd, collocation_points, riemann, evaluate, framematrix,
        supporter_functions, fft_supporter_functions, interpolate_supporter_functions, coefficient_supporter_functions, inverse_fourier_transform, fractional_heat_fourier_solve,
        fft_mathematica_supporter_functions, mathematica_correction 


# Affine transform to scale and shift polys. 
affinetransform(a,b,x) = 2 /(b-a) * (x-(a+b)/2)

end # module
