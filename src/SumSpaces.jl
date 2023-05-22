module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, DelimitedFiles, HypergeometricFunctions, BandedMatrices, MultivariateOrthogonalPolynomials,
    SingularIntegrals

import ClassicalOrthogonalPolynomials: ∞, Derivative, jacobimatrix, @simplify, HalfLine, Weight, orthogonalityweight, recurrencecoefficients
import SingularIntegrals: sqrtx2, Hilbert, RecurrenceArray
import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf
import BlockArrays: block, blockindex, Block, _BlockedUnitRange#, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LazyBandedMatrices: Tridiagonal
import SemiclassicalOrthogonalPolynomials: Interlace
import HarmonicOrthogonalPolynomials: AbsLaplacianPower
import HypergeometricFunctions: _₂F₁general2

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
include("laguerre/symmetriclaguerre.jl")
include("hermite/extendedhermite.jl")
include("zernike/extendedzernike.jl")

export  ∞, oneto, Block, Derivative, Hilbert, BlockArray, Fill, Weighted, AbsLaplacianPower,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace, jacobimatrix,
        ElementSumSpace, ElementSumSpaceP, ElementSumSpaceD, ElementAppendedSumSpace, coefficient_interlace, coefficient_stack,
        ExtendedJacobi, ExtendedWeightedJacobi, DerivativeExtendedJacobi, DerivativeExtendedWeightedJacobi, GeneralExtendedJacobi, 
        SumSpaceJacobi, SumSpaceJacobiP, SumSpaceJacobiD,
        ExtendedWeightedLaguerre, HilbertWeightedLaguerre,
        ExtendedHermite, ExtendedNormalizedHermite,
        SymmetricLaguerre, ExtendedSymmetricLaguerre, SymmetricLaguerreWeight,
        ExtendedZernike, ExtendedWeightedZernike,
        SumSpaceL, SumSpaceLD,
        solvesvd, collocation_points, riemann, evaluate, framematrix, riemannf, riemannT, riemannTf,
        supporter_functions, fft_supporter_functions, interpolate_supporter_functions, coefficient_supporter_functions, inverse_fourier_transform, load_supporter_functions, save_supporter_functions


# Affine transform to scale and shift polys. 
affinetransform(a,b,x) = 2 /(b-a) * (x-(a+b)/2)

end # module
