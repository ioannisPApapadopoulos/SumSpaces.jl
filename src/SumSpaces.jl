module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, MathLink, DelimitedFiles

import ClassicalOrthogonalPolynomials: Hilbert, ∞
import Base: in, axes, getindex
import ContinuumArrays: Basis, AbstractQuasiArray
import InfiniteArrays: OneToInf

include("extendedchebyshev.jl")
include("sumspace.jl")

export ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
            SumSpace, SumSpaceP, SumSpaceD, AppendedSumSpace


# Affine transform to scale and shift polys. 
function affinetransform(a,b,x)
    y = 2 ./(b.-a) .* (x.-(a.+b)./2)
end

end # module
