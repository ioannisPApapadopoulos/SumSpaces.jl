module SumSpaces

using SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, DomainSets,
    FillArrays, LazyBandedMatrices, LazyArrays, FFTW, Interpolations, InfiniteArrays,
    QuasiArrays, MathLink, DelimitedFiles

# Affine transform to scale and shift polys. 
function affinetransform(a,b,x)
    y = 2 ./(b.-a) .* (x.-(a.+b)./2)
end

end # module
