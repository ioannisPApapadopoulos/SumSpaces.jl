## SumSpaces.jl

[![Build status (Github Actions)](https://github.com/ioannisPApapadopoulos/SumSpaces.jl/workflows/CI/badge.svg)](https://github.com/ioannisPApapadopoulos/SumSpaces.jl/actions)
[![codecov.io](http://codecov.io/github/ioannisPApapadopoulos/SumSpaces.jl/coverage.svg?branch=main)](http://codecov.io/github/ioannisPApapadopoulos/SumSpaces.jl?branch=main)


A Julia package for sum spaces.

This package contains ongoing research on sum spaces of extended 
and weighted orthogonal polynomials. 


# Sum spaces

We can construct the primal and dual sum spaces. 
The former consists of extended Chebyshev functions 
of the first kind and weighted Chebyshev polynomials of 
the second kind. The latter consists of extended Chebyshev functions 
of the second kind and weighted Chebyshev polynomials of 
the first kind.

```julia
julia> Sp = SumSpaceP() # Primal sum space
SumSpace{1, Vector{Float64}, Float64}

julia> Sp[0.1,1:5] # first 5 functions
5-element Vector{Float64}:
  1.0
  0.99498743710662
  0.1
  0.198997487421324
 -0.98
```
Note that apart from the first function. The sum space
functions are grouped in twos following the pattern
|wU_k, eT_k+1|. This is accessible as the columns of `Sp`
are blocked a la BlockArrays.jl:
```julia
julia> Sp[0.1, Block.(1:3)]
3-blocked 5-element BlockArrays.PseudoBlockVector{Float64, Vector{Float64}, Tuple{BlockArrays.BlockedUnitRange{StepRange{Int64, Int64}}}}:
  1.0
 ──────────────────
  0.99498743710662
  0.1
 ──────────────────
  0.198997487421324
 -0.98
```

# Operators

Quasimatrix operators can be deduced via the same API as
ClassicalOrthogonalPolynomials.jl. For example consider the ℵ₀×ℵ₀
matrix, E, that maps the primal sum space to the
dual sum space such that Sp = Sd*E:
```julia
julia> Sp = SumSpaceP(); Sd = SumSpaceD();
julia> Sd \ Sp
ℵ₀×ℵ₀-blocked ℵ₀×ℵ₀ BlockBandedMatrices.BandedBlockBandedMatrix{Float64, BlockArrays.PseudoBlockMatrix{Float64, LazyBandedMatrices.BlockHcat{Float64, Tuple{Vector{Float64}, LinearAlgebra.Adjoint{Float64, LazyBandedMatrices.BlockBroadcastArray{Float64, 2, typeof(hcat), Tuple{LazyArrays.BroadcastVector{Float64, typeof(*), Tuple{LazyArrays.BroadcastVector{Int64, typeof(^), Tuple{Int64, InfiniteArrays.InfUnitRange{Int64}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Zeros{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Zeros{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Zeros{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}, LazyArrays.BroadcastVector{Float64, typeof(*), Tuple{LazyArrays.BroadcastVector{Int64, typeof(^), Tuple{Int64, InfiniteArrays.InfUnitRange{Int64}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}}}, BlockArrays.BlockVector{Float64, FillArrays.Fill{FillArrays.Zeros{Float64, 1, Tuple{Base.OneTo{Int64}}}, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Tuple{BlockArrays.BlockedUnitRange{InfiniteArrays.InfStepRange{Int64, Int64}}}}}}}}}, Tuple{BlockArrays.BlockedUnitRange{StepRange{Int64, Int64}}, BlockArrays.BlockedUnitRange{LazyArrays.ApplyArray{Int64, 1, typeof(vcat), Tuple{StepRange{Int64, Int64}, InfiniteArrays.InfStepRange{Int64, Int64}}}}}}, BlockArrays.BlockedUnitRange{LazyArrays.ApplyArray{Int64, 1, typeof(vcat), Tuple{StepRange{Int64, Int64}, InfiniteArrays.InfStepRange{Int64, Int64}}}}}:
 -1.0  │    ⋅     ⋅   │    ⋅     ⋅   │   ⋅     ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
 ──────┼──────────────┼──────────────┼─────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────  …
  0.0  │   0.5    ⋅   │    ⋅     ⋅   │   ⋅     ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
  0.0  │   0.0  -0.5  │    ⋅     ⋅   │   ⋅     ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅ 
 ──────┼──────────────┼──────────────┼─────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────
  0.0  │   0.0    ⋅   │   0.5    ⋅   │   ⋅     ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
  1.0  │   0.0   0.0  │   0.0  -0.5  │   ⋅     ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
 ──────┼──────────────┼──────────────┼─────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────
   ⋅   │  -0.5    ⋅   │   0.0    ⋅   │  0.5    ⋅   │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅   …
   ⋅   │   0.0   0.5  │   0.0   0.0  │  0.0  -0.5  │   ⋅     ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
 ──────┼──────────────┼──────────────┼─────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────
   ⋅   │    ⋅     ⋅   │  -0.5    ⋅   │  0.0    ⋅   │  0.5    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
   ⋅   │    ⋅     ⋅   │   0.0   0.5  │  0.0   0.0  │  0.0  -0.5  │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅    ⋅   │   ⋅
 ──────┼──────────────┼──────────────┼─────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────
  ⋮                            ⋮                          ⋮                        ⋮                        ⋱
```

# Expansion

Finding the expansion of a function in the sum space is nontrivial.
We utilize the framework of "frames". We first construct a Vandermonde
matrix and then use a custom SVD solver to find the expansion.
```julia
julia> Sp = SumSpaceP(); N = 2; # Truncation degree
julia> M = max(N^2,5001);  # Number of collocation points in [-1,1]
julia> Me = M ÷ 10;  # Number of collocation points in [-2,-1) and (1,2].
julia> x = collocation_points(M, Me); # Collocation points
julia> A = framematrix(x, Sp, N, M, Me); # Blocked frame matrix
julia> solvesvd(A, evaluate(x, x->ExtendedChebyshevT()[x,2]))
4-blocked 7-element BlockVector{Float64}:
  9.71445146547012e-17  
 ───────────────────────
 -9.43689570931383e-16
  0.9999999999999991
 ───────────────────────
  8.326672684688674e-16
 -2.220446049250313e-16
 ───────────────────────
 -3.3306690738754696e-16
  2.636779683484747e-16
```