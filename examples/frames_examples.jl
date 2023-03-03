using SumSpaces, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots


###
# Examples with extended/weighted Chebyshev polynomials
###
intervals = [-5.,-3,-1,1,3,5]
u0 = x -> 1. / (x^2 + 1) 

T = Float64
T̃ = ExtendedChebyshevT{T}()
V = ExtendedWeightedChebyshevT{T}()
Ũ = ExtendedChebyshevU{T}()
W = ExtendedWeightedChebyshevU{T}()

# Evalulation is SIGNIFICANTLY faster if we pass the type of the tuple
Sₚ = SumSpace{T, Tuple{typeof(T̃[:, 2:∞]), typeof(W)}}((T̃[:, 2:∞], W), intervals)

M = 5001; Me = 5001; Mn = 211
xc = collocation_points(M, Me, I=intervals, endpoints=[-20*one(T),20*one(T)], innergap=1e-4)

@time Aₚ = Matrix(Sₚ[xc, 1:Mn]);
u₀ = Aₚ \ u0.(xc)

plot(xc, Sₚ[xc, 1:Mn]*u₀)
plot!(xc, u0.(xc))

###
# Examples with sums of extended/weighted Jacobi
###

s = 1/3
T̃ = ExtendedJacobi{T}(-s,-s)
V = ExtendedWeightedJacobi{T}(-s,-s)
Ũ = ExtendedJacobi{T}(s, s)
W = ExtendedWeightedJacobi{T}(s,s)
P = T̃ + V
Q = W + Ũ


# Create sumspace of {T̃+V} ∪ {W+Ũ} ∪ {W}
S = SumSpace{T, Tuple{typeof(P), typeof(Q), typeof(W)}}((P, Q, W), intervals)

@time A = Matrix(S[xc, 1:Mn]);
u₀ = A \ u0.(xc)

plot(xc, S[xc, 1:Mn]*u₀)
plot!(xc, u0.(xc))