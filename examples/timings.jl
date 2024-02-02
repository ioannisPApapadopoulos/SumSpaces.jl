using SumSpaces
using LinearAlgebra, SpecialFunctions, HypergeometricFunctions

"""
This script measures the timings for

(1) Assembly of the least-squares matrix G for the expansion of the right-hand side;
(2) The SVD factorization of G;
(3) Assembly of the sparse matrix L;
(4) Solving the linear system for the solution coefficient vector with backslash.

as detailed in 

"A sparse spectral method for fractional differential equations in one-spatial dimension" by I. P. A. Papadopoulos and S. Olver.
"""

# To compute the first case, simply set μ=0 and η=0.
λ, μ, η = 1, 1, 1 # Constants
M = 6001  # Number of collocation points.
T = Float64

function assemble_matrix(N, K, eSp, eSd, x, H)
    Hm = [(1/π).*( (eSp \ (H * eSp)[j] )[1:2N+3,1:2N+3]) for j in 1:K]    # Hilbert: Sp -> Sp
    Cm = [(eSd \ (Derivative(x)*eSp)[j])[1:2N+7,1:2N+3] for j in 1:K]     # Derivative: Sp -> Sd
    Bm = [(eSd \ eSp)[j][1:2N+7,1:2N+3] for j in 1:K]                     # Identity: Sp -> Sd

    Lm =  [λ.*Bm[j] + μ.*Bm[j]*Hm[j] + η.*Cm[j] + Cm[j]*Hm[j] for j in 1:K]     # Helmholtz-like operator: Sp -> Sd   
    Lm = [hcat(zeros(size(Lm[j],1), 4),Lm[j]) for j in 1:K] # Adding 4 columns to construct: ASp -> Sd
    for j in 1:K
        Lm[j][2:3,1:2] = LinearAlgebra.I[1:2,1:2]; Lm[j][end-1:end,3:4] = LinearAlgebra.I[1:2,1:2]
        if j == 1
            # In first element permute T0 column to start
            Lm[j] = [Lm[j][:,5] Lm[j][:,1:4] Lm[j][:,6:end]] 
        else
            # In the rest delete the T0 column and row
            Lm[j] = [Lm[j][:,1:4] Lm[j][:,6:end]]
            Lm[j] = Lm[j][2:end,:] 
        end
    end
    return Lm
end

# RHS function
tfa = x -> ((λ - 2η*x) * exp(-x^2) 
            + μ * exp(-x^2) * abs(x) * erfi(abs(x)) / x
            + 2/sqrt(π) * _₁F₁(1,1/2,-x^2)
)
# Approximate RHS actual value at 0 (otherwise we get an NaN)
fa = x -> x ≈ 0 ? ( tfa(-eps()) + tfa(eps()) ) / 2 : tfa(x)

for (N, intervals) = zip([200, 20, 100], [[-1.,1], -20.0:2:20, -5.0:2:5])
    K = length(intervals)-1
    eSp = ElementSumSpaceP{T}(intervals)
    eSd = ElementSumSpaceD{T}(intervals)


    xc = collocation_points(M, M, I=intervals, endpoints=[-25,25], innergap=1e-5) # Collocation points


    @time A = framematrix(xc, eSd, N, normtype=evaluate);
    @time f = Matrix(A) \ fa.(xc);

    fd_Sd = f;
    fd_Sd = BlockArray(fd_Sd, vcat(1,Fill(K,(length(fd_Sd)-1)÷K)))
    fd_Sd = coefficient_stack(fd_Sd, N, K)

    x = axes(eSp, 1); H = inv.( x .- x');
    @time Lm = assemble_matrix(N, K, eSp, eSd, x, H);

    u = []

    @time for j = 1:K
        # Solve for each element seperately and append to form global
        # vector of coefficients
        append!(u, Lm[j]\fd_Sd[Block.(j)])
    end
end