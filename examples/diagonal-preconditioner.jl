using SumSpaces, LinearAlgebra
using Plots

"""
This script implements the diagonal preconditioner described in Section 5.3 in

"A sparse spectral method for fractional differential equations in one-spatial dimension" by I. P. A. Papadopoulos and S. Olver. 

We want to check that κ(L) = O(n) and assemble a diagonal preconditioner such that κ(LP⁻¹) = C,

where C is independent of n.

"""
λ, μ, η = 1, 1, 1 # Constants

intervals = [-1, 1.]
K = length(intervals)-1

eSp = ElementSumSpaceP(intervals) # Primal element sum space
eSd = ElementSumSpaceD(intervals) # Dual element sum space

function assemble_matrix(N, K, eSp, eSd, x, H)
    Hm = [(1/π).*( (eSp \ (H * eSp)[j] )[1:2N+3,1:2N+3]) for j in 1:K]    # Hilbert: Sp -> Sp
    Cm = [(eSd \ (Derivative(x)*eSp)[j])[1:2N+7,1:2N+3] for j in 1:K]     # Derivative: Sp -> Sd
    Bm = [(eSd \ eSp)[j][1:2N+7,1:2N+3] for j in 1:K]                     # Identity: Sp -> Sd

    Dm =  [λ.*Bm[j] + μ.*Bm[j]*Hm[j] + η.*Cm[j] + Cm[j]*Hm[j] for j in 1:K]     # Helmholtz-like operator: Sp -> Sd   
    Dm = [hcat(zeros(size(Dm[j],1), 4),Dm[j]) for j in 1:K] # Adding 4 columns to construct: ASp -> Sd
    for j in 1:K
        Dm[j][2:3,1:2] = LinearAlgebra.I[1:2,1:2]; Dm[j][end-1:end,3:4] = LinearAlgebra.I[1:2,1:2]
        if j == 1
            # In first element permute T0 column to start
            Dm[j] = [Dm[j][:,5] Dm[j][:,1:4] Dm[j][:,6:end]] 
        else
            # In the rest delete the T0 column and row
            Dm[j] = [Dm[j][:,1:4] Dm[j][:,6:end]]
            Dm[j] = Dm[j][2:end,:] 
        end
    end
    return Dm
end

x = axes(eSp, 1); H = inv.( x .- x')
conds, pconds = [], []

Ns = [5, 10, 20, 40, 80, 100, 120, 160]
for N in Ns
    # Linear growth in condition number
    Dm = assemble_matrix(N, K, eSp, eSd, x, H)
    append!(conds, cond(Dm[1]))

    # Diagoanl preconditioner induces constant condition number
    P = Diagonal(vcat(ones(5), (Vector(2:(size(Dm[1],1)-4)).÷2)))
    pD = Dm[1] / P
    append!(pconds, cond(pD))
end

pconds
conds

Plots.plot(Ns, [conds pconds],
    linewidth=2,
    markershape=[:square :circle],
    ylabel="2-norm condition number",
    xlabel="n",
    label=["Unpreconditioned" "Preconditioned"])