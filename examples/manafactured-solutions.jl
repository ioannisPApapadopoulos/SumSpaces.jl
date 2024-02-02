using SumSpaces
using LinearAlgebra, SpecialFunctions, HypergeometricFunctions
using DelimitedFiles, LaTeXStrings, Plots

# For almost-machine precision convergence, we require Mathematica
# routines provided by SumSpacesMathLink to compute the support functions

using SumSpacesMathLink

"""
This script implements the "Manafactured solutions (second case)" example found in

"A sparse spectral method for fractional differential equations in one-spatial dimension" by I. P. A. Papadopoulos and S. Olver. 

We want to solve 

(I + d/dx + H + (-Δ)^1/2) u(x) = (1-2x)exp(-x²) + exp(-x²)|x| erfi(|x|)/ x + 2 ₁F₁(1;1/2;-x²)/√π.

This has the exact solution u(x) = exp(-x²). 

We measure the inf-norm error with increasing n and plot the (spectral) convergence.

"""

# To contain the inf-norm errors at each time-step
errors = []

# To compute the first case, simply set μ=0 and η=0.
λ = 1; μ = 1; η = 1 # Constants

intervals = [-5,-3,-1.,1,3,5]
K = length(intervals)-1
eSp = ElementSumSpaceP(intervals)
eSd = ElementSumSpaceD(intervals)

# Actual solution
ua = x -> exp(-x^2)

# RHS function
tfa = x -> ((λ - 2η*x) * exp(-x^2) 
            + μ * exp(-x^2) * abs(x) * erfi(abs(x)) / x
            + 2/sqrt(π) * _₁F₁(1,1/2,-x^2)
)
# Approximate RHS actual value at 0 (otherwise we get an NaN)
fa = x -> x ≈ 0 ? ( tfa(-eps()) + tfa(eps()) ) / 2 : tfa(x)

M = 6001  # Number of collocation points.
xc = collocation_points(M, M, I=intervals, endpoints=[-25,25]) # Collocation points


for N in [3,5,7,11,13,15,21,25,31,41] 
    
    A = framematrix(xc, eSp, N, normtype=evaluate) 
    f = Matrix(A) \ fa.(xc)

    # Compute/load support functions
    # Uncomment these two lines if support function already computed and saved.
    
    # filepath = "uS-lmbda-$λ-mu-$μ-eta-$η/uS-N-$N.txt"
    # uS = load_supporter_functions(filepath, intervals);

    # Comment this out if support functions already computed
    uS = fft_mathematica_supporter_functions(λ, μ, η, I=intervals, N=N, W=1e4, δ=1e-2, stabilise=true, maxrecursion=100)
    
    # Element primal sum space coefficients
    cuS = coefficient_supporter_functions(A, xc, uS, 2N+3, normtype=evaluate, tol=1e-12) 
    # Create appended sum space
    ASp = ElementAppendedSumSpace(uS, cuS, intervals)

    # Create global identity matrix ASd -> eSd.
    Id = (eSd \ ASp)[1:1+K*(2N+6),1:1+K*(2N+6)]

    x = axes(eSp, 1); H = inv.( x .- x')
    Hm = [(1/π).*( (eSp \ (H * eSp)[j] )[1:2N+3,1:2N+3]) for j in 1:K]    # Hilbert: Sp -> Sp
    Cm = [(eSd \ (Derivative(x)*eSp)[j])[1:2N+7,1:2N+3] for j in 1:K]     # Derivative: Sp -> Sd
    Bm = [(eSd \ eSp)[j][1:2N+7,1:2N+3] for j in 1:K]                     # Identity: Sp -> Sd

    Dm =  [λ.*Bm[j] + μ.*Bm[j]*Hm[j] + η.*Cm[j] + Cm[j]*Hm[j] for j in 1:K]     # Helmholtz-like operator: Sp -> Sd   
    Dm = [hcat(zeros(size(Dm[j],1), 4),Dm[j]) for j in 1:K] # Adding 4 columns to construct: ASp -> Sd
    for j in 1:K
        Dm[j][2:3,1:2] = I[1:2,1:2]; Dm[j][end-1:end,3:4] = I[1:2,1:2]
        if j == 1
            # In first element permute T0 column to start
            Dm[j] = [Dm[j][:,5] Dm[j][:,1:4] Dm[j][:,6:end]] 
        else
            # In the rest delete the T0 column and row
            Dm[j] = [Dm[j][:,1:4] Dm[j][:,6:end]]
            Dm[j] = Dm[j][2:end,:] 
        end
    end

    u = []
    fd = [f[1]' zeros(K*4)' f[2:end]']'
    fd_Sd = Id*fd

    # Make into correct BlockArray structure
    fd_Sd = BlockArray(fd_Sd, vcat(1,Fill(K,(length(fd_Sd)-1)÷K)))
    # Rearrange coefficients element-wise
    fd_Sd = coefficient_stack(fd_Sd, N, K)

    for j = 1:K
        # Solve for each element seperately and append to form global
        # vector of coefficients
        append!(u, Dm[j]\fd_Sd[Block.(j)])
    end
    # Rearrange coefficients back to interlaced
    u = coefficient_interlace(u, N, K)

    xx = Array(-5.:0.01:5); 
    append!(errors, [[norm(ua.(xx) .- ASp[xx,1:length(u)]*u, Inf), N]])
    writedlm("errors-x-inf-ex2.txt", errors)
end

###
# Plot spectral convergence of error
###
# errors = readdlm("old_logs/errors-x-inf-ex2.txt")
N = Int32.(errors[:,2])
errors = errors[:,1]
plot(N, errors, #log10.(errors),
    title=L"$\mathrm{Error}$",
    legend=false,
    ylabel=L"$\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$n$",
    ylim=[1e-15, 1e-2],
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    marker=:dot,
    markersize=5
)
savefig("manafactured-solution-errors-ex2.pdf")