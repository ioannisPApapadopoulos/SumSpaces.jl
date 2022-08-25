using SumSpaces, LinearAlgebra
using LaTeXStrings, Plots
using DelimitedFiles

# For almost-machine precision convergence, we require Mathematica
# routines provided by SumSpacesMathLink to compute the support functions

using SumSpacesMathLink

"""
This script implements the "Discontinuous right-hand side" example found in

"A sparse spectral method for a one-dimensional fractional PDE problem" by I. P. A. Papadopoulos and S. Olver. 

We first approximate the error of a discontious function f(x) = 1 if |x| ≤ 1, 0 otherwise, as expanded
in the sum space and dual sum space. 

Next we approximately solve the equation:

((-Δ)^1/2 + I) u(x) = f(x), u(x) → 0 as |x| → ∞. 
"""

function riemann_2_norm(diff, h)
    diff2 = diff.^2
    out = h*( (diff2[1]^2 + diff2[end]^2)/2 + sum(diff2[2:end-1].^2) )
    return sqrt(out)
end

fa = x -> abs(x) < 1 ? 1 : (abs(x)==1 ? 1/2 : 0)

intervals = [-5,-3,-1.,1,3,5]; 
eSp = ElementSumSpaceP(intervals); 
eSd = ElementSumSpaceD(intervals);

### Collect errors
Nn = [3,5,7,11,15,21,31,41,51,61,71,81,91,101]
errors1 = []; errors2 = [];
xx1 = -5:0.01:5; xx2 = -5:0.0001:5
for N in Nn
    M = max(N^2,6001)
    x = collocation_points(M, M, I=intervals, endpoints=[-10,10]) # Collocation points
    A = framematrix(x, eSp, N, normtype=evaluate) 
    f = Matrix(A) \ fa.(x)

    ff1 = eSp[xx1,1:length(f)]*f; ff2 = eSp[xx2,1:length(f)]*f;
    diff1 = abs.(fa.(xx1).-ff1); diff2 = abs.(fa.(xx2).-ff2); 
    append!(errors1, riemann_2_norm(diff1, step(xx1))) 
    append!(errors2, riemann_2_norm(diff2, step(xx2))) 
    # writedlm("errors-rhs-jump-primal-1e-2.txt", errors1)
    # writedlm("errors-rhs-jump-primal-1e-4.txt", errors2)
end

errors3 = []; errors4 = [];
for N in Nn
    M = max(N^2,6001);
    xdual = collocation_points(M, M, I=intervals, endpoints=[-10,10], innergap=1e-2)
    Adual = framematrix(xdual, eSd, N, normtype=evaluate)
    fdual = Matrix(Adual) \ fa.(xdual)
    ffd1 = eSd[xx1,1:length(fdual)]*fdual; ffd2 = eSd[xx2,1:length(fdual)]*fdual
    diff1 = abs.(fa.(xx1).-ffd1); diff2 = abs.(fa.(xx2).-ffd2); 
    diff1[isnan.(ffd1)] .= 0; diff2[isnan.(ffd2)] .= 0;
    append!(errors3, riemann_2_norm(diff1, step(xx1))) 
    append!(errors4, riemann_2_norm(diff2, step(xx2))) 
    # writedlm("errors-rhs-jump-dual-1e-2.txt", errors3)
    # writedlm("errors-rhs-jump-dual-1e-4.txt", errors4)
end

# errors1 = readdlm("errors-rhs-jump-primal-1e-2.txt")
# errors2 = readdlm("errors-rhs-jump-primal-1e-4.txt")
# errors3 = readdlm("errors-rhs-jump-dual-1e-2.txt")
# errors4 = readdlm("errors-rhs-jump-dual-1e-4.txt")

plot(Nn, errors1,
    title="Error",
    legend=:bottomleft,
    markers=:circle,
    xlabel=L"$n$",
    ylabel=L"$L^2\mathrm{-norm \;\; error}$",
    yscale=:log10,
    # ylim=[1e-25,1e-3],
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    marker=:dot,
    markersize=5,
    yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25],
    label=L"Sum space $h=10^{-2}$")
plot!(Nn, errors2, title="Error",markers=:circle,xlabel=L"$n$",ylabel=L"$L^2\mathrm{-norm \;\; error}$", yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,linewidth=2,marker=:dot,markersize=5,label=L"Sum space $h=10^{-4}$")
plot!(Nn, errors3, title="Error",markers=:circle,xlabel=L"$n$",ylabel=L"$L^2\mathrm{-norm \;\; error}$", yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,linewidth=2,marker=:dot,markersize=5,label=L"Dual sum space $h=10^{-2}$")
plot!(Nn, errors4, title="Error",markers=:circle,xlabel=L"$n$",ylabel=L"$L^2\mathrm{-norm \;\; error}$", yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,linewidth=2,marker=:dot,markersize=5,label=L"Dual sum space $h=10^{-4}$")

# savefig("errors-rhs-infty2.pdf")


###
# Solve a fractional PDE
###

N = 41
λ = 1; μ = 0; η = 0 # Constants
K = length(intervals)-1

# Compute/load support functions
# Uncomment these two lines if support function already computed and saved.

# filepath = "uS-lmbda-$λ-mu-$μ-eta-$η/uS-N-$N.txt"
# uS = load_supporter_functions(filepath, intervals);

# Comment this out if support functions already computed
uS = fft_mathematica_supporter_functions(λ, μ, η, I=intervals, N=N, W=1e4, δ=1e-2, stabilise=true, maxrecursion=100)

# Element primal sum space coefficients
M = max(N^2,6001)
xc = collocation_points(M, M, I=intervals, endpoints=[-10,10]) # Collocation points
A = framematrix(xc, eSp, N, normtype=evaluate) 
cuS = coefficient_supporter_functions(A, xc, uS, 2N+3) 

# Create appended sum space
ASp = ElementAppendedSumSpace(uS, cuS, intervals)

 # Create global identity matrix ASp -> eSd
Id = (eSd \ ASp)[1:1+K*(2N+6),1:1+K*(2N+6)]

x = axes(eSp, 1); H = inv.( x .- x')
Hm = [(1/π).*( (eSp \ (H * eSp)[j] )[1:2N+3,1:2N+3]) for j in 1:K]    # Hilbert: Sp -> Sp
Cm = [(eSd \ (Derivative(x) * eSp)[j])[1:2N+7,1:2N+3] for j in 1:K]   # Derivative: Sp -> Sd
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

f = Matrix(A) \ fa.(xc) # Expand RHS in sum space 
fA = [f[1]' zeros(K*4)' f[2:end]']' # Add in zeros for support function positions
fA = Id*fA # Map to expansion in dual sum space
fA = BlockArray(fA, vcat(1,Fill(K,(length(fA)-1)÷K))) # Block structure for solve

# Rearrange coefficients element-wise
fA = coefficient_stack(fA, N, K)
for j = 1:K
    # Solve for each element seperately and append to form global
    # vector of coefficients
    append!(u, Dm[j] \ fA[Block.(j)])
end
# Rearrange coefficients back to interlaced
u = coefficient_interlace(u, N, K)
fA = coefficient_interlace(fA, N, K)

###
# Plot the solution
###
xx = Array(-5.:0.01:5)
yy = ASp[xx,1:length(u)]*u

p = plot(xx, yy, ylabel=L"$y$", xlabel=L"$x$", title="Numerical Solution", ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,legend=:topleft, label=:none)
# savefig(p, "example-rhs-jump.pdf")