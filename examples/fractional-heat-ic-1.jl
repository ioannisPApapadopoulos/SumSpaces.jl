using SumSpaces
import LinearAlgebra: I, norm
using LaTeXStrings, Plots

"""
This script implements the "Fractional heat equation (first initial condition)" example found in

"A sparse spectral method for fractional differential equations in one-spatial dimension" by I. P. A. Papadopoulos and S. Olver. 

We want to solve 

((-Δ)^1/2 + ∂ₜ) u(x, t) = 0, u(x, 0) = 1/(1+x²), u(x, t) → 0 as |x| → ∞. 

This has the exact solution u(x, t) = (1+t)/(x² + (1+t)²).

We discretize in time with backward Euler and obtain

((-Δ)^1/2 + λ) vₖ₊₁(x) = λvₖ(x)

where λ = 1/timestep. We can solve this repeatedly. 

We measure the inf-norm error over time with the exact solution.

"""


N = 5 # Truncation degree
λ = 1e2; μ = 0; η = 0; Δt = 1/λ # Constants

intervals = [-5,-3,-1,1.,3,5] # 3 elements at [-3,-1] ∪ [-1,1] ∪ [1,3]
K = length(intervals)-1

eSp = ElementSumSpaceP(intervals) # Primal element sum space
eSd = ElementSumSpaceD(intervals) # Dual element sum space

M = 5001 # Number of collocation points inside intervals.
Me = M ÷ 10 + 1  # Number of collocation points outside intervals.
xc = collocation_points(M, Me, I=intervals, endpoints=[-20.,20]) # Collocation points

A = framematrix(xc, eSp, N, normtype=evaluate) # Blocked frame matrix

# Compute support functions
uS = fft_supporter_functions(λ, μ, η, I=intervals, W=1e4, δ=1e-2); # Actual functions
# Element primal sum space coefficients
cuS = coefficient_supporter_functions(A, xc, uS, 2N+3, normtype=evaluate) 

# Create appended sum space
ASp = ElementAppendedSumSpace(uS, cuS, intervals)

# Create matrix for element 1
Id = (eSd \ ASp)[1:1+K*(2N+6),1:1+K*(2N+6)]

x = axes(eSp, 1); H = inv.( x .- x')
Hm = [(1/π).*((eSp \ (H*eSp)[j])[1:2N+3,1:2N+3]) for j in 1:K]   # Hilbert: Sp -> Sp
Cm = [(eSd \ (Derivative(x)*eSp)[j])[1:2N+7,1:2N+3] for j in 1:K]# Derivative: Sp -> Sd
Bm = [(eSd \ eSp)[j][1:2N+7,1:2N+3] for j in 1:K]                # Identity: Sp -> Sd


Dm =  [λ.*Bm[j] + μ.*Bm[j]*Hm[j] + Cm[j]*Hm[j] for j in 1:K]     # Helmholtz-like operator: Sp -> Sd   
Dm = [hcat(zeros(size(Dm[j],1), 4),Dm[j]) for j in 1:K] # Adding 4 columns to construct: ASp -> Sd
for j in 1:K
    Dm[j][2:5,1:4] = I[1:4,1:4]
    if j == 1
        # In first element permute T0 column to start
        Dm[j] = [Dm[j][:,5] Dm[j][:,1:4] Dm[j][:,6:end]] 
    else
        # In the rest delete the T0 column and row
        Dm[j] = [Dm[j][:,1:4] Dm[j][:,6:end]]
        Dm[j] = Dm[j][2:end,:] 
    end
end

u0 = x -> 1. / ((x^2 + 1) )
A = framematrix(xc, eSp, N, normtype=riemann)
u₀ =  Matrix(A) \ riemannf(xc, u0) 

# Add zero coefficients corresponding to support functions
u₀ = [u₀[1]' zeros(K*4)' u₀[2:end]']' 
u = [u₀]

# Run solve loop for time-stepping
timesteps=100
for k = 1:timesteps
    u1 = []
    
    # Map from ASp to Sd
    v = Id * u[k]
    # Multiply RHS with λ
    v = λ.*v
    v = BlockArray(v, vcat(1,Fill(K,(length(v)-1)÷K)))
    
    # Rearrange coefficients element-wise
    v = coefficient_stack(v, N, K)

    for j = 1:K
        # Solve for each element seperately and append to form global
        # vector of coefficients
        append!(u1, Dm[j]\v[Block.(j)])
     end

    
     # Rearrange coefficients back to interlaced
    u1 = coefficient_interlace(u1, N, K)

    # Append solution to list for different time-steps
    append!(u,  [u1])
end


# Plot solution and collect errors
p = plot() 
xx = -20:0.01:20
xlim = [xx[1],xx[end]]; ylim = [-0.02,1]
y = (x,t) -> (1 + t) / ((x^2 + (1+t)^2))
d = (x,t,u) -> abs.(y.(x,t) .- ASp[x,1:length(u)]*u)
errors = []

for k = 1:timesteps+1
    t = Δt*(k-1)
    
    tdisplay = round(t, digits=2)
    yy = ASp[xx,1:length(u[k])]*u[k]
    
    dx = x->d(x,t,u[k])
    append!(errors, norm(dx(xx), Inf))

    # p = plot(xx,yy, title="time=$tdisplay (s)", label="Fractional heat equation", legend=:topleft, xlim=xlim, ylim=ylim)
    # p = plot!(xx, y.(xx, t), label="True solution", legend=:topleft, xlim=xlim, ylim=ylim)
    # display(p)
end


###
# Plot inf-norm error at each timestep
###
plot(1:length(errors), errors, legend=:none, 
    title="Error",
    markers=:circle,
    xlabel=L"$k$",
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    ylabel=L"$\Vert u(x,k\Delta t)-\mathbf{S}^{\mathbf{I},\!\!\!\!+}_5\!\!\!\!\!(x) \mathbf{u}_k)\Vert_\infty$")
# savefig("errors-infty.pdf")

###
# Plot solution snapshots at t=0, 0.5, and 1
###
xx = -10:0.01:10
xlim = [xx[1],xx[end]]; ylim = [-0.02, 1]
p = plot()
for k = [1,51,101]
    t = Δt*(k-1)
    
    tdisplay = round(t, digits=2)
    yy = ASp[xx,1:length(u[k])]*u[k]
    
    p = plot!(xx,yy,
            label=L"$\mathrm{time}=$"*"$tdisplay"*L"$\ \mathrm{(s)}$", 
            legendfontsize = 10, legend=:topleft, xlim=xlim, ylim=ylim,
            xlabel=L"$x$",
            xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
            ylabel=L"$\mathbf{S}^{\mathbf{I},\!\!\!\!+}_5\!\!\!\!\!(x) \mathbf{u}$")

    display(p)
end  
# savefig(p, "ic1.pdf")

###
# Plot spy plots of matrices
###
p = plot(spy(Dm[1], markersize=4,color=:darktest), 
        xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
        title= L"$\mathrm{Spy \ plot \ of} \ \lambda E + A^{I_1} \; (n=5)$")
# savefig(p, "spy-1.pdf")
p = plot(spy(Dm[2], markersize=4,color=:darktest), 
        xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
        title= L"$\mathrm{Spy \ plot \ of} \ \lambda E + A^{I_k}, k \geq 2 \; (n=5)$")
# savefig(p, "spy-2.pdf")

###
# Plot contour plot of solution
###
uaa(x,t) = ASp[x,1:length(u[1])]'*u[Int64(round(t/Δt))][1:end] 
x = -3:0.01:3; t = 0.01:0.01:1.01;
X = repeat(reshape(x, 1, :), length(t), 1)
T = repeat(t, 1, length(x))
Z = map(uaa, X, T)
p = contour(x,t,Z,fill=true, rev=true,
        xlabel=L"$x$",ylabel=L"$t$",
        xtickfontsize=8, ytickfontsize=8,xlabelfontsize=15,ylabelfontsize=15)
# savefig(p, "bc1-frac-heat-contour.pdf")