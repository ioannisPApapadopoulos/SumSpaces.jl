using SumSpaces
using LinearAlgebra, Interpolations
using PyPlot
using DelimitedFiles, LaTeXStrings, FFTW

"""
This script implements the "Wave propagation" example found in

"A sparse spectral method for a one-dimensional fractional PDE problem" by I. P. A. Papadopoulos and S. Olver. 

We want to solve 

((-Δ)^1/2 + H + ∂ₜₜ) u(x,t) = W₄(x)exp(-t²), u(x,t) → 0 as |x| → ∞.

We first approximate the frequency-space solutions for a variety of wavelengths and then compute an IFT to 
obtain the physical-space solution. 

I.e. we first solve

((-Δ)^1/2 + H - ω²) ̂u(x,ω) = √(π)W₄(x)exp(-ω²/4), ̂u(x,ω) → 0 as |x| → ∞,

for a variety of values of ω. Then we approximate the IFT of ̂u(x,ω) via an FFT.
"""

N = 7;         # Truncation degree
μ = 1; η = 0;  # Constants for Hilbert and derivative terms, respectively
# RHS in frequency-space
fxλ = (x, λ) -> (sqrt(π)*exp(λ/4))*ExtendedWeightedChebyshevU()[x,5]

# To store solutions
solns = []
yylist = []


intervals = [-5,-3,-1.,1,3,5]; K = length(intervals)-1
eSp = ElementSumSpaceP(intervals)
eSd = ElementSumSpaceD(intervals)
M = max(N^2,6001)  # Number of collocation points
xc = collocation_points(M, M, I=intervals, endpoints=[-25,25]) # Collocation points
A = framematrix(xc, eSp, N, normtype=evaluate) 

for λ in 0:-0.1:-20

    fa = x -> fxλ(x, λ)
    f = Matrix(A) \ fa.(xc) # RHS expansion in sum space

    # Compute support functions approximated via FFT
    uS = fft_supporter_functions(λ, μ, η, I=intervals, N=N, W=1e3, δ=1e-3, stabilise=true)
    # Expand support functions in sum space
    cuS = coefficient_supporter_functions(A, xc, uS, 2N+3, normtype=evaluate) 
    
    # Create appended sum space
    ASp = ElementAppendedSumSpace(uS, cuS, intervals)

    # Create global identity matrix ASp -> eSd
    Id = (eSd \ ASp)[1:1+K*(2N+6),1:1+K*(2N+6)]

    x = axes(eSp, 1); H = inv.( x .- x')
    Hm = [(1/π).*(( eSp \ (H*eSp)[j] )[1:2N+3,1:2N+3]) for j in 1:K]   # Hilbert: Sp -> Sp
    Cm = [(eSd \ (Derivative(x)*eSp)[j] )[1:2N+7,1:2N+3] for j in 1:K] # Derivative: Sp -> Sd
    Bm = [(eSd\eSp)[j][1:2N+7,1:2N+3] for j in 1:K]                    # Identity: Sp -> Sd


    Dm =  [λ.*Bm[j] + μ.*Bm[j]*Hm[j] + η.*Cm[j] + Cm[j]*Hm[j] for j in 1:K]     # Helmholtz-like operator: Sp -> Sd   
    Dm = [hcat(zeros(size(Dm[j],1), 4),Dm[j]) for j in 1:K]         # Adding 4 columns to construct: ASp -> Sd
    
    if λ == 0
        for j in 1:K
            Dm[j][2:3,1:2] = I[1:2,1:2]; Dm[j][end-1:end,3:4] = I[1:2,1:2]
            Dm[j] = [Dm[j][:,1:4] Dm[j][:,6:end]]
            Dm[j] = Dm[j][2:end,:] 
        end
    else
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
    end
    
    u = [] # To store solution
    # Add zero coefficients corresponding to support functions
    fd = [f[1]' zeros(K*4)' f[2:end]']' 
    # Map expansion of RHS to expansion in dual sum space
    fd = Id*fd
    # Make into correct BlockArray structure
    fd = BlockArray(fd, vcat(1,Fill(K,(length(fd)-1)÷K)))
    # Rearrange coefficients element-wise
    fd = coefficient_stack(fd, N, K)

    if λ == 0
        for j = 1:K
            if j == 1
                fd_t = fd[Block.(j)][2:end]
            else
                fd_t = fd[Block.(j)]
            end
            # Solve for each element seperately and append to form global
            # vector of coefficients
            u_t = Dm[j]\fd_t
            if j==1
                u_t = vcat(zeros(1),u_t)
            end
            append!(u, u_t)
        
        end
    else
        for j = 1:K
            # Solve for each element seperately and append to form global
            # vector of coefficients
            append!(u, Dm[j]\fd[Block.(j)])
        end
    end

    # Rearrange coefficients back to interlaced
    u = coefficient_interlace(u, N, K)
    fd = coefficient_interlace(fd[1:end],N, K)
    
    append!(solns, [u])
    writedlm("wave-propogation-solns.txt", solns)

    xx = -10:0.01:10
    yy = ASp[xx,1:length(u)]*u
    append!(yylist, [yy])
    writedlm("wave-propogation-y.txt", yylist)

end  

yylist = readdlm("wave-propogation-y.txt")

###
# Contour plot of frequency space solution
###
xx = -10:0.01:10; ω2 = 0:0.1:20
uaa(x,ω) = yylist[findall(x->x==ω,ω2)[1],findall(y->y==x,xx)[1]]
X = repeat(reshape(xx, 1, :), length(ω2), 1)
Ω2 = repeat(ω2, 1, length(xx))
Z = map(uaa, X, Ω2)
PyPlot.rc("font", family="serif", size=14)
rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
rcParams["text.usetex"] = true
figure()
p = contourf(xx,sqrt.(ω2),Z,
        levels=100,
        cmap=get_cmap("bwr"),
        vmin=-findmax(abs.(Z))[1],
        vmax=findmax(abs.(Z))[1]
)
xlabel(L"$x$", fontsize=15, fontname="serif")
ylabel(latexstring(L"$\omega$"), fontsize=15, fontname="serif")
PyPlot.title(latexstring(L"$\hat{u}(x,\omega)$"),fontsize=18)
colorbar()
gca().grid(false)
for c in p.collections
    c.set_edgecolor("face")
end
display(gcf())
savefig("wave-propogation-hilbert-contour-py.pdf")

###
# Approximate inverse Fourier transform
###

yylist_ifft = []
ω = -1000:0.01:1000; ω = ω[1:end-1]
for step in 1:length(xx)
    Fu = extrapolate(interpolate((sqrt.(ω2),), yylist[:,step], Gridded(Linear())), 0)
    FFu = ω -> Fu(abs.(ω))
    (t, u) = inverse_fourier_transform(FFu, ω)
    append!(yylist_ifft, [u])
end

###
# Contour plot of physical space solution
###

τ = ifftshift(fftfreq(length(ω), 1/step(ω)) * 2 * pi)
tt = τ[findall(x->x==0,τ)[1]:findall(x->x==0,τ)[1]+2000]
uifft(x,t) = real.(yylist_ifft[findall(y->y==x,xx)[1]][findall(x->x==t,τ)[1]])
X = repeat(reshape(xx, 1, :), length(tt), 1)
T = repeat(tt, 1, length(xx))
Z = map(uifft, X, T)

figure()
p = contourf(xx,tt,Z,
        levels=100,
        cmap=get_cmap("bwr"),
        vmin=-findmax(abs.(Z))[1],
        vmax=findmax(abs.(Z))[1]
)
xlabel(L"$x$", fontsize=15, fontname="serif")
ylabel(latexstring(L"t"), fontsize=15, fontname="serif")
PyPlot.title(latexstring(L"$u(x,t)$"),fontsize=18)
colorbar()
gca().grid(false)
for c in p.collections
    c.set_edgecolor("face")
end
display(gcf())

savefig("wave-propogation-hilbert-contour-ifft-0-py.pdf")