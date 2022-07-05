function supporter_functions(λ::Number, μ::Number, η::Number; W::Real=1000., δ::Real=0.001, 
    s::AbstractVector=[1.0], N::Int=5, stabilise::Bool=false)
    
    ω=range(-W, W, step=δ)
    # ω=range(-W, W, length=2000000); δ = step(ω);
    ω = ω[1:end-1]

    # FIXME: Should probably find a nicer way around this
    if μ ≈ 0 && real(λ) < 0 && λ ∈ ω
        λ += 1e2*eps() * π
        @warn "μ ≈ 0, λ < 0, and λ ∈ Sample, we slightly perturb λ to avoid NaNs in the FFT computation in supporter_functions in cft.jl"
    end

    fmultiplier = k -> (λ - im*μ*sign(k) + im*η*k + abs(k))
    hfmultiplier = k -> (- λ*im*sign(k) - μ + η*abs(k) - im*k)

    # For certain values of λ, μ and η, fmultiplier can be equal to 0. This causes
    # NaNs in the ifft routine. To avoid this we watch if fmultiplier=0 and if it does
    # then we average the Fourier transform around 0 to get an approximate of the limit.
    
    # FIXME: These list comprehensions are slow, can we speed it up? 

    ## Define function F[wT0] / (λ - i*μ*sgn(k)+ iηk + abs(k))
    tFwT0 = (k,κ) -> pi * besselj(0, abs(k)) / fmultiplier(κ)
    if λ ≈ 0
        FwT0 = (k,κ) -> fmultiplier(k) ≈ 0 ? (tFwT0(k-eps(),κ-eps())+tFwT0(k+eps(),κ+eps()))/2 : tFwT0(k,κ)
    else
        FwT0 = (k,κ) -> tFwT0(k,κ)
    end
    
    # If stabilise is false, then we compute the supporter functions associated with ̃U₀ and V₁ (U0 and wT1)
    # otherwise if it's true, then we compute the support functions asscoiated with ̃Uₙ₊₁ and Vₙ₊₂. 

    ## Define function F[wT1] / (λ - i*μ*sgn(k)+ iηk + abs(k))
    if stabilise == false
        tFwT1 = (k, κ) -> -im * pi * besselj(1, k) / fmultiplier(κ)
    else
        tFwT1 = (k, κ) -> (-im)^(N+2) * pi * besselj(N+2, k) / fmultiplier(κ)
    end
    
    if λ ≈ 0
        FwT1 = (k, κ) -> fmultiplier(k) ≈ 0 ? (tFwT1(k-eps(),κ-eps())+tFwT1(k+eps(),κ+eps()))/2 : tFwT1(k,κ)
    else
        FwT1 = (k,κ) -> tFwT1(k,κ)
    end
    
    ## Define function F[U0] / (λ - i*μ*sgn(k)+ iηk + abs(k))
    if stabilise == false
        tFU0 = (k,κ) -> (pi * besselj(1, abs(k)) + 2 *sin(k)/k - 2 * sin(abs(k)) / abs(k)) / fmultiplier(κ)
    else
        tFU0 = (k,κ) -> ( (-im)^(N+2)*pi*besselj(N+2, k) ) / hfmultiplier(κ)
    end
    FU0 = (k,κ) -> k ≈ 0 ? (tFU0(k-eps(),κ-eps())+tFU0(k+eps(),κ+eps()))/2 : tFU0(k,κ)
    
    ## Define function F[U-1] / (λ - i*μ*sgn(k)+ iηk + abs(k))
    tFU_1 = (k,κ) -> ( im*pi*k*besselj(0,abs(k)) / abs(k)) / fmultiplier(κ)
    FU_1 = (k,κ) -> k ≈ 0 ? (tFU_1(k-eps(),κ-eps())+tFU_1(k+eps(),κ+eps()))/2 : tFU_1(k,κ)
    
    ## Different element supporter functions on the same sized elements are simply translations
    ## of the supporter functions on a reference element s*[-1,1] where s is the scaling.
    ## Hence, we only want to do 4 FFTs per differently sized element to compute
    ## the supporter functions on the reference elements s*[-1,1]. These are then translated in
    ## interpolate_supporter_functions. 
    
    # Independent variable in physical space
    x = ifftshift((fftfreq(length(ω), 1/δ)) * 2 * pi) 
    
    # Compute reference supporter functions
    ywT0 = []; ywT1 = []; yU0 = []; yU_1 = []
    for ss in s
        sFwT0 = k -> (1/ss)*FwT0((1/ss)*k, k)
        sFwT1 = k -> (1/ss)*FwT1((1/ss)*k, k)
        sFU0 = k -> (1/ss)*FU0((1/ss)*k, k)
        sFU_1 = k -> (1/ss)*FU_1((1/ss)*k, k)
        append!(ywT0, [cifft(sFwT0, ω, δ, W, x)])
        append!(ywT1, [cifft(sFwT1, ω, δ, W, x)])
        append!(yU0, [cifft(sFU0, ω, δ, W, x)])
        append!(yU_1, [cifft(sFU_1, ω, δ, W, x)])
    end

    return (x, (ywT0, yU_1, ywT1, yU0)) 
end

# FFT approximation of the inverse Fourier Transforms
# This approximates (1/2π) ∫ f(ω)exp(i ω x) dω. 
function cifft(f::Function, ω::AbstractVector, δ::Real, W::Real, x::AbstractVector)
    yf= ifftshift(ifft(f.(ω)))
    N = length(ω)
    return (δ .* N .* exp.(-im .*x .*W)  ./ (2*pi)) .* yf
end

# This takes in the discrete values computed by support_functions and interpolates them so that we 
# can use them like a Function. It also translates the supporter functions so that they are centred
# at the correct elements. 
function interpolate_supporter_functions(x1::AbstractVector, x2::AbstractVector, uS::NTuple{4, Vector{Any}}, I::AbstractVector, s::AbstractVector,)

    ## Scale and translate the reference supporter functions during the interpolation
    ## for each element. 
    (ywT0, yU_1, ywT1, yU0) = uS
    el_no = length(I)-1
    c = 2. ./ (I[2:end] - I[1:end-1]); d = (I[1:end-1] + I[2:end]) ./ 2


    yU_1 = [interpolate((x1 .+ d[j],), real.(yU_1[findall(x->x==c[j],s)[1]])[:], Gridded(Linear())) for j in 1:el_no]
    yU0 =  [interpolate((x2 .+ d[j],), real.(yU0[findall(x->x==c[j],s)[1]])[:], Gridded(Linear())) for j in 1:el_no]
    ywT0 = [interpolate((x1 .+ d[j],), real.(ywT0[findall(x->x==c[j],s)[1]])[:], Gridded(Linear())) for j in 1:el_no]
    ywT1 = [interpolate((x2 .+ d[j],), real.(ywT1[findall(x->x==c[j],s)[1]])[:], Gridded(Linear())) for j in 1:el_no]
    return (ywT0, yU_1, ywT1, yU0)
end

# This combines the FFT approximation and the interpolation. 
function fft_supporter_functions(λ::Number, μ::Number, η::Number; W::Real=1000., δ::Real=0.001, 
    I::AbstractVector=[-1.,1.], N::Int=5, stabilise::Bool=false)
    
    s = unique(2. ./ (I[2:end] - I[1:end-1]))
    
    # Special case analytical expressions
    if λ == μ == η ≈ 0

        if s != [1.0]
            return error("λ == μ == η ≈ 0, currently can only handle translations of the reference interval [-1,1].")
        end
        ywT0 = []; ywT1 = []; yU0 = []; yU_1 = []
        
        half_laplace_wT0 = x -> abs(x) <= 1 ? log(2)-Base.MathConstants.eulergamma : log(2)-Base.MathConstants.eulergamma-asinh(sqrt(x^2-1))
        half_laplace_wT1 = x -> ExtendedChebyshevT()[x,2]
        half_laplace_U_1 = x -> abs(x) <= 1 ? -asin(x) : -sign(x)*pi/2
        half_laplace_U0 = x -> ExtendedWeightedChebyshevU()[x,1]
        
        for els = 1 : length(I)-1
            # We broadcast to be consistent with the Interpolations in the non-special case.
            append!(ywT0, [x->half_laplace_wT0(affinetransform(I[els], I[els+1], x))])
            append!(ywT1, [x->half_laplace_wT1(affinetransform(I[els], I[els+1], x))])
            append!(yU0, [x->half_laplace_U0(affinetransform(I[els], I[els+1], x))])
            append!(yU_1, [x->half_laplace_U_1(affinetransform(I[els], I[els+1], x))])
        end
        return (ywT0, yU_1, ywT1, yU0)
    end 
    
    (x, uS) = supporter_functions(λ, μ, η, W=W, δ=δ, s=s, N=N, stabilise=stabilise)
    return interpolate_supporter_functions(x, x, uS, I, s)
end

function coefficient_supporter_functions(A::AbstractArray, x::AbstractVector, uS::NTuple{4, Vector}, 
    N::Int; tol::Real=1e-6, normtype::Function=riemann)

    (ywT0, yU_1, ywT1, yU0) = uS
    el_no = length(yU0)
    yu_1 = [solvesvd(A, normtype(x, yU_1[j]); tol=tol) for j in 1:el_no]
    yu0 = [solvesvd(A, normtype(x, yU0[j]); tol=tol) for j in 1:el_no]
    ywt0 = [solvesvd(A, normtype(x, ywT0[j]); tol=tol) for j in 1:el_no]
    ywt1 = [solvesvd(A, normtype(x, ywT1[j]); tol=tol) for j in 1:el_no]
    yu_1 = [expansion_sum_space(yu_1[j],  N, el_no) for j in 1:el_no]
    yu0 = [expansion_sum_space(yu0[j], N, el_no) for j in 1:el_no]
    ywt0 = [expansion_sum_space(ywt0[j], N, el_no) for j in 1:el_no]
    ywt1 = [expansion_sum_space(ywt1[j],  N, el_no) for j in 1:el_no]
    return (ywt0, yu_1, ywt1, yu0)
end

# FFT approximation of the inverse Fourier Transforms
# This approximates (1/2π) ∫ f(ω)exp(i ω x) dω. 
function inverse_fourier_transform(F::Function, ω::AbstractVector)
    
    δ = step(ω); W = abs(ω[1])
    x = ifftshift(fftfreq(length(ω), 1/δ) * 2 * pi)
    N = length(ω)

    f = ifftshift(ifft(F.(ω)))
    return (x, (δ .* N .* exp.(-im .*x .*W)  ./ (2*pi)) .* f)
end

# Save support functions
function save_supporter_functions(filepath, x1, x2, uS)
    (ywT0, yU_1, ywT1, yU0) = uS
    writedlm(filepath, [x1, x2, real.(ywT0[1]), real.(yU_1[1]), real.(ywT1[1]), real.(yU0[1])])
end

# Load a saved set of support functions
function load_supporter_functions(filepath, I)
    # Load saved supporter_functions
    supp = readdlm(filepath)
    
    # Extract numbers
    x1 = []; x2 = [];
    ywT0 = []; yU_1 = []; ywT1 = []; yU0 = []
    append!(ywT0, [supp[3,:][supp[3,:].!=""]]); append!(yU_1, [supp[4,:][supp[4,:].!=""]]); 
    append!(ywT1, [supp[5,:][supp[5,:].!=""]]); append!(yU0, [supp[6,:][supp[6,:].!=""]]); 
    
    # Extract x
    x1 = supp[1,:][supp[1,:].!=""]; x2 = supp[2,:][supp[2,:].!=""];
    
    # Interpolate solutions
    s = unique(2. ./ (I[2:end] - I[1:end-1]))
    if s != [1.0]
        @warn "Scaling vector is not [1.0], the loaded solutions are probably not correct."
    end
    return interpolate_supporter_functions(x1, x2, (ywT0, yU_1, ywT1, yU0), I, s)
end