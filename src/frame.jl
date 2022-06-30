# Custom SVD solver for least squares problems
function solvesvd(A::AbstractArray, b::AbstractVector; tol::Real=1e-7)
    U,σ,V = svd(Matrix(A)) # BlockBandedMatrices cannot do SVD
    filter!(>(tol), σ)
    r = length(σ)
    return V[:,1:r] * (inv.(σ) .* (U[:,1:r]' * b))
end

# Affine transformation
at(a,b,x) = (b-a)/2 * x .+ (b+a)/2

# Construct collocation points
function collocation_points(M::Int, Me::Int; I::AbstractVector=[-1.,1.], endpoints::AbstractVector=[-5.,5.], innergap::Real = 0.)
    Tp = eltype(I)
    el_no = length(I)-1

    x = Array{Tp}(undef,el_no*M+2*Me)
    xnodes = LinRange{Tp}(innergap,1-innergap,M)
    # chebnodes = sort(cos.(π.*xnodes))

    xxnodes = LinRange{Tp}(-1+innergap,1-innergap,M)
    for el = 1:el_no
        x[(el-1)*M+1:el*M] = at(I[el], I[el+1], xxnodes) 
    end
    xnodes = LinRange{Tp}(innergap,1-innergap,Me)
    # chebnodes = sort(cos.(π.*xnodes))

    xxnodes = LinRange{Tp}(-1+innergap,1-innergap,Me)
    x[el_no*M+1:el_no*M+Me] = at(endpoints[1], I[1], xxnodes) 
    x[el_no*M+1+Me:el_no*M+2*Me] = at(I[end],endpoints[2],xxnodes)
    return sort(unique(x))
end

# Convert function evaluation to Riemann sum
function riemann(x::AbstractVector, f::Union{Function, Interpolations.GriddedInterpolation})
    y = sort(x)
    h = 0.5 .* (
            append!(y[2:end], y[end]) .- y
         .+ y .- append!([y[1]], y[1:end-1])
    )
    b = sqrt.(h).*f(y)
    return b
end

# Just function evaluation
function evaluate(x::AbstractVector, f::Union{Function, Interpolations.GriddedInterpolation})
    y = sort(x)
    b = f(y)
    return b
end

# Fit low order expansion to higher order expansion
function expansion_sum_space(c::AbstractVector, N::Int, el_no::Int)
    v = zeros(1 + el_no*(N-1))
    v[1:length(c)] = c
    return v
end

# Construct Least Squares matrix for frame coefficient computation

function framematrix(x::AbstractVector, Sp::SumSpaceP, Nn::Int; normtype::Function=riemann)
    Tp = eltype(Sp)
    el = length(Sp.I) - 1
    rows = [length(x)]; cols = vcat([1], Fill(2, el*(Nn+1)))
    # Create correct block structure
    A = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (sum(rows),sum(cols)))
    A[:,Block.(1:length(cols))] = normtype(x, x->Sp[x, Block.(1:length(cols))])
    return A
end

function framematrix(x::AbstractVector, Sp::ElementSumSpaceP, Nn::Int; normtype::Function=riemann)
    Tp = eltype(Sp)
    el = length(Sp.I) - 1
    rows = [length(x)]; cols = vcat([1], Fill(el, (2*Nn+2)))
    # Create correct block structure
    A = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (sum(rows),sum(cols)))
    # Form columns of Least Squares matrix.
    A[:,Block.(1:length(cols))] = normtype(x, x->Sp[x, Block.(1:length(cols))])
    return A
end

# Construct Least Squares matrix for dual sum space
function framematrix(x::AbstractVector, Sd::SumSpaceD, Nn::Int; normtype::Function=riemann)
    Tp = eltype(Sd)
    el = length(Sd.I) - 1
    rows = [length(x)]; cols = vcat([1], Fill(2, el*(Nn+3)))
    # Create correct block structure
    A = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (sum(rows),sum(cols)))
    # Form columns of Least Squares matrix.
    A[:,Block.(1:length(cols))] = normtype(x, x->Sd[x, Block.(1:length(cols))])
    return A
end

function framematrix(x::AbstractVector, Sd::ElementSumSpaceD, Nn::Int; normtype::Function=riemann)
    Tp = eltype(Sd)
    el = length(Sd.I) - 1
    rows = [length(x)]; cols = vcat([1], Fill(el, (2*Nn+6)))
    # Create correct block structure
    A = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (sum(rows),sum(cols)))
    # Form columns of Least Squares matrix.
    A[:,Block.(1:length(cols))] = normtype(x, x->Sd[x, Block.(1:length(cols))])
    return A
end