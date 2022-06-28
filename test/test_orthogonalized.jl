using SumSpaces, LazyArrays, LazyBandedMatrices, ClassicalOrthogonalPolynomials, FillArrays, Test

@testset "orthogonalised" begin
    P = SumSpaceP()
    Q = SumSpaceD()

    x = axes(P, 1)
    H = inv.(x .- x')
    D = Derivative(x)
    Δ_s = (Q\D*H*P) # π * (im*k * (-im*k))

    P'Q
    P[0.1,1]
    Q[0.1,1]

    T = ChebyshevT()
    U = ChebyshevU()
    W = Weighted(U)
    V = Weighted(T)
    sum(T; dims=1)

    BlockVcat(-Inf,1,1), sum(Weighted(T); dims=1))

    W'U
    


    Ũ = ExtendedChebyshevU()
    T̃ = ExtendedChebyshevT()
    M = BlockBroadcastArray(hvcat, 2, unitblocks(-(U'T)), unitblocks([Zeros(∞) W'U]),
                                      unitblocks(T[:,2:end]'V), unitblocks(U'T))

    N = 10
    M_half = M[Block.(1:N), Block.(1:N+1)]*Δ_s[Block.(2:N+2), Block.(2:N+1)]/π

    R = cholesky(Symmetric(M_half)).U

    Δ_s[Block.(2:N+2), Block.(2:N+1)] * inv(R)


    W'W
    W'T[:,2:end]
    M = BlockBroadcastArray(hvcat, 2, unitblocks(W'W), unitblocks(W'T[:,2:end]),
                                      unitblocks(T[:,2:end]'W), unitblocks(W'W))

    inv(R)'M[Block.(1:N),Block.(1:N)]*inv(R)
end