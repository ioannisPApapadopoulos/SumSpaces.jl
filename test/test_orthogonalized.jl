using SumSpaces, LazyArrays, LazyBandedMatrices, Test

@testset "orthogonalised" begin
    P = SumSpaceP()
    Q = SumSpaceD()

    x = axes(P, 1)
    H = inv.(x .- x')
    D = Derivative(x)
    Δ_s = Q\D*H*P

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
    T̃ = ExtendedChebyshevU()
    M = BlockBroadcastArray(hvcat, 2, unitblocks(U'T), unitblocks([Zeros(∞) W'U]),
                                      unitblocks(T[:,2:end]'V), unitblocks(U'T))

    M_half = M[Block.(1:5), Block.(1:5)]*Δ_s[Block.(2:6), Block.(2:5)]

    M_half[Block(7,5)]
    M_half[Block(5,7)]
end