using Test, ClassicalOrthogonalPolynomials, SumSpaces

@testset "sumspace" begin
    @testset "basics" begin
        Sp = SumSpaceP()
        @test Sp.I ≈ [-1,1]

        a = [-5,3.]
        Spa = SumSpaceP(a)
        @test Spa.I ≈ [-5,3]

        @test Sp != Spa

        Sd = SumSpaceD()
        @test Sd.I ≈ [-1,1]

        Sda = SumSpaceD(a)
        @test Sda.I ≈ [-5,3]

        @test Sd != Sda
        @test Sp != Sd
        @test Spa != Sda
    end

    @testset "Evaluation" begin
        @testset "primal" begin
            Sp = SumSpaceP()
            wU = ExtendedWeightedChebyshevU()
            eT = ExtendedChebyshevT()

            @test Sp[0.1,oneto(0)] == Float64[]
            @test Sp[0.1,oneto(1)] ≈ [1]
            @test Sp[0.1,oneto(3)] ≈ [1, wU[0.1,1], eT[0.1,2]]
            for N = 1:10
                @test Sp[0.1,1:2:2*N-1] ≈ eT[0.1,1:N]
                @test Sp[0.1,2*N-1] ≈ eT[0.1,N]
                @test Sp[0.1,2:2:2*N] ≈ wU[0.1,1:N]
                @test Sp[0.1,2*N] ≈ wU[0.1,N]

                @test Sp[1.1,1:2:2*N-1] ≈ eT[1.1,1:N]
                @test Sp[1.1,2*N-1] ≈ eT[1.1,N]
                @test Sp[1.1,2:2:2*N] ≈ wU[1.1,1:N]
                @test Sp[1.1,2*N] ≈ wU[1.1,N]

            end

        end

        @testset "dual" begin
            Sd = SumSpaceD()
            wT = ExtendedWeightedChebyshevT()
            eU = ExtendedChebyshevU()

            @test Sd[0.1,oneto(0)] == Float64[]
            @test Sd[0.1,oneto(1)] ≈ eU[0.1,oneto(1)]
            @test Sd[0.1,oneto(3)] ≈ [eU[0.1,1], wT[0.1,1], eU[0.1,2]]
            for N = 1:10
                @test Sd[0.1,1:2:2*N-1] ≈ eU[0.1,1:N]
                @test Sd[0.1,2*N-1] ≈ eU[0.1,N]
                @test Sd[0.1,2:2:2*N] ≈ wT[0.1,1:N]
                @test Sd[0.1,2*N] ≈ wT[0.1,N]

                @test Sd[1.1,1:2:2*N-1] ≈ eU[1.1,1:N]
                @test Sd[1.1,2*N-1] ≈ eU[1.1,N]
                @test Sd[1.1,2:2:2*N] ≈ wT[1.1,1:N]
                @test Sd[1.1,2*N] ≈ wT[1.1,N]

            end
        end

        @testset "appended" begin
            f = x -> x
            uS = (f,f,f,f)
            ASp = AppendedSumSpace(uS, [])
            Sp = SumSpaceP()

            @test ASp[0.1,oneto(0)] == Float64[]
            @test ASp[0.1,oneto(1)] ≈ Sp[0.1,oneto(1)]
            @test ASp[0.1,oneto(7)] ≈ [Sp[0.1,1], f(0.1), f(0.1), f(0.1), f(0.1), Sp[0.1,2], Sp[0.1,3]]
        end
    end

    @testset "BlockStructure" begin
        Sp = SumSpaceP()
        Sd = SumSpaceD()
        f = x -> x
        uS = (f,f,f,f)
        ASp = AppendedSumSpace(uS, [])


        @test axes(Sp[0.1, Block.(1:3)]) == (1:1:5,)
        @test Sp[0.1, Block.(1:3)] == Sp[0.1,1:5]
        @test axes(Sd[0.1, Block.(1:3)]) == (1:1:5,)
        @test Sd[0.1, Block.(1:3)] == Sd[0.1,1:5]
        @test axes(ASp[0.1, Block.(1:3)]) == (1:1:5,)
        @test ASp[0.1, Block.(1:3)] == ASp[0.1,1:5]
    end

    @testset "Identity maps" begin
        Sp = SumSpaceP()
        Sd = SumSpaceD()

        A = (Sd \ Sp)
        @test axes(A) == (1:1:∞, 1:1:∞)
        @test A[1,1] ≈ -1
        @test A[5,1] ≈ 1
        @test A[2:3,2:3] ≈ [[0.5,0] [0,-0.5]]
        @test A[Block.(2), Block.(2)] == A[2:3,2:3]
        @test A[Block.(4), Block.(2)] == [[-0.5,0] [0,0.5]]
        @test A[Block.(3), Block.(1)] ≈ [[0] [1]]'
    end

    @testset "derivative" begin
        Sp = SumSpaceP()
        x = axes(Sp, 1)
        A_Sd = Derivative(x)*Sp

        Sd = SumSpaceD()

        A = Sd \ A_Sd
        @test axes(A) == (1:1:∞, 1:1:∞)
        @test A[1,1] ≈ 0
        @test A[2,1] ≈ 0
        @test A[2:3,2:3] ≈ [[0,0] [0,0]]
        @test A[4:5,2:3] ≈ [[-1,0] [0,1]]
        @test A[Block.(2), Block.(2)] == A[2:3,2:3]
        for N = 3:20
            @test A[Block.(N), Block.(N-1)] == [[-(N-2),0] [0,N-2]]
        end
    end

    @testset "hilbert" begin
        Sp = SumSpaceP()
        x = axes(Sp, 1)
        H = inv.(x .- x')
        H_Sp = H*Sp

        A = Sp \ H_Sp
        @test axes(A) == (1:1:∞, 1:1:∞)
        @test A[1,1] ≈ 0
        @test A[2,1] ≈ 0
        @test A[2:3,2:3] ≈ [[0,π] [-π,0]]
        @test A[4:5,2:3] ≈ [[0,0] [0,0]]
        @test A[Block.(2), Block.(2)] == A[2:3,2:3]
        for N = 2:10
            @test A[Block.(N), Block.(N)] ≈ [[0,π] [-π,0]]
        end
    end

    @testset "jacobimatrix" begin
        Sp = SumSpaceP()
        A = jacobimatrix(Sp)
        @test axes(A) == (1:1:∞, 1:1:∞)
        @test A[1,1] ≈ 0
        @test A[2,1] ≈ 0.5
        @test A[1,3] ≈ 0.5
        @test A[2:3,2:3] ≈ [[0,0] [0,0]]
        @test A[4:5,2:3] ≈ [[0.5,0] [0,0.5]]
        @test A[Block.(3), Block.(2)] == A[4:5,2:3]
        for N = 2:10
            @test A[Block.(N+1), Block.(N)] ≈ [[0.5,0] [0,0.5]]
        end
    end

    @testset "appended-identity" begin
        f = x -> x
        uS = (f,f,f,f)
        cuS = [[1.], [2.], [3.], [4.]]
        ASp = AppendedSumSpace(uS, cuS)
        Sp = SumSpaceP()
        Sd = SumSpaceD()

        A = Sd \ ASp
        B = Sd \ Sp

        @test A[1:10,1] == B[1:10,1]
        @test A[1,2:5] ≈ [-1,-2,-3,-4]
        @test A[5,2:5] ≈ [1,2,3,4]
        @test A[1:30, 6:35] == B[1:30,2:31]
    end

    @testset "sqrt-Δ" begin
        Sp = SumSpaceP()
        Sd = SumSpaceD()

        x = axes(Sp, 1)
        H = inv.(x .- x')
        D = Derivative(x)
        Δ_s = Sd\D*H*Sp
    end
end