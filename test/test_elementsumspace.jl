using Test, SumSpaces
import SumSpaces: Fill

@testset "element-sumspace" begin
    @testset "basics" begin
        eSp = ElementSumSpaceP()

        @test eSp.I ≈ [-1,1]

        I = [-5,3.]
        eSpa = ElementSumSpaceP(I)
        @test eSpa.I ≈ [-5,3]

        b = [-5,3.,5]
        eSpb = ElementSumSpaceP(b)
        @test eSpb.I ≈ [-5,3,5]

        @test eSp != eSpa

        eSd = ElementSumSpaceD()
        @test eSd.I ≈ [-1,1]

        eSda = ElementSumSpaceD(I)
        @test eSda.I ≈ [-5,3]

        b = [-5,3.,5]
        eSdb = ElementSumSpaceD(b)
        @test eSdb.I ≈ [-5,3,5]

        @test eSd != eSda
        @test eSp != eSd
        @test eSpa != eSda
    end
    
    
    @testset "Evaluation" begin 
        @testset "primal" begin
            I = [-3.,-1, 1, 3]
            eSp = ElementSumSpaceP(I)

            a1 = [-3,-1.]
            a2 = [-1.,1]
            a3 = [1.,3]

            a1Sp = SumSpaceP(a1)
            a2Sp = SumSpaceP(a2)
            a3Sp = SumSpaceP(a3)

            @test eSp[0.1,oneto(0)] == Float64[]
            @test eSp[0.1,oneto(1)] ≈ [1]
            @test eSp[0.1,oneto(4)] ≈ [1, a1Sp[0.1,2], a2Sp[0.1,2], a3Sp[0.1,2]]
            for N = 2:10
                @test eSp[0.1, Block.(N)] ≈ [a1Sp[0.1,N], a2Sp[0.1,N], a3Sp[0.1,N]]
                @test eSp[10, Block.(N)] ≈ [a1Sp[10,N], a2Sp[10,N], a3Sp[10,N]]
            end
        end

        @testset "dual" begin
            I = [-3.,-1, 1, 3]
            eSd = ElementSumSpaceD(I)

            a1 = [-3,-1.]
            a2 = [-1.,1]
            a3 = [1.,3]

            a1Sd = SumSpaceD(a1)
            a2Sd = SumSpaceD(a2)
            a3Sd = SumSpaceD(a3)

            @test eSd[0.1,oneto(0)] == Float64[]
            @test eSd[0.1,oneto(1)] == a1Sd[0.1,oneto(1)]
            @test eSd[0.1,oneto(4)] ≈ [a1Sd[0.1,1], a1Sd[0.1,2], a2Sd[0.1,2], a3Sd[0.1,2]]
            for N = 2:10
                @test eSd[0.1, Block.(N)] ≈ [a1Sd[0.1,N], a2Sd[0.1,N], a3Sd[0.1,N]]
                @test eSd[10, Block.(N)] ≈ [a1Sd[10,N], a2Sd[10,N], a3Sd[10,N]]
            end
        end

        @testset "appended" begin
            I = [-3.,-1, 1, 3]
            f = x -> x
            uS = ([f,f,f],[f,f,f],[f,f,f],[f,f,f])
            ASp = ElementAppendedSumSpace(uS, [], I)
            Sp = ElementSumSpaceP(I)

            @test ASp[0.1,oneto(0)] == Float64[]
            @test ASp[0.1,oneto(1)] ≈ Sp[0.1,oneto(1)]
            for N = 2:13
                @test ASp[0.1,N] ≈ f(0.1)
                @test ASp[0.1,Block.(N+4)] ≈ Sp[0.1,Block.(N)]
                @test ASp[1.1,Block.(N+4)] ≈ Sp[1.1,Block.(N)]
            end
        end

    end

    @testset "BlockStructure" begin
        I = [-3.,-1, 1, 3]
        Sp = ElementSumSpaceP(I)
        Sd = ElementSumSpaceD(I)
        f = x -> x
        uS = ([f,f,f],[f,f,f],[f,f,f],[f,f,f])
        ASp = ElementAppendedSumSpace(uS, [], I)


        @test axes(Sp[0.1, Block.(1:3)]) == (1:1:7,)
        @test Sp[0.1, Block.(1:3)] == Sp[0.1,1:7]
        @test axes(Sd[0.1, Block.(1:3)]) == (1:1:7,)
        @test Sd[0.1, Block.(1:3)] == Sd[0.1,1:7]
        @test axes(ASp[0.1, Block.(1:3)]) == (1:1:7,)
        @test ASp[0.1, Block.(1:3)] == ASp[0.1,1:7]
    end

    @testset "Identity maps" begin
        I = [-5.,-1, 1, 2]
        eSp = ElementSumSpaceP(I)
        eSd = ElementSumSpaceD(I)
        Sp = SumSpaceP()
        Sd = SumSpaceD()

        eA = (eSd \ eSp)
        A = (Sd \ Sp)
        @test eA[1][1:100,1:100] == A[1:100,1:100]

        # Should not need first for loop but A[Block.(1:20), Block.(1:20)]
        # errors and it's not within this packages' scope.
        for N = 1:20
            for j in 1:3
                @test eA[j][Block.(N), Block.(1:20)] == A[Block.(N), Block.(1:20)]
            end
        end
    end

    @testset "derivative" begin
        I = [-5.,-1, 1, 2] 
        Sp = ElementSumSpaceP(I)
        
        x = axes(Sp, 1)
        A_Sd = Derivative(x)*Sp

        Sd = ElementSumSpaceD(I)

        A = [Sd \ A_Sd[j] for j = 1:3]
        @test axes(A[1]) == (1:1:∞, 1:1:∞)
        for j = 1:3
            @test A[j][1,1] ≈ 0
            @test A[j][2,1] ≈ 0
            @test A[j][2:3,2:3] ≈ [[0,0] [0,0]]
            @test A[j][4:5,2:3] ≈ [[-2/(I[j+1]-I[j]),0] [0,2/(I[j+1]-I[j])]]
            @test A[j][Block.(2), Block.(2)] == A[j][2:3,2:3]
            for N = 3:20
                @test A[j][Block.(N), Block.(N-1)] == [[-2/(I[j+1]-I[j])*(N-2),0] [0,2/(I[j+1]-I[j])*(N-2)]]
            end
        end
    end

    @testset "hilbert" begin
        I = [-5.,-1, 1, 2]
        eSp = ElementSumSpaceP(I)
        eSd = ElementSumSpaceD(I)
        Sp = SumSpaceP()
        Sd = SumSpaceD()

        x = axes(eSp, 1)
        H = inv.(x .- x')
        H_eSp = H*eSp
        H_Sp = H*Sp

        eA = [eSp \ H_eSp[j] for j in 1:3]
        A = Sp \ H_Sp
        for j in 1:3
            @test axes(eA[j]) == (1:1:∞, 1:1:∞)
            @test eA[j][1:100,1:100] == A[1:100,1:100]
            for N = 2:10
                @test eA[j][Block.(N), Block.(N)] == A[Block.(N), Block.(N)]
            end
        end
    end

    @testset "appended-identity" begin
        I = [-5.,-1, 1, 2]
        f = x -> x
        uS = ([f,f,f],[f,f,f],[f,f,f],[f,f,f])
        cuS = [[[1.],[1.],[1.]], [[2.],[2.],[2.]], [[3.],[3.],[3.]], [[4.],[4.],[4.]]]
        
        ASp = ElementAppendedSumSpace(uS, cuS, I)
        Sp = ElementSumSpaceP(I)
        Sd = ElementSumSpaceD(I)

        A = Sd \ ASp
        B = Sd \ Sp

        el_no = length(I)-1

        @test A[1,1] ≈ -1
        @test A[3el_no+2, 1] ≈ 1 
        for N = 2:5
            @test A[1,Block.(N)] ≈ -[N-1,N-1,N-1]
            @test A[3el_no+2, Block.(N)] ≈ [N-1,N-1,N-1]
        end
        @test A[2:4,Block.(6)] ≈ [[0.5,0,0] [0,0.5,0] [0,0,0.5]]
        @test A[3el_no+5:3el_no+7,Block.(6)] ≈ -[[0.5,0,0] [0,0.5,0] [0,0,0.5]]
    
        ASp = ElementAppendedSumSpace(uS, cuS)
        @test ASp.I ≈ [-1,1]
    end

    @testset "coefficient_stack" begin
        K = 2; N = 1
        v = Array(1:1+K*(2N+6))
        v = BlockArray(v, vcat(1,Fill(K,(length(v)-1)÷K)))
        w = coefficient_stack(v, N, K)
        @test w[1] == v[1]
        @test w[2:length(w)÷2+1] == v[2:2:end]
        @test w[length(w)÷2+2:end] == v[3:2:end]
    end

    @testset "coefficient_interlace" begin
        K = 2; N = 1
        v = Array(1:1+K*(2N+6))
        w = coefficient_interlace(v, N, K)

        @test length(w.blocks) == 9
        @test length(w.blocks[1]) == 1
        for j = 2:9
            @test length(w.blocks[j]) == 2
        end
        
        @test w[1] == v[1]
        for j in 2:9
            w[Block.(j)] = v[j:(2N+6):end] 
        end
    end

end