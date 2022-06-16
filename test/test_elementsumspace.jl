using Test, SumSpaces, ClassicalOrthogonalPolynomials

@testset "element-sumspace" begin
    @testset "basics" begin
        eSp = ElementSumSpaceP()

        @test eSp.I ≈ [-1,1]

        a = [-5,3.]
        eSpa = ElementSumSpaceP(a)
        @test eSpa.I ≈ [-5,3]

        b = [-5,3.,5]
        eSpb = ElementSumSpaceP(b)
        @test eSpb.I ≈ [-5,3,5]

        @test eSp != eSpa

        eSd = ElementSumSpaceD()
        @test eSd.I ≈ [-1,1]

        eSda = ElementSumSpaceD(a)
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
            a = [-3.,-1, 1, 3]
            eSp = ElementSumSpaceP(a)

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
            a = [-3.,-1, 1, 3]
            eSd = ElementSumSpaceD(a)

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
            a = [-3.,-1, 1, 3]
            f = x -> x
            uS = ([f,f,f],[f,f,f],[f,f,f],[f,f,f])
            ASp = ElementAppendedSumSpace(uS, [], a)
            Sp = ElementSumSpaceP(a)

            @test ASp[0.1,oneto(0)] == Float64[]
            @test ASp[0.1,oneto(1)] ≈ Sp[0.1,oneto(1)]
            for N = 2:13
                @test ASp[0.1,N] ≈ f(0.1)
                @test ASp[0.1,Block.(N+4)] ≈ Sp[0.1,Block.(N)]
                @test ASp[1.1,Block.(N+4)] ≈ Sp[1.1,Block.(N)]
            end
        end

    end
end