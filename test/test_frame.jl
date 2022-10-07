using Test, SumSpaces

@testset "frame" begin
    @testset "solve svd" begin
        A = Matrix([[1,2] [3,4] [5,6] [7,8]])'
        b = [1,2,3,4]
        x = A \ b
        x2 = solvesvd(A, b, tol=1e-15)
        @test x ≈ x2
    end


    @testset "affine transform" begin
        import SumSpaces: at
        x = -1:0.5:1
        y = at(1,5,x)
        y2 = 1:1:5.
        @test y == y2
    end

    @testset "collocation points" begin
        @test collocation_points(5, 2) == [-5.0,-1.0,-0.5,0.0,0.5,1.0,5.0]
        @test collocation_points(5, 2, I=[-2, 2.]) == [-5.0,-2.0,-1.0,0.0,1.0,2.0,5.0]
        @test collocation_points(5, 2, endpoints=[-10,10.]) == [-10.0,-1.0,-0.5,0.0,0.5,1.0,10.0]
        @test collocation_points(5, 2, I=[-2, 2.], endpoints=[-10,10.]) == [-10.0,-2.0,-1.0,0.0,1.0,2.0,10.0]
        @test collocation_points(5, 3, I=[-2, 2.], endpoints=[-10,10.]) == [-10.0,-6.0,-2.0,-1.0,0.0,1.0,2.0,6.0,10.0]
        @test collocation_points(5, 3, I=[-2, 2.], endpoints=[-10,10.], innergap=0.5) ==  [-8.0,-6.0,-4.0,-1.0,-0.5,0.0,0.5,1.0,4.0,6.0,8.0]
        @test collocation_points(5, 3, I=[-2, 2.,3]) ==  [-5.0,-3.5,-2.0,-1.0,0.0,1.0,2.0,2.25,2.5,2.75,3.0,4.0,5.0]
        @test collocation_points(5, 3, I=[-2, 2.,3], remove_endpoints=true) == [-5.0,-3.5,-1.0,0.0,1.0,2.25,2.5,2.75,4.0,5.0]
    end

    @testset "riemann/evaluate" begin
        x = Array(-1:0.5:1)
        f = x -> x
        riemann(x, f) ≈  [-0.5,-0.353553,0.0,0.353553,0.5]
        evaluate(x, f) == Array(-1:0.5:1)
    end

    @testset "expansion" begin
        import SumSpaces: expansion_sum_space
        c = [1,2,3]
        expansion_sum_space(c, 3, 1) == c
        expansion_sum_space(c, 5, 1) == [1.,2,3,0,0]
        expansion_sum_space(c, 5, 2) == [1.,2,3,0,0,0,0,0,0]
    end

    @testset "framematrix" begin
        @testset "SumSpaceP" begin
            Sp = SumSpaceP()
            x = [-0.5,0.5]
            A = framematrix(x, Sp, 1)
            @test A[Block.(1), Block.(1)] ≈ [0.5, 0.5]
            @test A[Block.(1), Block.(2)] ≈ [[0.4330127018922193, 0.4330127018922193] [-0.25,0.25]]
            @test A[Block.(1), Block.(3)] ≈ [[-0.4330127018922193,0.4330127018922193] [-0.25,-0.25]]
        
            A = framematrix(x, Sp, 1, normtype=evaluate)
            @test A[Block.(1), Block.(1)] ≈ [1,1]
            @test A[Block.(1), Block.(2)] ≈ [[Sp[-0.5,2],Sp[0.5,2]] [Sp[-0.5,3], Sp[0.5,3]]]
            @test A[Block.(1), Block.(3)] ≈ [[Sp[-0.5,4],Sp[0.5,4]] [Sp[-0.5,5], Sp[0.5,5]]]
        end
    end

    @testset "SumSpaceD" begin
        Sd = SumSpaceD()
        x = [-0.5,0.5]
        A = framematrix(x, Sd, 1)
        @test A[Block.(1), Block.(1)] ≈ [0, 0]
        @test isapprox(A[Block.(1), Block.(2)], [[0.57735,0.57735] [0,0]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[-0.288675,0.288675] [0.5,0.5]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[-0.288675,-0.288675] [-0.5,0.5]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[0.57735,-0.57735] [0,0]], atol=1e-5)
    
        A = framematrix(x, Sd, 1, normtype=evaluate)
        @test A[Block.(1), Block.(1)] ≈ [0,0]
        @test A[Block.(1), Block.(2)] ≈ [[Sd[-0.5,2],Sd[0.5,2]] [Sd[-0.5,3], Sd[0.5,3]]]
        @test A[Block.(1), Block.(3)] ≈ [[Sd[-0.5,4],Sd[0.5,4]] [Sd[-0.5,5], Sd[0.5,5]]]
        @test A[Block.(1), Block.(4)] ≈ [[Sd[-0.5,6],Sd[0.5,6]] [Sd[-0.5,7], Sd[0.5,7]]]
        @test A[Block.(1), Block.(5)] ≈ [[Sd[-0.5,8],Sd[0.5,8]] [Sd[-0.5,9], Sd[0.5,9]]]
    end

    @testset "ElementSumSpaceP" begin
        Sp = ElementSumSpaceP([-2,-1,1.])
        x = [-0.5,0.5]
        A = framematrix(x, Sp, 1)
        @test isapprox(A[Block.(1), Block.(1)], [0.5,0.5], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(2)], [[0,0] [0.433013,0.433013]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[0.133975, 0.0635083 ] [-0.25,0.25]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[0,0] [-0.433013,0.433013]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[0.0358984, 0.00806662 ] [-0.25,-0.25]], atol=1e-5)       

        A = framematrix(x, Sp, 1, normtype=evaluate)
        @test A[Block.(1), Block.(1)] ≈ [1,1]
        @test A[Block.(1), Block.(2)] ≈ [[Sp[-0.5,2],Sp[0.5,2]] [Sp[-0.5,3], Sp[0.5,3]]]
        @test A[Block.(1), Block.(3)] ≈ [[Sp[-0.5,4],Sp[0.5,4]] [Sp[-0.5,5], Sp[0.5,5]]]
        @test A[Block.(1), Block.(4)] ≈ [[Sp[-0.5,6],Sp[0.5,6]] [Sp[-0.5,7], Sp[0.5,7]]]
        @test A[Block.(1), Block.(5)] ≈ [[Sp[-0.5,8],Sp[0.5,8]] [Sp[-0.5,9], Sp[0.5,9]]]
    end

    @testset "ElementSumSpaceD" begin
        Sd = ElementSumSpaceD([-2,-1,1.])
        x = [-0.5,0.5]
        A = framematrix(x, Sd, 1)
        @test isapprox(A[Block.(1), Block.(1)], [ -0.5773502691896258,-0.5163977794943222], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(2)], [[0,0] [0.57735, 0.57735]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[-0.288675,-0.129099] [0,0]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[0,0] [-0.288675,0.288675]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[-0.0773503,-0.0163978] [0.5,0.5]], atol=1e-5)       
        @test isapprox(A[Block.(1), Block.(9)], [[-0.0055535,-0.000264549] [0,0]], atol=1e-5)  

        A = framematrix(x, Sd, 1, normtype=evaluate)
        @test A[Block.(1), Block.(1)] ≈ [Sd[-0.5,1],Sd[0.5,1]]
        @test A[Block.(1), Block.(2)] ≈ [[Sd[-0.5,2],Sd[0.5,2]] [Sd[-0.5,3], Sd[0.5,3]]]
        @test A[Block.(1), Block.(3)] ≈ [[Sd[-0.5,4],Sd[0.5,4]] [Sd[-0.5,5], Sd[0.5,5]]]
        @test A[Block.(1), Block.(4)] ≈ [[Sd[-0.5,6],Sd[0.5,6]] [Sd[-0.5,7], Sd[0.5,7]]]
        @test A[Block.(1), Block.(5)] ≈ [[Sd[-0.5,8],Sd[0.5,8]] [Sd[-0.5,9], Sd[0.5,9]]]
        @test A[Block.(1), Block.(9)] ≈ [[Sd[-0.5,16],Sd[0.5,16]] [Sd[-0.5,17], Sd[0.5,17]]]
    end
end