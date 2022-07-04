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
            @test A[Block.(1), Block.(1)] ≈ [sqrt(2)/2, sqrt(2)/2]
            @test A[Block.(1), Block.(2)] ≈ [[0.6123724356957946,0.6123724356957946] [-0.3535533905932738,0.3535533905932738]]
            @test A[Block.(1), Block.(3)] ≈ [[-0.6123724356957946,0.6123724356957946] [-0.3535533905932738,-0.3535533905932738]]
        
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
        @test isapprox(A[Block.(1), Block.(2)], [[0.816497,0.816497] [0,0]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[-0.408248,0.408248] [0.707107,0.707107]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[-0.408248,-0.408248] [-0.707107,0.707107]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[0.816497,-0.816497] [0,0]], atol=1e-5)
    
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
        @test isapprox(A[Block.(1), Block.(1)], [0.707107,0.707107], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(2)], [[0,0] [0.612372,0.612372]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[0.189469,0.0898143] [-0.353553,0.353553]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[0,0] [-0.612372,0.612372]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[0.050768,0.0114079] [-0.353553,-0.353553]], atol=1e-5)       

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
        @test isapprox(A[Block.(1), Block.(1)], [ -0.816497,-0.730297], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(2)], [[0,0] [0.816497,0.816497]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(3)], [[-0.408248,-0.182574] [0,0]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(4)], [[0,0] [-0.408248,0.408248]], atol=1e-5)
        @test isapprox(A[Block.(1), Block.(5)], [[-0.10939,-0.02319] [0.707107,0.707107]], atol=1e-5)       
        @test isapprox(A[Block.(1), Block.(9)], [[-0.00785383,-0.000374129] [0,0]], atol=1e-5)  

        A = framematrix(x, Sd, 1, normtype=evaluate)
        @test A[Block.(1), Block.(1)] ≈ [Sd[-0.5,1],Sd[0.5,1]]
        @test A[Block.(1), Block.(2)] ≈ [[Sd[-0.5,2],Sd[0.5,2]] [Sd[-0.5,3], Sd[0.5,3]]]
        @test A[Block.(1), Block.(3)] ≈ [[Sd[-0.5,4],Sd[0.5,4]] [Sd[-0.5,5], Sd[0.5,5]]]
        @test A[Block.(1), Block.(4)] ≈ [[Sd[-0.5,6],Sd[0.5,6]] [Sd[-0.5,7], Sd[0.5,7]]]
        @test A[Block.(1), Block.(5)] ≈ [[Sd[-0.5,8],Sd[0.5,8]] [Sd[-0.5,9], Sd[0.5,9]]]
        @test A[Block.(1), Block.(9)] ≈ [[Sd[-0.5,16],Sd[0.5,16]] [Sd[-0.5,17], Sd[0.5,17]]]
    end
end