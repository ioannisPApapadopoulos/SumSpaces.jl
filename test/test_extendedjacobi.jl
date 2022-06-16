using Test, ClassicalOrthogonalPolynomials
using SumSpaces

@testset "ExtendedJacobi" begin
    
    @testset "basics" begin
        a = 0.5; b = 0.5
        ewP = ExtendedWeightedJacobi(a,b)

        @test ewP == ewP[:,1:∞]
        @test ewP[:,1:∞] == ewP
    end

    @testset "Evaluation" begin
        @testset "Extended Weighted Jacobi" begin
            a = 0.5; b = 0.5
            ewP = ExtendedWeightedJacobi(a,b)
            wP = Weighted(Jacobi(a,b))
            @test ewP[0.1, 1:20] == wP[0.1, 1:20]
            @test ewP[1.1, 1:20] == zeros(20)
        end

        @testset "Extended Jacobi" begin
            a = 0.5; b = 0.5
            eP = ExtendedJacobi(a,b)
            P = Jacobi(a,b)

            @test eP[0.1, 1:20] == P[0.1, 1:20]
            # Not implemented yet
            # @test eP[1.1, 1:20] == ???
        end
    end

end