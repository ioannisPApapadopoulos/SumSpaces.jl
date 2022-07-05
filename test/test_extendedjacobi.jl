using Test, ClassicalOrthogonalPolynomials
using SumSpaces

@testset "ExtendedJacobi" begin
    
    @testset "basics" begin
        a = 0.5; b = 0.5
        ewP = ExtendedWeightedJacobi(a,b)

        @test ewP == ewP[:,1:∞]
        @test ewP[:,1:∞] == ewP

        @test ewP == ExtendedWeightedJacobi(a,b)
        @test ewP != ExtendedWeightedJacobi(-a,-b)
        @test ewP != ExtendedWeightedJacobi(-a,b)
        @test axes(ewP[1:1,:]) === (oneto(1), oneto(∞))

        eP = ExtendedJacobi(a,b)
        @test eP == eP[:,1:∞]
        @test eP[:,1:∞] == eP
        @test eP == ExtendedJacobi(a,b)
        @test eP != ExtendedJacobi(-a,-b)
        @test eP != ExtendedJacobi(-a,b)
        @test axes(eP[1:1,:]) === (oneto(1), oneto(∞))

    end

    @testset "Evaluation" begin
        
        # Constant conversions for a = 0.5; b = 0.5
        wJ = n -> 2^(2(n-1)+1)/binomial(2(n-1)+2,n)
        
        @testset "Extended Weighted Jacobi" begin
            a = 0.5; b = 0.5
            ewP = ExtendedWeightedJacobi(a,b)
            wP = Weighted(Jacobi(a,b))
            # ewU = ExtendedWeightedChebyshevU()

            @test ewP[0.1, 1:20] == wP[0.1, 1:20]
            @test ewP[1.1, 1:20] == zeros(20)
            # @test ewP[0.1, 1:20] ≈ ewU[0.1, 3:22]
        end

        @testset "Extended Jacobi" begin
            a = 0.5; b = 0.5
            eP = ExtendedJacobi(a,b)
            P = Jacobi(a,b)
            eU = ExtendedChebyshevU()

            @test eP[0.1, 1:20] == P[0.1, 1:20]
            @test wJ.(1:20) .* eP[0.1, 1:20] ≈ eU[0.1, 3:22]
            # Not implemented yet
            # @test eP[1.1, 1:20] == ???
        end
    end

end