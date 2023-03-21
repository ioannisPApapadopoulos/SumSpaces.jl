using Test, ClassicalOrthogonalPolynomials
using SumSpaces
import ForwardDiff: derivative

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
        
        # Constant conversions for to extended ChebyshevU, a = 0.5; b = 0.5
        wPU = n -> 2^(2(n-1)+1)/binomial(2(n-1)+2,n)

        # Constant conversions for to extended ChebyshevT, a = -0.5; b = -0.5
        wPT = n -> 2^(2(n-1))/binomial(2(n-1),n-1)
        
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
            a = -0.3; b = -0.3
            eP = ExtendedJacobi(a,b)
            P = Jacobi(a,b)
            @test eP[0.1, 1:20] == P[0.1, 1:20]
            @test eP[1.1, 1:5] ≈ [0.7366510982212272, 0.28439670853378274, 0.14277057893441303, 0.07782569907567304, 0.044185715144731255]

            # Check special cases equal to extended ChebyshevT or ChebyshevU (modulo a constant)
            x = [-5., -3.1, -1.1, 0.1, 0.8, 1.3, 5.7]
            @test ExtendedJacobi(0.5,0.5)[x, 1:20] .* wPU.(1:20)' ≈ ExtendedChebyshevU()[x, 3:22]
            @test ExtendedJacobi(-0.5,-0.5)[x, 2:21] .* wPT.(2:21)' ≈ ExtendedChebyshevT()[x, 2:21]
        end

        @testset "derivative" begin
            xx = 3:0.1:3
            for s in [-2/3, -1/3, 1/3, 2/3]
                P = ExtendedJacobi(s,s)
                Q = ExtendedWeightedJacobi(s,s)

                x = axes(P,1)
                ∂P = Derivative(x) * P
                ∂Q = Derivative(x) * Q
                ∂p(x, n) = derivative(x->ExtendedJacobi{eltype(x)}(s,s)[x, n], x)
                ∂q(x, n) = derivative(x->ExtendedWeightedJacobi{eltype(x)}(s,s)[x, n], x)

                for n = 1:10
                    @test ∂p.(xx, n) ≈ ∂P[xx, n]
                    @test ∂q.(xx, n) ≈ ∂Q[xx, n]
                end
            end
        end
    end

end