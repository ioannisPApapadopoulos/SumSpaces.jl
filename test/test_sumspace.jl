using Test, ClassicalOrthogonalPolynomials, SumSpaces
import SumSpaces: affinetransform

@testset "sumspace" begin
    T̃ = ExtendedChebyshevT()
    W = ExtendedWeightedChebyshevU()

    @testset "basics" begin
        S = SumSpace(T̃, W)
        @test S.I ≈ [-1,1]

        a = [-5,3.]
        Sa = SumSpace{Float64}(T̃, W, a)
        @test Sa.I ≈ [-5,3]

        @test S != Sa
    end

    @testset "Evaluation" begin

        Sp = SumSpaceP()
        S = SumSpace(T̃, W)

        @test S[0.1,oneto(0)] == Float64[]
        @test S[0.1,oneto(1)] ≈ Sp[0.1, oneto(1)]
        @test S[0.1,oneto(3)] ≈ Sp[0.1,oneto(3)]
        @test S[0.8:0.1:1.2,oneto(10)] ≈ Sp[0.8:0.1:1.2,oneto(10)]
    end

    @testset "Interval Evaluation" begin
        S = SumSpace{Float64}(W, T̃[:, 2:∞], [-2., 1., 3., 5])
        x = 0.1
        for i = 1:3
            y = affinetransform(S.I[i],S.I[i+1], x)
            @test S[x, 2i-1:6:2i-1+30] ≈ W[y, 1:6]
            @test S[x, 2i:6:2i+30] ≈ T̃[y, 2:7]
        end

    end
end