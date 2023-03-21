using Test, SumSpaces
import SumSpaces: explicit_sum

@testset "explicit_sum" begin
    @testset "V_n" begin

        # x = 1.01, n=0, λ=1, μ=η=0, computing v_0. Actually value computed using Mathematica
        d = explicit_sum(1.01, 0, 1, 0, 0, f1v, f2v) - 0.802832403803213980022076067458878446381
        @test abs(d) < 1e-12
    end
end