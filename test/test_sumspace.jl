using Test, SumSpaces, ClassicalOrthogonalPolynomials

"""
Test functions in sumspace.jl
"""

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

    @testset "primal" begin
        Sp = SumSpaceP()
        @test Sp[0.1,oneto(0)] == Float64[]
    end

end