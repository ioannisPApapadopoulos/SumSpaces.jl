using Test, SumSpaces


"""
Test functions in extendedchebyshev.jl
"""

@testset "Chebyshev" begin
    
    @testset "basics" begin
        wU = ExtendedWeightedChebyshevU()
        @test wU == wU[:,1:∞]
        @test wU[:,1:∞] == wU
    end
    
    @testset "Evaluation" begin
        @testset "wU" begin
            wU = ExtendedWeightedChebyshevU()
            @test wU == wU[:,1:∞]
            @test wU[:,1:∞] == wU
        end
    end


end

