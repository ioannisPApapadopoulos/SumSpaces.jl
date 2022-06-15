using Test, SumSpaces, ClassicalOrthogonalPolynomials
import ClassicalOrthogonalPolynomials: sqrtx2

"""
Test functions in extendedchebyshev.jl
"""

@testset "ExtendedChebyshev" begin
    
    @testset "basics" begin
        wU = ExtendedWeightedChebyshevU()
        @test wU == wU[:,1:∞]
        @test wU[:,1:∞] == wU
    end
    
    @testset "Evaluation" begin
        @testset "wU" begin
            wU = ExtendedWeightedChebyshevU()
            @test @inferred(wU[0.1,oneto(0)]) == Float64[]
            @test @inferred(wU[0.1,oneto(1)]) ≈ [sqrt(1-0.1^2)]
            @test @inferred(wU[0.1,oneto(2)]) ≈ [sqrt(1-0.1^2),2*0.1*sqrt(1-0.1^2)]
            for N = 1:10
                @test @inferred(wU[0.1,oneto(N)]) ≈ @inferred(wU[0.1,1:N]) ≈ sqrt(1-0.1^2).*ChebyshevU()[0.1,1:N]
                @test @inferred(wU[0.1,N]) ≈ sqrt(1-0.1^2).*ChebyshevU()[0.1,N]
                @test @inferred(wU[1.1,oneto(N)])≈ @inferred(wU[1.1,1:N]) ≈ zeros(1:N)
            end
            
            @test axes(wU[1:1,:]) === (oneto(1), oneto(∞))
        end

        @testset "wT" begin
            wT = ExtendedWeightedChebyshevT()
            @test @inferred(wT[0.1,oneto(0)]) == Float64[]
            @test @inferred(wT[0.1,oneto(1)]) ≈ [1/sqrt(1-0.1^2)]
            @test @inferred(wT[0.1,oneto(2)]) ≈ [1/sqrt(1-0.1^2),0.1/sqrt(1-0.1^2)]
            for N = 1:10
                @test @inferred(wT[0.1,oneto(N)]) ≈ @inferred(wT[0.1,1:N]) ≈ 1/sqrt(1-0.1^2).*ChebyshevT()[0.1,1:N]
                @test @inferred(wT[0.1,N]) ≈ 1/sqrt(1-0.1^2).*ChebyshevT()[0.1,N]
                @test @inferred(wT[1.1,oneto(N)])≈ @inferred(wT[1.1,1:N]) ≈ zeros(1:N)
            end
            
            @test axes(wT[1:1,:]) === (oneto(1), oneto(∞))
        end

        
        @testset "eT" begin
            eT = ExtendedChebyshevT()
            T = ChebyshevT()

            @test @inferred(eT[0.1,oneto(0)]) == Float64[]
            @test @inferred(eT[0.1,oneto(1)]) ≈ T[0.1,oneto(1)]
            @test @inferred(eT[0.1,oneto(2)]) ≈ T[0.1,oneto(2)]
            for N = 1:10
                @test @inferred(eT[0.1,oneto(N)]) ≈ @inferred(eT[0.1,1:N]) ≈ T[0.1,1:N]
                @test @inferred(eT[0.1,N]) ≈ T[0.1,N]
                @test @inferred(eT[1.1,oneto(N)])≈ @inferred(eT[1.1,1:N]) ≈ (1.1-sqrt(1.1^2-1)).^(0:N-1)
                @test @inferred(eT[-1.1,oneto(N)])≈ @inferred(eT[-1.1,1:N]) ≈ (-1.1+sqrt(1.1^2-1)).^(0:N-1)
            end
            
            @test axes(eT[1:1,:]) === (oneto(1), oneto(∞))
        end

        @testset "eU" begin
            eU = ExtendedChebyshevU()
            U = ChebyshevU()
            @test @inferred(eU[0.1,oneto(0)]) == Float64[]
            @test @inferred(eU[0.1,3:3]) ≈ U[0.1,oneto(1)]
            @test @inferred(eU[0.1,3:4]) ≈ U[0.1,oneto(2)]

            @test @inferred(eU[0.1,1]) ≈ 0
            @test @inferred(eU[1.1,1]) ≈ - 1.1 / sqrt(1.1^2 - 1)
            @test @inferred(eU[0.1,2]) ≈ 0
            @test @inferred(eU[1.1,2]) ≈ - 1 / sqrt(1.1^2 - 1)

            η = x->inv.(x .+ sqrtx2.(x))
            ξo = (x,j) -> 2 .* sum(η(x).^(2:2:j-3)) .+ 1 .- sign(x) .* x ./ sqrt.(x.^2 .- 1)
            ξe = (x,j) -> 2 .* sum(η(x).^(1:2:j-3)) .- sign.(x) ./ sqrt.(x.^2 .- 1)

            for N = 3:2:11
                @test @inferred(eU[0.1,3:N+2]) ≈ U[0.1,1:N]
                @test @inferred(eU[0.1,N+2]) ≈ U[0.1,N]
                
                @test @inferred(eU[1.1,N]) ≈ ξo(1.1, N)
                @test @inferred(eU[-1.1,N])≈ ξo(-1.1, N)
            end
            
            for N = 4:2:12
                @test @inferred(eU[0.1,3:N+2]) ≈ U[0.1,1:N]
                @test @inferred(eU[0.1,N+2]) ≈ U[0.1,N]
                
                @test @inferred(eU[1.1,N]) ≈ ξe(1.1, N)
                @test @inferred(eU[-1.1,N])≈ ξe(-1.1, N)
            end

            @test axes(eU[1:1,:]) === (oneto(1), oneto(∞))
        end

    end


end