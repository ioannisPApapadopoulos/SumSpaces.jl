using Test, ClassicalOrthogonalPolynomials
using SumSpaces, StaticArrays, MultivariateOrthogonalPolynomials
import ForwardDiff: derivative

function _coeff_zernike(s::T) where T
    d = 2
    k = mortar(Base.OneTo.(oneto(∞))) 
    ℓ = mortar(Fill.(oneto(∞),oneto(∞))) .- 1
    m = k .- isodd.(k) .* iseven.(ℓ) .- iseven.(k) .* isodd.(ℓ) 
    n = ℓ .- m
    T(4)^s*gamma.(1+s .+ n) .* gamma.((d.+2*(s.+ℓ)) ./ 2) ./ (factorial.(n).*gamma.((d.+s*ℓ) ./ 2))
end

@testset "ExtendedZernike" begin
    @testset "basics" begin
        a = 0.; b = 0.5
        ewP = ExtendedWeightedZernike(a,b)

        @test ewP == ewP[:,1:∞]
        @test ewP[:,1:∞] == ewP

        @test ewP == ExtendedWeightedZernike(a,b)

        eP = ExtendedZernike(a,b)
        @test eP == eP[:,1:∞]
        @test eP[:,1:∞] == eP
        @test eP == ExtendedZernike(a,b)
    end

    @testset "evaluation" begin
        for s in [-2/3, -1/2, 1/3, 2/3]
            T̃ = ExtendedZernike(0., s)
            Z = Zernike(0., s)

            W = ExtendedWeightedZernike(0., s)
            wZ = Weighted(Zernike(0., s))

            for xy in SVector.(-1+2*eps():0.1:1-2*eps(), 0.)
                @test T̃[xy, Block.(1:5)] ≈ Z[xy, Block.(1:5)] .* _coeff_zernike(s)[Block.(1:5)]
                @test W[xy, Block.(1:5)] == wZ[xy, Block.(1:5)]
            end
        end
    end

end