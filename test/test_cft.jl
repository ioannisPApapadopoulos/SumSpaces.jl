using Test, SumSpaces

@testset "cft" begin

    @testset "supporter_functions" begin
        (x, uS) = supporter_functions(1, 1, 1, W=1., δ=0.5, s=[1.0], N=5)
        @test isapprox(x, [-6.28319,-3.14159,0.0,3.14159], atol=1e-5)
        for i = 1:4   
            @test size(uS[i]) == (1,) 
        end
        @test isapprox(real.(uS[1][1]), [0.0641088,0.248197,0.627191,0.0605033],atol=1e-5)
        @test isapprox(real.(uS[2][1]), [-0.031153,0.156541,-0.218847,-0.406541],atol=1e-5)
        @test isapprox(real.(uS[3][1]), [-0.0242268,-0.0726805,0.0242268,0.0726805],atol=1e-5)
        @test isapprox(real.(uS[4][1]), [-0.0176742,-0.0307795,0.127687,-0.0792332],atol=1e-5)
        @test isapprox(imag.(uS[1][1]), [-1.57021e-17,-3.03954e-17,0.0,7.40952e-18],atol=1e-5)
        @test isapprox(imag.(uS[2][1]), [-0.0956497,0.0956497,-0.0956497,0.0956497],atol=1e-5)
        @test isapprox(imag.(uS[3][1]), [ 0.0550063,-0.0550063,0.0550063,-0.0550063],atol=1e-5)
        @test isapprox(imag.(uS[4][1]), [4.32893e-18,3.7694e-18,0.0,-9.70326e-18],atol=1e-5)

        (x2, uS) = supporter_functions(0, 1, 1, W=1., δ=0.5, s=[1.0, 2.0], N=5, stabilise=true)
        @test x == x2
        for i = 1:4   
            @test size(uS[i]) == (2,) 
        end
        @test isapprox(real.(uS[1][1]), [-0.277935,0.277935,0.660534,-0.660534],atol=1e-5)
        @test isapprox(real.(uS[2][1]), [0.219235,0.219235,-0.719235,-0.719235],atol=1e-5)
        @test isapprox(real.(uS[3][1]), [6.00793e-9,6.00793e-9,-6.00793e-9,-6.00793e-9],atol=1e-5)
        @test isapprox(real.(uS[4][1]), [ -3.69574e-7,3.69574e-7,-3.81589e-7,3.81589e-7],atol=1e-5)
        @test isapprox(imag.(uS[1][1]), [ 6.80746e-17,-3.40373e-17,  0.0,  -8.08921e-17],atol=1e-5)
        @test isapprox(imag.(uS[2][1]), [-0.191299,0.191299,  -0.191299 , 0.191299],atol=1e-5)
        @test isapprox(imag.(uS[3][1]), [-3.75581e-7,3.75581e-7, -3.75581e-7,  3.75581e-7],atol=1e-5)
        @test isapprox(imag.(uS[4][1]), [9.05194e-23,-4.52597e-23,  0.0 , 4.67312e-23],atol=1e-5)
        @test isapprox(real.(uS[1][2]), [ -0.1288,0.1288 , 0.363418,  -0.363418],atol=1e-5)
        @test isapprox(real.(uS[2][2]), [ 0.121109,0.121109,  -0.371109,  -0.371109],atol=1e-5)
        @test isapprox(real.(uS[3][2]), [2.36065e-11, 2.36065e-11,  -2.36065e-11,  -2.36065e-11],atol=1e-5)
        @test isapprox(real.(uS[4][2]), [-1.47838e-9, 1.47838e-9,  -1.52559e-9,  1.52559e-9],atol=1e-5)
        @test isapprox(imag.(uS[1][2]), [ 3.1547e-17,-1.57735e-17,  0.0,  -4.45058e-17],atol=1e-5)
        @test isapprox(imag.(uS[2][2]), [ -0.117309, 0.117309,  -0.117309,  0.117309],atol=1e-5)
        @test isapprox(imag.(uS[3][2]), [ -1.50198e-9,1.50198e-9,  -1.50198e-9,  1.50198e-9],atol=1e-5)
        @test isapprox(imag.(uS[4][2]), [ 3.62098e-25,-1.81049e-25, 0.0,  1.86831e-25],atol=1e-5)
    end

    @testset "cifft" begin
        import SumSpaces: cifft

        # Check that the FFT approximated IFT of √(π)exp(-x²/4) is 
        # approximately exp(-x²).

        f = x -> sqrt(π).*exp.(-x.^2 ./ 4)
        W = 1e3; δ = 1e-3; ω=range(-W, W, step=δ); ω = ω[1:end-1]
        x = ifftshift((fftfreq(length(ω), 1/δ)) * 2 * pi) 

        IFT_f = cifft(f, ω, δ, W, x)
        @test isapprox(imag.(IFT_f), zeros(length(IFT_f)), atol=1e-12)
        @test isapprox(real.(IFT_f), exp.(-x.^2), atol=1e-13)
    end

    @testset "interpolate_supporter_functions" begin
        
        # One element
        s = [1.0]
        (x, uS) = supporter_functions(1, 1, 1, W=1., δ=0.5, s=s, N=5)
        I = [-1.,1.]
        uS2 = interpolate_supporter_functions(x, x, uS, I, s)
        for i = 1:4
            @test real.(uS[i][1]) == uS2[i][1].coefs
        end
        @test uS2[1][1](0.1) ≈ 0.6091524345778905
        
        # Two elements - same size
        I = [-3.,-1, 1.]
        s=[1.0]
        (x, uS) = supporter_functions(1, 1, 1, W=1., δ=0.5, s=s, N=5)
        uS2 = interpolate_supporter_functions(x, x, uS, I, s)
        for i = 1:4   
            @test size(uS[i]) == (1,) 
            @test size(uS2[i]) == (2,) 
        end
        for i = 1:4
            for j = 1:2
                @test real.(uS[i][1]) == uS2[i][j].coefs
            end
        end
        @test uS2[1][1](0.1) ≈ 0.24838806629590976
        @test uS2[1][2](0.1) ≈ 0.6091524345778905

        # Two elements - different size
        I = [-5.,-1, 1.]
        @test_throws BoundsError interpolate_supporter_functions(x, x, uS, I, s)
        s=[0.5, 1.0]
        (x, uS) = supporter_functions(1, 1, 1, W=1., δ=0.5, s=s, N=5)
        uS2 = interpolate_supporter_functions(x, x, uS, I, s)
        for i = 1:4
            for j = 1:2
                @test real.(uS[i][j]) == uS2[i][j].coefs
            end
        end
        @test uS2[1][1](0.1) ≈ 0.3005744301391276
        @test uS2[1][2](0.1) ≈ 0.6091524345778905
    end

    @testset "fft_supporter_functions" begin
        uS = fft_supporter_functions(1, 1, 1, N=5, W=1, δ=0.5)

        @test typeof(uS) == NTuple{4, Vector{Interpolations.GriddedInterpolation{Float64, 1, Float64, Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{Float64}}}}}
        for i = 1:4   
            @test size(uS[i]) == (1,) 
        end
        @test isapprox(uS[1][1].coefs, [0.0641088,0.248197,0.627191,0.0605033],atol=1e-5)
        @test isapprox(uS[2][1].coefs, [-0.031153,0.156541,-0.218847,-0.406541],atol=1e-5)
        @test isapprox(uS[3][1].coefs, [-0.0242268,-0.0726805,0.0242268,0.0726805],atol=1e-5)
        @test isapprox(uS[4][1].coefs, [-0.0176742,-0.0307795,0.127687,-0.0792332],atol=1e-5)
        
        @test uS[1][1](0.2) ≈ 0.5911142161637916

        uS = fft_supporter_functions(1, 1, 1, N=5, W=1, δ=0.5, stabilise=true)
        for i = 1:4   
            @test size(uS[i]) == (1,) 
        end
        @test isapprox(uS[1][1].coefs, [0.0641088,0.248197,0.627191,0.0605033],atol=1e-5)
        @test isapprox(uS[2][1].coefs, [-0.031153,0.156541,-0.218847,-0.406541],atol=1e-5)
        @test isapprox(uS[3][1].coefs, [1.20159e-9,3.60476e-9,-1.20159e-9,-3.60476e-9],atol=1e-5)
        @test isapprox(uS[4][1].coefs, [-1.84186e-7,1.86589e-7,-1.91395e-7,1.88992e-7],atol=1e-5)

        uS = fft_supporter_functions(1, 1, 1, N=5, W=1, δ=0.5, I = [-3.,-1,1], stabilise=true)
        for i = 1:4   
            @test size(uS[i]) == (2,) 
            @test uS[i][1].coefs == uS[i][2].coefs
        end
        @test uS[1][1](0.1) ≈ 0.24838806629590976
        @test uS[1][2](0.1) ≈ 0.6091524345778905

        uS = fft_supporter_functions(1, 1, 1, N=5, W=1, δ=0.5, I = [-5.,-1,1], stabilise=true)
        for i = 1:4    
            @test size(uS[i]) == (2,)
            @test uS[i][1].coefs != uS[i][2].coefs
        end
        @test uS[1][1](0.1) ≈ 0.3005744301391276
        @test uS[1][2](0.1) ≈ 0.6091524345778905

        uS = fft_supporter_functions(1, 1, 1, N=5, W=1, δ=0.5, I = [-5.,-1,1,2,6,10], stabilise=true)
        for i = 1:4
            @test size(uS[i]) == (5,)
        end
    end

    @testset "coefficient_supporter_functions" begin
        N = 101
        uS = fft_supporter_functions(1, 1, 1, N=N, W=1e2, δ=1e-2)
        Sp = SumSpaceP()
        M = 5001
        x = collocation_points(M, M)
        A = framematrix(x, Sp, N, normtype=evaluate) 
        cuS = coefficient_supporter_functions(A, x, uS, 2N+3, normtype=evaluate, tol=1e-15) 
        
        @test typeof(cuS) <: NTuple{4, Vector{Vector{Float64}}}
        xx = -2:0.01:0
        for i = 1:4
            @test norm(uS[i][1](xx) .- Sp[xx,1:length(cuS[1][1])]*cuS[i][1]) < 1e-2
        end
    end


    @testset "inverse_fourier_transform" begin

        # Check that the FFT approximated IFT of √(π)exp(-x²/4) is 
        # approximately exp(-x²).

        f = x -> sqrt(π).*exp.(-x.^2 ./ 4)
        W = 1e3; δ = 1e-3; ω=range(-W, W, step=δ); ω = ω[1:end-1]

        (x, IFT_f) = inverse_fourier_transform(f, ω)
        @test isapprox(imag.(IFT_f), zeros(length(IFT_f)), atol=1e-12)
        @test isapprox(real.(IFT_f), exp.(-x.^2), atol=1e-13)
    end

end