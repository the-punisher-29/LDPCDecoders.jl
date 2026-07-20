@testitem "Small-Set-Flip Decoder" begin

using Test
using SparseArrays
using Random
using StableRNGs
import LDPCDecoders: SmallSetFlipDecoder, decode!, reset!

    # Utility: create a simple cycle parity check matrix
    function create_cycle_matrix(n::Int)
        I_idx = Int[]
        J_idx = Int[]
        for j in 1:n
            push!(I_idx, j)
            push!(J_idx, mod(j, n) + 1)
            push!(I_idx, j)
            push!(J_idx, j)
        end
        V = fill(true, length(I_idx))
        return sparse(I_idx, J_idx, V, n, n)
    end

    @testset "Zero syndrome" begin
        H = create_cycle_matrix(8)
        decoder = SmallSetFlipDecoder(H, 50)
        syndrome = zeros(Bool, 8)
        guess, converged = decode!(decoder, syndrome)
        @test converged == true
        @test all(guess .== 0)
    end

    @testset "Single-bit errors on cycle matrix" begin
        for n in [4, 8, 16]
            H = create_cycle_matrix(n)
            decoder = SmallSetFlipDecoder(H, 50)

            for bit in 1:n
                error = zeros(Int, n)
                error[bit] = 1
                syndrome = Bool.(mod.(H * error, 2))
                guess, converged = decode!(decoder, syndrome)
                decoded_syndrome = Bool.(mod.(H * guess, 2))
                @test decoded_syndrome == syndrome
            end
        end
    end

    @testset "Random LDPC code" begin
        rng = StableRNG(42)
        H = LDPCDecoders.parity_check_matrix(100, 5, 4)
        decoder = SmallSetFlipDecoder(H, 100)

        # Low error rate should be decodable
        for _ in 1:10
            error = rand(rng, 100) .< 0.02
            syndrome = Bool.(mod.(H * error, 2))
            guess, converged = decode!(decoder, syndrome)
            decoded_syndrome = Bool.(mod.(H * guess, 2))
            @test decoded_syndrome == syndrome
        end
    end

    @testset "Subset size parameter t" begin
        H = create_cycle_matrix(8)

        # t=1 should behave like single-bit flip
        decoder_t1 = SmallSetFlipDecoder(H, 50; t=1)
        # t=2 allows pairs
        decoder_t2 = SmallSetFlipDecoder(H, 50; t=2)
        # t=0 (default) uses full support
        decoder_full = SmallSetFlipDecoder(H, 50)

        syndrome = Bool.(mod.(H * [1, 0, 0, 0, 0, 0, 0, 0], 2))

        for dec in [decoder_t1, decoder_t2, decoder_full]
            guess, converged = decode!(dec, syndrome)
            decoded_syndrome = Bool.(mod.(H * guess, 2))
            @test decoded_syndrome == syndrome
        end
    end

    @testset "Reset between decodings" begin
        H = create_cycle_matrix(8)
        decoder = SmallSetFlipDecoder(H, 50)

        # Decode twice to ensure reset works
        for _ in 1:3
            error = zeros(Int, 8)
            error[1] = 1
            syndrome = Bool.(mod.(H * error, 2))
            guess, converged = decode!(decoder, syndrome)
            decoded_syndrome = Bool.(mod.(H * guess, 2))
            @test decoded_syndrome == syndrome
        end
    end

    @testset "Integer syndrome input" begin
        H = create_cycle_matrix(8)
        decoder = SmallSetFlipDecoder(H, 50)
        error = [1, 0, 0, 0, 0, 0, 0, 0]
        # Pass Int syndrome (not Bool) — should work via !iszero()
        syndrome = (H * error) .% 2
        guess, converged = decode!(decoder, syndrome)
        decoded_syndrome = Bool.(mod.(H * guess, 2))
        @test decoded_syndrome == Bool.(syndrome)
    end

end
